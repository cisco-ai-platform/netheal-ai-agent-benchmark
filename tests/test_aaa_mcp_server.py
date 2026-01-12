# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the NetHeal MCP server.

Tests both MCP protocol and HTTP helper endpoints.
"""
import asyncio
import socket
import threading
import time

import pytest
import uvicorn
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from netheal.aaa.mcp_server import EpisodeRuntime, NetHealMCPServer
from netheal.environment.actions import (
    ActionCategory,
    ActionSpec,
    DiagnosticAction,
    TopologyAction,
)
from netheal.faults import FaultType


class DummyDiscoveryMatrix:
    def __init__(self) -> None:
        self._devices = ["device_a", "device_b"]

    def get_discovered_devices(self):
        return list(self._devices)


class DummyActionSpace:
    def __init__(self, action_map):
        self.action_map = action_map
        self.n = len(action_map)

    def get_valid_actions(self, _devices):
        return list(self.action_map.keys())


class DummyEnv:
    def __init__(self, action_map):
        self.action_space = DummyActionSpace(action_map)
        self.action_space_manager = DummyActionSpace(action_map)
        self.observation = type("Obs", (), {"discovery_matrix": DummyDiscoveryMatrix()})()
        self.step_calls = []
        self.network = type("Net", (), {"get_all_devices": lambda self: ["device_a", "device_b"]})()
        self.action_specs = list(action_map.values())

    def step(self, action_id: int):
        self.step_calls.append(action_id)
        spec = self.action_specs[action_id]
        info = {
            "action_spec": spec.to_dict() if hasattr(spec, "to_dict") else {"description": spec.description},
            "valid_actions": list(range(len(self.action_specs))),
        }
        observation = {"state": action_id}
        reward = 1.0
        terminated = False
        truncated = False
        return observation, reward, terminated, truncated, info


class DummyWrapper:
    def __init__(self, env: DummyEnv):
        self.env = env
        self.action_space = env.action_space
        self.action_specs = env.action_specs

    def step(self, action_id: int):
        return self.env.step(action_id)


_ACTION_MAP = {
    0: ActionSpec(
        category=ActionCategory.TOPOLOGY_DISCOVERY,
        action_type=TopologyAction.SCAN_NETWORK,
        parameters={},
        description="scan network",
    ),
    1: ActionSpec(
        category=ActionCategory.DIAGNOSTIC,
        action_type=DiagnosticAction.PING,
        parameters={"source": "device_a", "destination": "device_b"},
        description="ping a->b",
    ),
    2: ActionSpec(
        category=ActionCategory.DIAGNOSTIC,
        action_type=DiagnosticAction.CHECK_STATUS,
        parameters={"device": "device_a"},
        description="check status device_a",
    ),
    3: ActionSpec(
        category=ActionCategory.DIAGNOSIS,
        action_type=FaultType.DEVICE_FAILURE,
        parameters={"location": "device_a"},
        description="diagnose device",
    ),
}


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def running_server():
    """Start an MCP server in a background thread."""
    try:
        port = _pick_free_port()
    except PermissionError:
        pytest.skip("Socket binding not permitted")

    dummy_env = DummyEnv(_ACTION_MAP)
    wrapper = DummyWrapper(dummy_env)
    runtime = EpisodeRuntime(
        env=wrapper,
        observation={"state": "initial"},
        info={"valid_actions": list(range(len(_ACTION_MAP)))},
    )

    try:
        server = NetHealMCPServer(runtime, port=port, log_level="error")
    except PermissionError:
        pytest.skip("Socket binding not permitted")

    app = server._mcp.streamable_http_app()
    config = uvicorn.Config(app, host=server.host, port=server.port, log_level="error")
    uvicorn_server = uvicorn.Server(config)
    thread = threading.Thread(target=uvicorn_server.run, daemon=True)

    try:
        thread.start()
    except PermissionError:
        pytest.skip("Socket binding not permitted")

    deadline = time.time() + 5
    while not uvicorn_server.started:
        if time.time() > deadline:
            uvicorn_server.should_exit = True
            thread.join(timeout=1)
            raise RuntimeError("Timed out waiting for MCP server to start")
        time.sleep(0.05)

    try:
        yield server, dummy_env
    finally:
        uvicorn_server.should_exit = True
        thread.join(timeout=5)


class TestMCPProtocol:
    """Test MCP protocol functionality."""

    def test_mcp_list_tools(self, running_server):
        """Test tool discovery via MCP protocol."""
        server, _ = running_server

        async def _test():
            async with streamablehttp_client(server.base_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()

                    tool_names = [t.name for t in tools.tools]
                    assert "get_state" in tool_names
                    assert "scan_network" in tool_names
                    assert "ping" in tool_names
                    assert "submit_diagnosis" in tool_names
                    assert len(tool_names) == 9

        asyncio.run(_test())

    def test_mcp_call_scan_network(self, running_server):
        """Test calling scan_network via MCP."""
        server, dummy_env = running_server

        async def _test():
            async with streamablehttp_client(server.base_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("scan_network", {})
                    assert result.content is not None

        asyncio.run(_test())
        assert 0 in dummy_env.step_calls

    def test_mcp_call_ping(self, running_server):
        """Test calling ping via MCP."""
        server, dummy_env = running_server

        async def _test():
            async with streamablehttp_client(server.base_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        "ping",
                        {"source": "device_a", "destination": "device_b"},
                    )
                    assert result.content is not None

        asyncio.run(_test())
        assert 1 in dummy_env.step_calls

    def test_mcp_call_submit_diagnosis(self, running_server):
        """Test submitting diagnosis via MCP."""
        server, dummy_env = running_server

        async def _test():
            async with streamablehttp_client(server.base_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        "submit_diagnosis",
                        {"fault_type": "device_failure", "location": "device_a"},
                    )
                    assert result.content is not None

        asyncio.run(_test())
        assert server._diagnosis_submitted is True
        assert 3 in dummy_env.step_calls


class TestHTTPEndpoints:
    """Test HTTP helper endpoints."""

    def test_http_list_tools(self, running_server):
        """Test GET /tools endpoint."""
        server, _ = running_server

        response = httpx.get(f"{server.http_helper_url}/tools")
        assert response.status_code == 200

        data = response.json()
        assert "tools" in data
        tool_names = [t["name"] for t in data["tools"]]
        assert "scan_network" in tool_names
        assert "submit_diagnosis" in tool_names

    def test_http_get_state(self, running_server):
        """Test GET /state endpoint."""
        server, _ = running_server

        response = httpx.get(f"{server.http_helper_url}/state")
        assert response.status_code == 200

        data = response.json()
        assert "observation" in data
        assert "info" in data
        assert "diagnosis_submitted" in data

    def test_http_get_actions(self, running_server):
        """Test GET /actions endpoint."""
        server, _ = running_server

        response = httpx.get(f"{server.http_helper_url}/actions")
        assert response.status_code == 200

        data = response.json()
        assert "valid_actions" in data
        assert "count" in data

    def test_http_scan_network(self, running_server):
        """Test POST /tools/scan_network endpoint."""
        server, dummy_env = running_server

        response = httpx.post(f"{server.http_helper_url}/tools/scan_network")
        assert response.status_code == 200
        assert 0 in dummy_env.step_calls

    def test_http_ping(self, running_server):
        """Test POST /tools/ping endpoint."""
        server, dummy_env = running_server

        response = httpx.post(
            f"{server.http_helper_url}/tools/ping",
            params={"source": "device_a", "destination": "device_b"},
        )
        assert response.status_code == 200
        assert 1 in dummy_env.step_calls

    def test_http_check_status(self, running_server):
        """Test POST /tools/check_status endpoint."""
        server, dummy_env = running_server

        response = httpx.post(
            f"{server.http_helper_url}/tools/check_status",
            params={"device": "device_a"},
        )
        assert response.status_code == 200
        assert 2 in dummy_env.step_calls

    def test_http_submit_diagnosis(self, running_server):
        """Test POST /tools/submit_diagnosis endpoint."""
        server, dummy_env = running_server

        response = httpx.post(
            f"{server.http_helper_url}/tools/submit_diagnosis",
            params={"fault_type": "device_failure", "location": "device_a"},
        )
        assert response.status_code == 200
        assert server._diagnosis_submitted is True

    def test_http_submit_diagnosis_twice_fails(self, running_server):
        """Test that submitting diagnosis twice returns error."""
        server, _ = running_server

        response1 = httpx.post(
            f"{server.http_helper_url}/tools/submit_diagnosis",
            params={"fault_type": "device_failure", "location": "device_a"},
        )
        assert response1.status_code == 200

        response2 = httpx.post(
            f"{server.http_helper_url}/tools/submit_diagnosis",
            params={"fault_type": "device_failure", "location": "device_a"},
        )
        assert response2.status_code == 200
        data = response2.json()
        assert "error" in data
        assert "already submitted" in data["error"].lower()


class TestServerLifecycle:
    """Test server start/stop functionality."""

    def test_server_start_stop(self):
        """Test that server starts and stops cleanly."""
        try:
            port = _pick_free_port()
        except PermissionError:
            pytest.skip("Socket binding not permitted")

        dummy_env = DummyEnv(_ACTION_MAP)
        wrapper = DummyWrapper(dummy_env)
        runtime = EpisodeRuntime(
            env=wrapper,
            observation={},
            info={"valid_actions": list(range(len(_ACTION_MAP)))},
        )

        server = NetHealMCPServer(runtime, port=port)

        server.start(timeout=5.0)

        response = httpx.get(f"{server.http_helper_url}/tools", timeout=5.0)
        assert response.status_code == 200

        server.stop(timeout=5.0)

        with pytest.raises(httpx.ConnectError):
            httpx.get(f"{server.http_helper_url}/tools", timeout=1.0)

    def test_base_url_format(self):
        """Test that base_url and http_helper_url are formatted correctly."""
        dummy_env = DummyEnv(_ACTION_MAP)
        wrapper = DummyWrapper(dummy_env)
        runtime = EpisodeRuntime(
            env=wrapper,
            observation={},
            info={"valid_actions": []},
        )

        server = NetHealMCPServer(runtime, host="127.0.0.1", port=9999)

        assert server.base_url == "http://127.0.0.1:9999/mcp"
        assert server.http_helper_url == "http://127.0.0.1:9999"

