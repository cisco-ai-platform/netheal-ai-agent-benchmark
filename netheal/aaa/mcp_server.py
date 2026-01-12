"""
MCP Server for NetHeal environment tools.

Exposes diagnostic tools to solver agents via the MCP protocol. The green
agent creates one server per assessment episode and shares its URL with
purple agents over the A2A channel.

Interfaces:
    MCP Protocol (/mcp): Native MCP tool discovery and invocation
    HTTP Fallback: REST endpoints for non-MCP clients
        GET  /tools              - List available tools
        GET  /state              - Current observation
        GET  /actions            - Valid actions
        POST /tools/{tool_name}  - Execute a tool
"""
from __future__ import annotations

import asyncio
import logging
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

try:
    import numpy as np
except ImportError:
    np = None

from netheal.environment.actions import (
    ActionCategory,
    ActionSpec,
    DiagnosticAction,
    TopologyAction,
)
from netheal.evaluation.wrapper import MetricsCollectorWrapper
from netheal.faults import FaultType

LOGGER = logging.getLogger(__name__)


@dataclass
class EpisodeRuntime:
    """Per-episode state for the MCP server."""

    env: MetricsCollectorWrapper
    observation: Dict[str, Any]
    info: Dict[str, Any]


class NetHealMCPServer:
    """
    MCP server exposing NetHeal diagnostic tools.

    Maps tool calls to environment actions, maintaining consistency with
    the MetricsCollectorWrapper for proper trace recording.

    Supports both MCP protocol (/mcp) and HTTP endpoints (/tools/*).
    """

    def __init__(
        self,
        runtime: EpisodeRuntime,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        log_level: str = "warning",
    ) -> None:
        self.runtime = runtime
        self.host = host
        self.port = port or self._reserve_ephemeral_port(host)
        self.log_level = log_level

        self._mcp = FastMCP(name="netheal-mcp")
        self._uvicorn: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = asyncio.Lock()
        self._diagnosis_submitted = False

        self._register_mcp_tools()
        self._register_http_routes()

    @property
    def base_url(self) -> str:
        """MCP protocol endpoint."""
        return f"http://{self.host}:{self.port}/mcp"

    @property
    def http_helper_url(self) -> str:
        """HTTP helper endpoint base URL."""
        return f"http://{self.host}:{self.port}"

    def start(self, timeout: float = 5.0) -> None:
        """Start server in background thread."""
        if self._thread and self._thread.is_alive():
            return

        app = self._mcp.streamable_http_app()
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level=self.log_level,
        )
        self._uvicorn = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._uvicorn.run, daemon=True)
        self._thread.start()
        self._wait_until_serving(timeout)
        LOGGER.info("MCP server started on %s", self.base_url)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the server."""
        if self._uvicorn is not None:
            self._uvicorn.should_exit = True
        if self._thread:
            self._thread.join(timeout=timeout)
        LOGGER.info("MCP server stopped")

    def _register_mcp_tools(self) -> None:
        """Register diagnostic tools with FastMCP."""

        @self._mcp.tool(name="get_state", description="Get current observation and episode info.")
        async def tool_get_state() -> Dict[str, Any]:
            return self._state_snapshot()

        @self._mcp.tool(name="list_actions", description="List valid actions for discovered devices.")
        async def tool_list_actions() -> Dict[str, Any]:
            return self._valid_actions_payload()

        @self._mcp.tool(name="scan_network", description="Discover network devices.")
        async def tool_scan_network() -> Dict[str, Any]:
            return await self._execute_by_predicate(
                predicate=lambda spec: spec.category == ActionCategory.TOPOLOGY_DISCOVERY
                and spec.action_type == TopologyAction.SCAN_NETWORK,
                error_message="Network scan action unavailable.",
            )

        @self._mcp.tool(name="discover_neighbors", description="Find neighbors of a device.")
        async def tool_discover_neighbors(device: str) -> Dict[str, Any]:
            return await self._execute_by_predicate(
                predicate=lambda spec: spec.category == ActionCategory.TOPOLOGY_DISCOVERY
                and spec.action_type == TopologyAction.DISCOVER_NEIGHBORS
                and spec.parameters.get("device") == device,
                error_message=f"No discover_neighbors action for '{device}'.",
            )

        @self._mcp.tool(name="ping", description="Test connectivity between two devices.")
        async def tool_ping(source: str, destination: str) -> Dict[str, Any]:
            return await self._diagnostic_action(DiagnosticAction.PING, source, destination)

        @self._mcp.tool(name="traceroute", description="Trace path between two devices.")
        async def tool_traceroute(source: str, destination: str) -> Dict[str, Any]:
            return await self._diagnostic_action(DiagnosticAction.TRACEROUTE, source, destination)

        @self._mcp.tool(name="check_status", description="Check device operational status.")
        async def tool_check_status(device: str) -> Dict[str, Any]:
            return await self._execute_by_predicate(
                predicate=lambda spec: spec.category == ActionCategory.DIAGNOSTIC
                and spec.action_type == DiagnosticAction.CHECK_STATUS
                and spec.parameters.get("device") == device,
                error_message=f"No check_status action for '{device}'.",
            )

        @self._mcp.tool(name="check_interfaces", description="Inspect device network interfaces.")
        async def tool_check_interfaces(device: str) -> Dict[str, Any]:
            return await self._execute_by_predicate(
                predicate=lambda spec: spec.category == ActionCategory.DIAGNOSTIC
                and spec.action_type == DiagnosticAction.CHECK_INTERFACES
                and spec.parameters.get("device") == device,
                error_message=f"No check_interfaces action for '{device}'.",
            )

        @self._mcp.tool(
            name="submit_diagnosis",
            description="Submit final fault diagnosis. Ends the episode.",
        )
        async def tool_submit_diagnosis(fault_type: str, location: str) -> Dict[str, Any]:
            if self._diagnosis_submitted:
                return {"error": "Diagnosis already submitted."}

            try:
                fault_enum = FaultType(fault_type)
            except ValueError:
                valid = [ft.value for ft in FaultType]
                return {"error": f"Invalid fault_type '{fault_type}'. Valid: {valid}"}

            result = await self._execute_by_predicate(
                predicate=lambda spec: spec.category == ActionCategory.DIAGNOSIS
                and spec.action_type == fault_enum
                and spec.parameters.get("location") == location,
                error_message=f"No diagnosis action for {fault_type} at {location}.",
            )

            if "error" not in result:
                self._diagnosis_submitted = True

            return result

    def _register_http_routes(self) -> None:
        """Register HTTP fallback routes."""

        @self._mcp.custom_route("/tools", methods=["GET"], name="list_tools")
        async def http_list_tools(request: Request) -> JSONResponse:
            tools = {
                "tools": [
                    {
                        "name": "get_state",
                        "description": "Get current observation and episode info.",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    },
                    {
                        "name": "list_actions",
                        "description": "List valid actions for discovered devices.",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    },
                    {
                        "name": "scan_network",
                        "description": "Discover network devices.",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    },
                    {
                        "name": "discover_neighbors",
                        "description": "Find neighbors of a device.",
                        "parameters": {
                            "type": "object",
                            "properties": {"device": {"type": "string", "description": "Device ID"}},
                            "required": ["device"],
                        },
                    },
                    {
                        "name": "ping",
                        "description": "Test connectivity between two devices.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string", "description": "Source device ID"},
                                "destination": {"type": "string", "description": "Destination device ID"},
                            },
                            "required": ["source", "destination"],
                        },
                    },
                    {
                        "name": "traceroute",
                        "description": "Trace path between two devices.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string", "description": "Source device ID"},
                                "destination": {"type": "string", "description": "Destination device ID"},
                            },
                            "required": ["source", "destination"],
                        },
                    },
                    {
                        "name": "check_status",
                        "description": "Check device operational status.",
                        "parameters": {
                            "type": "object",
                            "properties": {"device": {"type": "string", "description": "Device ID"}},
                            "required": ["device"],
                        },
                    },
                    {
                        "name": "check_interfaces",
                        "description": "Inspect device network interfaces.",
                        "parameters": {
                            "type": "object",
                            "properties": {"device": {"type": "string", "description": "Device ID"}},
                            "required": ["device"],
                        },
                    },
                    {
                        "name": "submit_diagnosis",
                        "description": "Submit final fault diagnosis. Ends the episode.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "fault_type": {
                                    "type": "string",
                                    "enum": ["device_failure", "link_failure", "misconfiguration", "performance_degradation"],
                                },
                                "location": {"type": "string", "description": "Fault location"},
                            },
                            "required": ["fault_type", "location"],
                        },
                    },
                ]
            }
            return JSONResponse(tools)

        @self._mcp.custom_route("/state", methods=["GET"], name="get_state")
        async def http_state(request: Request) -> JSONResponse:
            return JSONResponse(self._state_snapshot())

        @self._mcp.custom_route("/actions", methods=["GET"], name="get_actions")
        async def http_actions(request: Request) -> JSONResponse:
            return JSONResponse(self._valid_actions_payload())

        @self._mcp.custom_route("/tools/scan_network", methods=["POST"], name="scan_network")
        async def http_scan_network(request: Request) -> JSONResponse:
            result = await self._execute_by_predicate(
                predicate=lambda spec: spec.category == ActionCategory.TOPOLOGY_DISCOVERY
                and spec.action_type == TopologyAction.SCAN_NETWORK,
                error_message="Network scan action unavailable.",
            )
            return JSONResponse(result)

        @self._mcp.custom_route("/tools/discover_neighbors", methods=["POST"], name="discover_neighbors")
        async def http_discover_neighbors(request: Request) -> JSONResponse:
            device = request.query_params.get("device", "")
            result = await self._execute_by_predicate(
                predicate=lambda spec: spec.category == ActionCategory.TOPOLOGY_DISCOVERY
                and spec.action_type == TopologyAction.DISCOVER_NEIGHBORS
                and spec.parameters.get("device") == device,
                error_message=f"No discover_neighbors action for '{device}'.",
            )
            return JSONResponse(result)

        @self._mcp.custom_route("/tools/ping", methods=["POST"], name="ping")
        async def http_ping(request: Request) -> JSONResponse:
            source = request.query_params.get("source", "")
            destination = request.query_params.get("destination", "")
            result = await self._diagnostic_action(DiagnosticAction.PING, source, destination)
            return JSONResponse(result)

        @self._mcp.custom_route("/tools/traceroute", methods=["POST"], name="traceroute")
        async def http_traceroute(request: Request) -> JSONResponse:
            source = request.query_params.get("source", "")
            destination = request.query_params.get("destination", "")
            result = await self._diagnostic_action(DiagnosticAction.TRACEROUTE, source, destination)
            return JSONResponse(result)

        @self._mcp.custom_route("/tools/check_status", methods=["POST"], name="check_status")
        async def http_check_status(request: Request) -> JSONResponse:
            device = request.query_params.get("device", "")
            result = await self._execute_by_predicate(
                predicate=lambda spec: spec.category == ActionCategory.DIAGNOSTIC
                and spec.action_type == DiagnosticAction.CHECK_STATUS
                and spec.parameters.get("device") == device,
                error_message=f"No check_status action for '{device}'.",
            )
            return JSONResponse(result)

        @self._mcp.custom_route("/tools/check_interfaces", methods=["POST"], name="check_interfaces")
        async def http_check_interfaces(request: Request) -> JSONResponse:
            device = request.query_params.get("device", "")
            result = await self._execute_by_predicate(
                predicate=lambda spec: spec.category == ActionCategory.DIAGNOSTIC
                and spec.action_type == DiagnosticAction.CHECK_INTERFACES
                and spec.parameters.get("device") == device,
                error_message=f"No check_interfaces action for '{device}'.",
            )
            return JSONResponse(result)

        @self._mcp.custom_route("/tools/submit_diagnosis", methods=["POST"], name="submit_diagnosis")
        async def http_submit_diagnosis(request: Request) -> JSONResponse:
            try:
                body = await request.json()
                fault_type = body.get("fault_type", "")
                location = body.get("location", "")
            except Exception:
                fault_type = request.query_params.get("fault_type", "")
                location = request.query_params.get("location", "")

            if self._diagnosis_submitted:
                return JSONResponse({"error": "Diagnosis already submitted."})

            try:
                fault_enum = FaultType(fault_type)
            except ValueError:
                valid = [ft.value for ft in FaultType]
                return JSONResponse({"error": f"Invalid fault_type '{fault_type}'. Valid: {valid}"})

            result = await self._execute_by_predicate(
                predicate=lambda spec: spec.category == ActionCategory.DIAGNOSIS
                and spec.action_type == fault_enum
                and spec.parameters.get("location") == location,
                error_message=f"No diagnosis action for {fault_type} at {location}.",
            )

            if "error" not in result:
                self._diagnosis_submitted = True

            return JSONResponse(result)

    def _state_snapshot(self) -> Dict[str, Any]:
        """Build current state as JSON-safe dict."""
        info = self.runtime.info or {}
        env = self.runtime.env

        current_step = getattr(env, "current_step", None)
        if current_step is None:
            current_step = info.get("episode_step", 0)

        max_steps = getattr(env, "max_episode_steps", None)
        if max_steps is None and hasattr(env, "env"):
            max_steps = getattr(env.env, "max_episode_steps", 25)

        remaining_steps = (max_steps - current_step) if max_steps else None

        return {
            "observation": _serialize(self.runtime.observation),
            "info": _serialize(info),
            "diagnosis_submitted": self._diagnosis_submitted,
            "step_budget": {
                "current_step": current_step,
                "max_steps": max_steps,
                "remaining_steps": remaining_steps,
                "warning": "LOW STEPS - submit diagnosis soon!" if remaining_steps and remaining_steps <= 5 else None,
            },
        }

    def _valid_actions_payload(self) -> Dict[str, Any]:
        """Build valid actions as JSON-safe dict."""
        env = self.runtime.env
        valid_indices = self.runtime.info.get("valid_actions", list(range(env.action_space.n)))
        specs: List[ActionSpec] = env.action_specs

        valid_actions = []
        for idx in valid_indices:
            if 0 <= idx < len(specs):
                spec = specs[idx]
                valid_actions.append({
                    "index": idx,
                    "category": spec.category.value if hasattr(spec.category, "value") else str(spec.category),
                    "action_type": spec.action_type.value if hasattr(spec.action_type, "value") else str(spec.action_type),
                    "parameters": spec.parameters,
                    "description": spec.description,
                })

        return {"valid_actions": valid_actions, "count": len(valid_actions)}

    async def _execute_by_predicate(
        self,
        predicate: Callable[[ActionSpec], bool],
        error_message: str,
    ) -> Dict[str, Any]:
        """Execute first action matching the predicate."""
        env = self.runtime.env
        valid_indices = self.runtime.info.get("valid_actions", list(range(env.action_space.n)))
        specs = env.action_specs

        for idx in valid_indices:
            if 0 <= idx < len(specs) and predicate(specs[idx]):
                return await self._step_env(idx)

        return {"error": error_message}

    async def _diagnostic_action(
        self,
        action_type: DiagnosticAction,
        source: str,
        destination: str,
    ) -> Dict[str, Any]:
        """Execute a source-destination diagnostic action."""
        return await self._execute_by_predicate(
            predicate=lambda spec: spec.category == ActionCategory.DIAGNOSTIC
            and spec.action_type == action_type
            and spec.parameters.get("source") == source
            and spec.parameters.get("destination") == destination,
            error_message=f"No {action_type.value} action for {source} -> {destination}.",
        )

    async def _step_env(self, action_idx: int) -> Dict[str, Any]:
        """Step environment and update runtime state."""
        obs, reward, terminated, truncated, info = self.runtime.env.step(action_idx)
        self.runtime.observation = obs
        self.runtime.info = info
        return {
            "observation": _serialize(obs),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": _serialize(info),
        }

    def _wait_until_serving(self, timeout: float) -> None:
        """Wait for server to accept connections."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with socket.create_connection((self.host, self.port), timeout=0.5):
                    return
            except OSError:
                time.sleep(0.1)
        raise TimeoutError(f"MCP server did not start within {timeout}s")

    @staticmethod
    def _reserve_ephemeral_port(host: str) -> int:
        """Reserve an ephemeral port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
            return sock.getsockname()[1]


def _serialize(obj: Any) -> Any:
    """Convert numpy/dataclass types to JSON-serializable Python types."""
    import dataclasses
    from enum import Enum

    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    if np is not None and isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, Enum):
        return obj.value
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _serialize(v) for k, v in dataclasses.asdict(obj).items()}
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return {k: _serialize(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return obj


__all__ = ["NetHealMCPServer", "EpisodeRuntime"]
