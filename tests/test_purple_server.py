# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the NetHeal purple agent A2A server.
"""
import asyncio
import socket
import threading
import time

import pytest
import uvicorn
import httpx

from netheal.aaa.purple_server import app, set_config


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def running_purple_server():
    """Start purple agent server in background thread."""
    try:
        port = _pick_free_port()
    except PermissionError:
        pytest.skip("Socket binding not permitted")

    set_config(f"http://127.0.0.1:{port}", "dummy")

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
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
            raise RuntimeError("Timed out waiting for server to start")
        time.sleep(0.05)

    base_url = f"http://127.0.0.1:{port}"

    try:
        yield base_url
    finally:
        uvicorn_server.should_exit = True
        thread.join(timeout=5)


class TestPurpleAgentCard:
    """Test purple agent A2A card endpoint."""

    def test_agent_card_endpoint(self, running_purple_server):
        """Test GET /.well-known/agent.json returns valid card."""
        base_url = running_purple_server

        response = httpx.get(f"{base_url}/.well-known/agent.json")
        assert response.status_code == 200

        card = response.json()
        assert "name" in card
        assert card["name"] == "netheal-purple-agent"
        assert "capabilities" in card
        assert card["capabilities"]["accepts_tasks"] is True
        assert "url" in card

    def test_agent_card_has_solver_type(self, running_purple_server):
        """Test that agent card includes solver type."""
        base_url = running_purple_server

        response = httpx.get(f"{base_url}/.well-known/agent.json")
        card = response.json()

        assert "capabilities" in card
        assert "solver_type" in card["capabilities"]


class TestPurpleTaskEndpoints:
    """Test purple agent task endpoints."""

    def test_create_task_endpoint(self, running_purple_server):
        """Test POST /tasks creates a task."""
        base_url = running_purple_server

        response = httpx.post(
            f"{base_url}/tasks",
            json={
                "task_id": "test-task-1",
                "episode_start": {
                    "episode_index": 0,
                    "total_episodes": 1,
                    "mcp_server_url": "http://fake-server:9999/mcp",
                    "hint": "Test hint",
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["task_id"] == "test-task-1"
        assert "status" in data

    def test_get_task_endpoint(self, running_purple_server):
        """Test GET /tasks/{id} returns task status."""
        base_url = running_purple_server

        create_response = httpx.post(
            f"{base_url}/tasks",
            json={
                "task_id": "test-task-2",
                "episode_start": {
                    "episode_index": 0,
                    "total_episodes": 1,
                    "mcp_server_url": "http://fake-server:9999/mcp",
                },
            },
        )
        assert create_response.status_code == 200

        get_response = httpx.get(f"{base_url}/tasks/test-task-2")
        assert get_response.status_code == 200

        data = get_response.json()
        assert "task_id" in data
        assert "status" in data
        assert "created_at" in data

    def test_get_nonexistent_task_returns_404(self, running_purple_server):
        """Test GET /tasks/{id} returns 404 for nonexistent task."""
        base_url = running_purple_server

        response = httpx.get(f"{base_url}/tasks/nonexistent-task")
        assert response.status_code == 404

    def test_duplicate_task_id_returns_409(self, running_purple_server):
        """Test POST /tasks with duplicate ID returns 409."""
        base_url = running_purple_server

        payload = {
            "task_id": "duplicate-task",
            "episode_start": {
                "episode_index": 0,
                "mcp_server_url": "http://fake:9999/mcp",
            },
        }

        response1 = httpx.post(f"{base_url}/tasks", json=payload)
        assert response1.status_code == 200

        response2 = httpx.post(f"{base_url}/tasks", json=payload)
        assert response2.status_code == 409


class TestPurpleServerConfiguration:
    """Test purple server configuration."""

    def test_set_config_changes_card_url(self):
        """Test that set_config updates the card URL."""
        from netheal.aaa.purple_server import _CARD_URL

        set_config("http://test-url:1234", "gpt")

        from netheal.aaa.purple_server import _CARD_URL as updated_url

        assert updated_url == "http://test-url:1234"
