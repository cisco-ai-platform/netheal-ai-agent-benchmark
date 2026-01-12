# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""API tests for the NetHeal FastAPI web demo.

These tests validate the basic episode lifecycle over HTTP:
- reset -> returns initial state with observation/info
- actions -> returns valid action IDs and descriptions
- step -> advances the environment and returns updated state

Run with: pytest tests/test_web_api.py -q
"""
from __future__ import annotations

from typing import Any, Dict
from fastapi.testclient import TestClient

from webapp.backend.app.main import app


client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_state_before_reset_returns_400():
    r = client.get("/api/env/state")
    assert r.status_code == 400


def test_reset_actions_step_flow():
    # Reset environment
    reset_payload: Dict[str, Any] = {
        "seed": 123,
        "max_devices": 6,
        "max_episode_steps": 10,
        "topology_types": None,
        "enable_user_hints": True,
        "hint_provider_mode": "heuristic",
        "user_context": {"access_point": "Guest-WiFi"},
    }
    r = client.post("/api/env/reset", json=reset_payload)
    assert r.status_code == 200, r.text

    state = r.json()
    assert "observation" in state and "info" in state
    assert isinstance(state["observation"], dict)
    assert "discovery_matrix" in state["observation"]
    assert "recent_diagnostics" in state["observation"]

    # Fetch valid actions
    r = client.get("/api/env/actions")
    assert r.status_code == 200
    actions = r.json()
    valid = actions.get("valid_actions", [])
    assert isinstance(valid, list) and len(valid) > 0

    # Take the first valid action
    r = client.post("/api/env/step", json={"action_id": valid[0]})
    assert r.status_code == 200, r.text

    state2 = r.json()
    assert "last_reward" in state2
    assert "terminated" in state2 and "truncated" in state2
    assert isinstance(state2.get("info", {}), dict)


def test_step_with_invalid_action_returns_penalty_not_500():
    # Reset first
    r = client.post("/api/env/reset", json={})
    assert r.status_code == 200

    # Intentionally choose a very large action id
    r = client.post("/api/env/step", json={"action_id": 999999})
    assert r.status_code == 200
    payload = r.json()
    # Should include a reward (penalty) and no crash
    assert "last_reward" in payload
    # Info should not be missing
    assert isinstance(payload.get("info", {}), dict)
