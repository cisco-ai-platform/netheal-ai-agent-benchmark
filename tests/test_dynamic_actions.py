"""Regression tests for dynamic action space rebuild and action unlocking.

These tests ensure that after a topology scan on star and hierarchical
networks, diagnostic and/or diagnosis actions become available, preventing
unsolvable episodes where only scan actions remain.
"""

import pytest

from netheal.environment.env import NetworkTroubleshootingEnv
from netheal.environment.actions import ActionCategory


def _find_scan_action_id(env: NetworkTroubleshootingEnv) -> int:
    for i in range(env.action_space.n):
        spec = env.action_space_manager.get_action_spec(i)
        if spec and spec.category == ActionCategory.TOPOLOGY_DISCOVERY and spec.action_type.value == "scan_network":
            return i
    return -1


def _has_unlocked_actions(env: NetworkTroubleshootingEnv) -> bool:
    """Return True if any valid action is DIAGNOSTIC or DIAGNOSIS."""
    valid_ids = env.get_valid_actions()
    for aid in valid_ids:
        spec = env.action_space_manager.get_action_spec(aid)
        if spec and spec.category in (ActionCategory.DIAGNOSTIC, ActionCategory.DIAGNOSIS):
            return True
    return False


@pytest.mark.parametrize("topology_type", ["star", "hierarchical"])
def test_actions_unlock_after_scan(topology_type: str):
    env = NetworkTroubleshootingEnv(
        max_devices=4,
        max_episode_steps=6,
        topology_types=[topology_type],
        render_mode=None,
    )

    obs, info = env.reset(seed=123)

    # Initially, it's acceptable if only scan is valid
    scan_id = _find_scan_action_id(env)
    assert scan_id >= 0, "scan_network action must exist"

    # Execute scan to populate discovered devices based on real device IDs
    obs, reward, terminated, truncated, step_info = env.step(scan_id)

    # After scan, some diagnostic or diagnosis actions should be valid
    assert _has_unlocked_actions(env), (
        f"After scan on {topology_type} topology, no diagnostic/diagnosis actions unlocked."
    )

    env.close()
