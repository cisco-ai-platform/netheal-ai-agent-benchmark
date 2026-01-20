import random
from typing import Any, Dict, List, Tuple

import numpy as np

from netheal.environment.env import NetworkTroubleshootingEnv


def _fingerprint_env(env: NetworkTroubleshootingEnv, info: Dict[str, Any]) -> Dict[str, Any]:
    network = env.network

    nodes: List[Tuple[str, Any, str, str]] = []
    for device_id in sorted(network.get_all_devices()):
        data = network.get_device_info(device_id)
        device_type = data.get("device_type")
        if hasattr(device_type, "value"):
            device_type = device_type.value
        nodes.append(
            (
                device_id,
                device_type,
                data.get("status"),
                data.get("ip_address"),
            )
        )

    edges: List[Tuple[str, str, str, float, float]] = []
    for source, dest in sorted(network.get_all_connections()):
        data = network.get_connection_info(source, dest)
        edges.append(
            (
                source,
                dest,
                data.get("status"),
                round(float(data.get("bandwidth", 0.0)), 6),
                round(float(data.get("latency", 0.0)), 6),
            )
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "ground_truth": info.get("ground_truth_fault"),
        "user_hint": info.get("user_hint"),
    }


def _numpy_state_equal(state_a: Tuple[Any, ...], state_b: Tuple[Any, ...]) -> bool:
    if len(state_a) != len(state_b):
        return False
    if state_a[0] != state_b[0]:
        return False
    if not np.array_equal(state_a[1], state_b[1]):
        return False
    return state_a[2:] == state_b[2:]


def test_seeded_reset_is_deterministic() -> None:
    env_kwargs = dict(
        min_devices=3,
        max_devices=8,
        max_episode_steps=50,
        topology_types=["star", "mesh", "hierarchical"],
        enable_user_hints=True,
        hint_provider_mode="heuristic",
    )

    env_a = NetworkTroubleshootingEnv(**env_kwargs)
    _, info_a = env_a.reset(seed=1234)
    fingerprint_a = _fingerprint_env(env_a, info_a)

    env_b = NetworkTroubleshootingEnv(**env_kwargs)
    _, info_b = env_b.reset(seed=1234)
    fingerprint_b = _fingerprint_env(env_b, info_b)

    assert fingerprint_a == fingerprint_b


def test_reset_does_not_mutate_global_rng_state() -> None:
    env = NetworkTroubleshootingEnv(
        min_devices=3,
        max_devices=8,
        max_episode_steps=50,
        topology_types=["star", "mesh", "hierarchical"],
        enable_user_hints=True,
        hint_provider_mode="heuristic",
    )

    py_state_before = random.getstate()
    np_state_before = np.random.get_state()

    env.reset(seed=1234)

    assert random.getstate() == py_state_before
    assert _numpy_state_equal(np.random.get_state(), np_state_before)
