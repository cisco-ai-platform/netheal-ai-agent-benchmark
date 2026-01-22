# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Scenario snapshot helpers for NetHeal."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4
import json
import random

from gymnasium import spaces

from netheal.environment.env import NetworkTroubleshootingEnv
from netheal.hints.provider import get_default_hint_provider
from netheal.environment.observation import StructuredObservation
from netheal.faults.injector import FaultInfo, FaultType, FaultInjector
from netheal.network.graph import NetworkGraph, DeviceType
from netheal.tools.simulator import ToolSimulator


SNAPSHOT_VERSION = "2.0.0"


@dataclass
class SnapshotRecord:
    """Container for snapshot metadata and payload."""

    snapshot_id: str
    payload: Dict[str, Any]


def export_snapshot(
    env: NetworkTroubleshootingEnv,
    metadata: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Export the current environment state as a snapshot dict."""
    snapshot_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    scaling_factor = getattr(
        getattr(env, "reward_calculator", None), "scaling_factor", None
    )
    config: Dict[str, Any] = {
        "min_devices": env.min_devices,
        "max_devices": env.max_devices,
        "max_episode_steps": env.max_episode_steps,
        "topology_types": list(env.topology_types),
        "fault_types": [ft.value for ft in env.fault_types],
        "enable_user_hints": env.enable_user_hints,
        "hint_provider_mode": env.hint_provider_mode,
        "reward_scaling_factor": scaling_factor,
    }
    if metadata:
        config.update(metadata)

    snapshot = {
        "version": SNAPSHOT_VERSION,
        "snapshot_id": snapshot_id,
        "created_at": created_at,
        "seed": seed,
        "rng_state": _serialize_rng_state(_get_rng_state(env)),
        "config": config,
        "network": _export_network(env.network),
        "fault": _export_fault(env.ground_truth_fault),
        "expected_solution": _expected_solution(env.ground_truth_fault),
        "user_hint": env.user_hint,
    }
    return snapshot


def apply_snapshot(env: NetworkTroubleshootingEnv, snapshot: Dict[str, Any]) -> None:
    """Apply a snapshot payload to an existing environment."""
    _validate_snapshot(snapshot)

    config = snapshot.get("config", {})
    env.min_devices = config.get("min_devices", env.min_devices)
    env.max_devices = config.get("max_devices", env.max_devices)
    env.max_episode_steps = config.get("max_episode_steps", env.max_episode_steps)
    env.topology_types = config.get("topology_types", env.topology_types)
    env.enable_user_hints = config.get("enable_user_hints", env.enable_user_hints)
    env.hint_provider_mode = config.get("hint_provider_mode", env.hint_provider_mode)
    scaling_factor = config.get("reward_scaling_factor")
    if scaling_factor is not None and hasattr(env, "reward_calculator"):
        env.reward_calculator.scaling_factor = float(scaling_factor)

    fault_types = config.get("fault_types")
    if fault_types:
        env.fault_types = [FaultType(ft) for ft in fault_types]

    rng_state = snapshot.get("rng_state")
    if rng_state is not None:
        env._rng = random.Random()
        env._rng.setstate(_deserialize_rng_state(rng_state))
    else:
        seed = snapshot.get("seed")
        env._rng = random.Random(seed) if seed is not None else random.Random()

    network = _import_network(snapshot.get("network", {}))
    env.network = network
    env.tool_simulator = ToolSimulator(network, rng=env._rng)
    env.fault_injector = FaultInjector(network, rng=env._rng)

    fault_info = _import_fault(snapshot.get("fault", {}))
    env.ground_truth_fault = fault_info
    if fault_info:
        _apply_fault_to_network(network, fault_info)
        env.fault_injector.active_faults = [fault_info]

    env.observation = StructuredObservation(env.max_devices)
    env.observation.episode_step = 0
    env.observation.max_episode_steps = env.max_episode_steps
    env.previous_observation = None
    env.step_count = 0
    env.episode_done = False
    env.episode_start_time = datetime.now(timezone.utc).timestamp()

    # Rebuild action space based on imported device IDs
    device_ids = network.get_all_devices()
    env.action_space_manager.rebuild_for_network(device_ids)
    env.action_space = spaces.Discrete(env.action_space_manager.total_actions)

    # Restore stored hint or regenerate if missing
    env.user_hint = snapshot.get("user_hint")
    if env.user_hint is None and env.enable_user_hints:
        try:
            provider = env.hint_provider or get_default_hint_provider(env.hint_provider_mode)
            gt = {
                "type": fault_info.fault_type.value if fault_info else None,
                "location": fault_info.location if fault_info else None,
            }
            context = {
                "ground_truth": gt,
                "network_size": len(network.get_all_devices()),
                "topology_types": env.topology_types,
                "user_context": env.user_context,
            }
            env.user_hint = provider.generate_hint(context) if provider else None
        except Exception:
            env.user_hint = None


def create_env_from_snapshot(snapshot: Dict[str, Any]) -> NetworkTroubleshootingEnv:
    """Create a new environment from a snapshot payload."""
    _validate_snapshot(snapshot)
    config = snapshot.get("config", {})
    env = NetworkTroubleshootingEnv(
        min_devices=config.get("min_devices", 3),
        max_devices=config.get("max_devices", 15),
        max_episode_steps=config.get("max_episode_steps", 100),
        topology_types=config.get("topology_types"),
        fault_types=[FaultType(ft) for ft in config.get("fault_types", [])] or None,
        enable_user_hints=config.get("enable_user_hints", True),
        hint_provider_mode=config.get("hint_provider_mode", "auto"),
        reward_scaling_factor=config.get("reward_scaling_factor", 10.0),
    )
    apply_snapshot(env, snapshot)
    return env


def load_snapshot_episodes(path: Path) -> List[Dict[str, Any]]:
    """Load snapshot episodes from a JSON/JSONL file or directory."""
    path = Path(path)
    snapshots: List[Dict[str, Any]] = []

    if path.is_dir():
        for file_path in sorted(path.glob("*.json")):
            snapshots.append(_load_snapshot_file(file_path))
        return snapshots

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                snapshot = json.loads(line)
                _validate_snapshot(snapshot)
                snapshots.append(snapshot)
        return snapshots

    snapshot = _load_snapshot_file(path)
    snapshots.append(snapshot)
    return snapshots


def _load_snapshot_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        snapshot = json.load(handle)
    _validate_snapshot(snapshot)
    return snapshot


def _validate_snapshot(snapshot: Dict[str, Any]) -> None:
    version = snapshot.get("version")
    if version != SNAPSHOT_VERSION:
        raise ValueError(f"Unsupported snapshot version: {version}")

    for key in ("snapshot_id", "created_at", "config", "network", "fault", "expected_solution"):
        if key not in snapshot:
            raise ValueError(f"Snapshot missing required field: {key}")


def _export_network(network: Optional[NetworkGraph]) -> Dict[str, Any]:
    if network is None:
        return {}

    nodes = []
    for node_id, data in network.graph.nodes(data=True):
        nodes.append(
            {
                "id": node_id,
                "type": str(data.get("device_type", "host")),
                "ip_address": data.get("ip_address"),
                "status": data.get("status", "up"),
            }
        )

    edges = []
    for src, dst, data in network.graph.edges(data=True):
        edges.append(
            {
                "source": src,
                "target": dst,
                "bandwidth": float(data.get("bandwidth", 100.0)),
                "latency": float(data.get("latency", 1.0)),
                "status": data.get("status", "up"),
            }
        )

    return {"nodes": nodes, "edges": edges}


def _export_fault(fault: Optional[FaultInfo]) -> Dict[str, Any]:
    if fault is None:
        return {}
    return {
        "type": fault.fault_type.value if hasattr(fault.fault_type, "value") else str(fault.fault_type),
        "location": fault.location,
        "details": fault.details or {},
    }


def _expected_solution(fault: Optional[FaultInfo]) -> Dict[str, Any]:
    if fault is None:
        return {"fault_type": None, "location": None}
    return {
        "fault_type": fault.fault_type.value if hasattr(fault.fault_type, "value") else str(fault.fault_type),
        "location": fault.location,
    }


def _import_network(network_data: Dict[str, Any]) -> NetworkGraph:
    network = NetworkGraph()
    nodes_data = network_data.get("nodes", [])
    for node in nodes_data:
        device_type_str = node.get("type", "host")
        try:
            device_type = DeviceType(device_type_str)
        except (ValueError, KeyError):
            device_type = DeviceType.HOST
        network.add_device(
            device_id=node["id"],
            device_type=device_type,
            ip_address=node.get("ip_address"),
            status=node.get("status", "up"),
        )

    edges_data = network_data.get("edges", [])
    for edge in edges_data:
        network.add_connection(
            source=edge["source"],
            destination=edge["target"],
            bandwidth=edge.get("bandwidth", 100.0),
            latency=edge.get("latency", 1.0),
            status=edge.get("status", "up"),
            bidirectional=True,
        )

    return network


def _import_fault(fault_data: Dict[str, Any]) -> Optional[FaultInfo]:
    if not fault_data:
        return None
    fault_type_str = fault_data.get("type", "")
    try:
        fault_type = FaultType(fault_type_str)
    except (ValueError, KeyError):
        fault_type = FaultType.LINK_FAILURE
    return FaultInfo(
        fault_type=fault_type,
        location=fault_data.get("location", ""),
        details=fault_data.get("details", {}),
    )


def _apply_fault_to_network(network: NetworkGraph, fault: FaultInfo) -> None:
    fault_type = fault.fault_type
    details = fault.details or {}

    if fault_type == FaultType.LINK_FAILURE:
        source = details.get("source")
        destination = details.get("destination")
        if not source or not destination:
            source, destination = _parse_link_location(fault.location)
        if source and destination and network.graph.has_edge(source, destination):
            network.set_connection_status(source, destination, "down")
            if network.graph.has_edge(destination, source):
                network.set_connection_status(destination, source, "down")

    elif fault_type == FaultType.DEVICE_FAILURE:
        device_id = details.get("device_id") or fault.location
        if device_id and network.graph.has_node(device_id):
            network.set_device_status(device_id, "down")

    elif fault_type == FaultType.MISCONFIGURATION:
        device_id = details.get("device_id") or fault.location
        blocked_destination = details.get("blocked_destination")
        if device_id and blocked_destination and network.graph.has_edge(device_id, blocked_destination):
            network.set_connection_status(device_id, blocked_destination, "down")

    elif fault_type == FaultType.PERFORMANCE_DEGRADATION:
        source = details.get("source")
        destination = details.get("destination")
        new_latency = details.get("new_latency")
        if not source or not destination:
            source, destination = _parse_link_location(fault.location)
        if source and destination and new_latency is not None:
            network.set_connection_latency(source, destination, float(new_latency))
            if network.graph.has_edge(destination, source):
                network.set_connection_latency(destination, source, float(new_latency))


def _parse_link_location(location: str) -> tuple[Optional[str], Optional[str]]:
    if "->" not in location:
        return (None, None)
    parts = [part.strip() for part in location.split("->") if part.strip()]
    if len(parts) != 2:
        return (None, None)
    return (parts[0], parts[1])


def _get_rng_state(env: NetworkTroubleshootingEnv) -> Optional[Any]:
    rng = getattr(env, "_rng", None)
    if rng is None:
        return None
    try:
        return rng.getstate()
    except Exception:
        return None


def _serialize_rng_state(state: Any) -> Any:
    if isinstance(state, tuple):
        return [_serialize_rng_state(item) for item in state]
    if isinstance(state, list):
        return [_serialize_rng_state(item) for item in state]
    return state


def _deserialize_rng_state(state: Any) -> Any:
    if isinstance(state, list):
        return tuple(_deserialize_rng_state(item) for item in state)
    return state
