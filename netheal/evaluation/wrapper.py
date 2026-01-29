# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Gymnasium wrapper that collects NetHeal episode traces and computes metrics.

This wrapper keeps the base environment untouched while emitting rich metrics
for benchmarks and competitions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import hashlib
import json
import time
import numpy as np
import gymnasium as gym

from .metrics import (
    ActionRecord,
    CompetitionEvaluator,
    EpisodeMetrics,
    EpisodeTrace,
    compute_episode_metrics,
)


class MetricsCollectorWrapper(gym.Wrapper):
    """Wrapper that records episode traces and computes metrics at termination."""

    def __init__(
        self,
        env: gym.Env,
        evaluator: Optional[CompetitionEvaluator] = None,
        skip_evaluator: bool = False,
    ) -> None:
        super().__init__(env)
        self.evaluator = evaluator or CompetitionEvaluator()
        self._trace: Optional[EpisodeTrace] = None
        self._last_metrics: Optional[EpisodeMetrics] = None
        self._last_trace: Optional[EpisodeTrace] = None
        self._skip_evaluator = skip_evaluator

    # ------------------------------------------------------------------ #
    # Gym API                                                            #
    # ------------------------------------------------------------------ #
    def reset(self, *args: Any, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        seed = kwargs.get("seed")
        if self._trace and self._trace.end_time is None:
            # Episode was interrupted; finalize with best effort.
            self._finalize_trace(
                final_observation=self._trace.final_observation,
                final_info=self._trace.final_info or {},
                terminated=False,
                truncated=True,
            )

        observation, info = self.env.reset(*args, **kwargs)
        self._start_new_trace(observation, info, seed=seed)
        return observation, info

    def step(
        self, action: int
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        if self._trace is None:
            raise RuntimeError("MetricsCollectorWrapper requires reset() before step().")

        observation, reward, terminated, truncated, info = self.env.step(action)
        self._record_action(
            action_id=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

        if terminated or truncated:
            self._finalize_trace(
                final_observation=observation,
                final_info=info,
                terminated=terminated,
                truncated=truncated,
                skip_evaluator=self._skip_evaluator,
            )

        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Public helpers                                                     #
    # ------------------------------------------------------------------ #
    @property
    def last_episode_metrics(self) -> Optional[EpisodeMetrics]:
        """Return metrics for the most recently completed episode."""
        return self._last_metrics

    @property
    def last_episode_trace(self) -> Optional[EpisodeTrace]:
        """Return the trace for the most recently completed episode."""
        return self._last_trace

    def add_metrics_to_evaluator(self) -> None:
        """Manually add the last episode metrics to the evaluator.

        Use this when skip_evaluator=True to add metrics after retries complete.
        """
        if self._last_metrics is not None:
            self.evaluator.add_episode_metrics(self._last_metrics)

    def record_tool_error(
        self,
        tool_name: str,
        message: str,
        tool_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an invalid tool call for metrics accounting."""
        if self._trace is None:
            return
        self._trace.tool_error_count += 1

    @property
    def action_specs(self):
        """Pass-through to underlying environment's action specs.
        
        Returns a list of ActionSpecs indexed by action_id.
        """
        manager = getattr(self.env, "action_space_manager", None)
        if manager is None:
            return []
        # action_map is a dict {action_id: ActionSpec}, convert to list
        action_map = getattr(manager, "action_map", {})
        if not action_map:
            return []
        # Build list indexed by action_id
        max_id = max(action_map.keys()) + 1
        specs = [None] * max_id
        for idx, spec in action_map.items():
            specs[idx] = spec
        return specs

    @property
    def current_step(self) -> int:
        """Pass-through to underlying environment's current step count."""
        return getattr(self.env, "step_count", 0)

    @property
    def max_episode_steps(self) -> int:
        """Pass-through to underlying environment's max episode steps."""
        return getattr(self.env, "max_episode_steps", 25)

    # ------------------------------------------------------------------ #
    # Internal utilities                                                 #
    # ------------------------------------------------------------------ #
    def _start_new_trace(
        self,
        observation: Any,
        info: Dict[str, Any],
        seed: Optional[int] = None,
    ) -> None:
        network_devices = []
        network_edges = []
        network_size = info.get("network_size", 0)

        network = getattr(self.env, "network", None)
        if network is not None:
            try:
                network_devices = list(network.get_all_devices())
            except Exception:
                network_devices = []
            try:
                network_edges = list(network.get_all_connections())
            except Exception:
                network_edges = []
            if not network_size:
                network_size = len(network_devices)

        scenario_fingerprint = self._build_scenario_fingerprint(
            network, info.get("ground_truth_fault")
        )

        self._trace = EpisodeTrace(
            ground_truth=info.get("ground_truth_fault"),
            network_size=network_size,
            max_episode_steps=getattr(self.env, "max_episode_steps", 1),
            network_devices=network_devices,
            network_edges=network_edges,
            start_time=time.time(),
            seed=seed,
            scenario_fingerprint=scenario_fingerprint,
        )
        self._trace.final_observation = observation
        self._trace.final_info = info
        self._last_metrics = None
        self._last_trace = None

    def _record_action(
        self,
        action_id: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> None:
        if self._trace is None:
            return

        action_spec = info.get("action_spec") or {}
        parameters = action_spec.get("parameters") or {}

        record = ActionRecord(
            step=len(self._trace.actions) + 1,
            action_id=action_id,
            category=action_spec.get("category"),
            action_type=action_spec.get("action_type"),
            parameters=dict(parameters),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=dict(info),
            timestamp=time.time(),
        )

        self._trace.actions.append(record)
        self._trace.total_reward += reward
        self._trace.final_info = info

    def _finalize_trace(
        self,
        final_observation: Any,
        final_info: Dict[str, Any],
        terminated: bool,
        truncated: bool,
        skip_evaluator: bool = False,
    ) -> None:
        if self._trace is None:
            return

        self._trace.final_observation = final_observation
        self._trace.final_info = dict(final_info)
        self._trace.end_time = time.time()
        self._trace.termination_reason = "terminated" if terminated else "truncated"

        discovery_matrix = getattr(
            getattr(self.env, "observation", None), "discovery_matrix", None
        )
        if discovery_matrix is not None:
            try:
                self._trace.discovered_nodes = len(discovery_matrix.get_discovered_devices())
            except Exception:
                self._trace.discovered_nodes = 0
            try:
                adjacency = np.asarray(discovery_matrix.adjacency)
                self._trace.discovered_edges = int(np.count_nonzero(adjacency))
            except Exception:
                self._trace.discovered_edges = 0

        metrics = compute_episode_metrics(self._trace)
        self._last_metrics = metrics
        self._last_trace = self._trace
        if not skip_evaluator:
            self.evaluator.add_episode_metrics(metrics)
        self._trace = None

    @staticmethod
    def _build_scenario_fingerprint(
        network: Any,
        ground_truth: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        if network is None:
            return None

        nodes = []
        for device_id in sorted(network.get_all_devices()):
            data = network.get_device_info(device_id)
            device_type = data.get("device_type")
            if hasattr(device_type, "value"):
                device_type = device_type.value
            nodes.append(
                {
                    "id": device_id,
                    "type": device_type,
                    "status": data.get("status"),
                    "ip": data.get("ip_address"),
                }
            )

        edges = []
        for source, dest in sorted(network.get_all_connections()):
            data = network.get_connection_info(source, dest)
            edges.append(
                {
                    "source": source,
                    "dest": dest,
                    "status": data.get("status"),
                    "bandwidth": round(float(data.get("bandwidth", 0.0)), 6),
                    "latency": round(float(data.get("latency", 0.0)), 6),
                }
            )

        payload = {
            "nodes": nodes,
            "edges": edges,
            "ground_truth": ground_truth or {},
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


__all__ = ["MetricsCollectorWrapper"]

