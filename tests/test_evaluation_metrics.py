# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import types
import numpy as np
import gymnasium as gym
import pytest

from netheal.evaluation.metrics import (
    ActionRecord,
    EpisodeTrace,
    EpisodeMetrics,
    CompetitionEvaluator,
    compute_episode_metrics,
)
from netheal.evaluation.wrapper import MetricsCollectorWrapper


def _make_action(
    step,
    category,
    action_type,
    parameters,
    reward=0.0,
    terminated=False,
    truncated=False,
    info=None,
):
    return ActionRecord(
        step=step,
        action_id=step,
        category=category,
        action_type=action_type,
        parameters=parameters,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=info or {},
        timestamp=step,
    )


def test_compute_episode_metrics_basic():
    trace = EpisodeTrace(
        ground_truth={"type": "device_failure", "location": "device_a"},
        network_size=3,
        max_episode_steps=10,
        network_devices=["device_a", "device_b", "device_c"],
        network_edges=[("device_a", "device_b")],
        start_time=0.0,
    )

    trace.discovered_nodes = 3
    trace.discovered_edges = 1
    trace.total_reward = 5.0
    trace.start_time = 0.0
    trace.end_time = 5.0
    trace.final_info = {"reward_breakdown": {"diagnosis_reward": 10.0}}

    trace.actions = [
        _make_action(
            1,
            "diagnostic",
            "ping",
            {"source": "device_a", "destination": "device_b"},
            reward=-0.1,
        ),
        _make_action(
            2,
            "diagnostic",
            "check_status",
            {"device": "device_c"},
            reward=-0.1,
        ),
        _make_action(
            3,
            "diagnosis",
            "device_failure",
            {"location": "device_a"},
            reward=10.0,
            terminated=True,
        ),
    ]

    metrics = compute_episode_metrics(trace)

    assert metrics.diagnosis_success is True
    assert metrics.network_size == 3
    assert metrics.steps == 3
    # 3 actions × DEFAULT_ACTION_COST (1.0) = 3.0 (no actual ToolResult provided)
    assert metrics.tool_cost == pytest.approx(3.0)
    assert metrics.evidence_sufficiency == pytest.approx(0.5)
    assert metrics.redundancy_count == 0
    assert metrics.topology_coverage == pytest.approx(1.0)
    assert metrics.composite_episode_score > 0.0


def test_competition_evaluator_summary():
    evaluator = CompetitionEvaluator()
    evaluator.add_episode_metrics(
        EpisodeMetrics(
            diagnosis_success=True,
            network_size=3,
            steps=3,
            normalized_steps=0.3,
            total_reward=10.0,
            tool_cost=3.0,
            tool_cost_normalized=0.2,
            topology_coverage=0.8,
            node_coverage=0.8,
            edge_coverage=0.8,
            evidence_sufficiency=0.75,
            redundancy_count=0,
            redundancy_rate=0.0,
            wall_time_seconds=1.0,
            ground_truth_type="device_failure",
            ground_truth_location="device_a",
            predicted_type="device_failure",
            predicted_location="device_a",
            composite_episode_score=0.9,
        )
    )
    evaluator.add_episode_metrics(
        EpisodeMetrics(
            diagnosis_success=False,
            network_size=1,
            steps=2,
            normalized_steps=0.5,
            total_reward=-5.0,
            tool_cost=2.0,
            tool_cost_normalized=0.25,
            topology_coverage=0.3,
            node_coverage=0.3,
            edge_coverage=0.3,
            evidence_sufficiency=0.0,
            redundancy_count=1,
            redundancy_rate=0.5,
            wall_time_seconds=2.0,
            ground_truth_type="link_failure",
            ground_truth_location="device_a->device_b",
            predicted_type="device_failure",
            predicted_location="device_c",
            composite_episode_score=0.2,
        )
    )

    summary = evaluator.compute_summary()

    assert summary["episodes"] == 2
    assert 0.0 < summary["diagnosis_success_rate"] < 1.0
    assert "fault_type_macro_f1" in summary
    assert "confusion_matrix" in summary


class DummyNetwork:
    def get_all_devices(self):
        return ["device_a", "device_b"]

    def get_all_connections(self):
        return [("device_a", "device_b"), ("device_b", "device_a")]


class DummyDiscoveryMatrix:
    def __init__(self):
        self.nodes = ["device_a", "device_b"]
        self.adjacency = np.array([[0, 1], [1, 0]], dtype=np.int8)

    def get_discovered_devices(self):
        return list(self.nodes)


class DummyObservation:
    def __init__(self):
        self.discovery_matrix = DummyDiscoveryMatrix()


class DummyEnv(gym.Env):
    metadata = {}

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(1)
        self.network = DummyNetwork()
        self.observation = DummyObservation()
        self.max_episode_steps = 5
        self._step = 0

    def reset(self, seed=None, options=None):
        self._step = 0
        info = {
            "network_size": 2,
            "ground_truth_fault": {"type": "device_failure", "location": "device_a"},
        }
        return 0, info

    def step(self, action):
        self._step += 1
        terminated = self._step == 2
        info = {
            "action_spec": {
                "category": "diagnostic" if not terminated else "diagnosis",
                "action_type": "ping" if not terminated else "device_failure",
                "parameters": {"device": "device_a"} if not terminated else {"location": "device_a"},
            },
            "action_result": types.SimpleNamespace(result=None),
            "reward_breakdown": {"diagnosis_reward": 10.0 if terminated else 0.0},
            "ground_truth_fault": {"type": "device_failure", "location": "device_a"},
            "network_size": 2,
        }
        reward = 1.0 if terminated else -0.1
        return 0, reward, terminated, False, info


def test_metrics_wrapper_collects_episode():
    env = DummyEnv()
    wrapper = MetricsCollectorWrapper(env)

    obs, info = wrapper.reset()
    assert info["ground_truth_fault"]["type"] == "device_failure"

    wrapper.step(0)
    _, _, terminated, _, _ = wrapper.step(1)
    assert terminated is True

    metrics = wrapper.last_episode_metrics
    assert metrics is not None
    assert metrics.diagnosis_success is True
    assert wrapper.evaluator.compute_summary()["episodes"] == 1

