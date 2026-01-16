# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Metric data structures and computations for the NetHeal environment.

This module implements the Stage 1 (wrapper-based) metrics described in the plan:
- Episode-level statistics (DSR, F1, TTD, tool costs, coverage, etc.)
- Aggregation utilities for competition-ready reporting
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

FAULT_TYPE_ORDER = [
    "device_failure",
    "link_failure",
    "misconfiguration",
    "performance_degradation",
]

MAX_TOOL_COST_PER_STEP = 5.0
DEFAULT_ACTION_COST = 1.0  # Fallback when ToolResult.cost is unavailable

# TODO: Validate that tool costs (defined in ToolSimulator) reflect real-world
# diagnostic complexity. Current values (ping=1.0, traceroute=2.0, check_status=0.5,
# check_interfaces=1.5) are placeholders and may need empirical validation.


@dataclass
class ActionRecord:
    """Single action execution captured by the metrics wrapper."""

    step: int
    action_id: int
    category: Optional[str]
    action_type: Optional[str]
    parameters: Dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]
    timestamp: float


@dataclass
class EpisodeTrace:
    """Raw episode trace including actions and environment metadata."""

    ground_truth: Optional[Dict[str, Any]]
    network_size: int
    max_episode_steps: int
    network_devices: List[str]
    network_edges: List[Tuple[str, str]]
    start_time: float
    actions: List[ActionRecord] = field(default_factory=list)
    total_reward: float = 0.0
    final_observation: Optional[Dict[str, Any]] = None
    final_info: Optional[Dict[str, Any]] = None
    end_time: Optional[float] = None
    termination_reason: Optional[str] = None
    discovered_nodes: int = 0
    discovered_edges: int = 0


@dataclass
class EpisodeMetrics:
    """Computed metrics for a single episode."""

    # Primary outcome
    diagnosis_success: bool
    
    # Episode context
    network_size: int
    steps: int
    normalized_steps: float  # steps / max_episode_steps
    
    # Reward (useful for RL training)
    total_reward: float
    
    # Tool usage
    tool_cost: float
    tool_cost_normalized: float
    
    # Exploration (for analysis)
    topology_coverage: float
    node_coverage: float
    edge_coverage: float
    
    # Investigation quality (for analysis)
    evidence_sufficiency: float
    
    # Redundancy (for debugging)
    redundancy_count: int
    redundancy_rate: float
    
    # Wall clock time (for reference)
    wall_time_seconds: float
    
    # Ground truth and prediction (for confusion matrix)
    ground_truth_type: Optional[str]
    ground_truth_location: Optional[str]
    predicted_type: Optional[str]
    predicted_location: Optional[str]
    
    # Final composite score
    composite_episode_score: float


class CompetitionEvaluator:
    """Aggregates episode metrics across runs for competition reporting."""

    def __init__(self) -> None:
        self.episodes: List[EpisodeMetrics] = []

    def add_episode_metrics(self, metrics: EpisodeMetrics) -> None:
        """Track metrics for an episode."""
        self.episodes.append(metrics)

    def clear(self) -> None:
        """Reset stored metrics."""
        self.episodes.clear()

    def compute_summary(self) -> Dict[str, Any]:
        """Compute aggregate statistics across all recorded episodes."""
        if not self.episodes:
            return {}

        weights = [max(1, m.network_size) for m in self.episodes]
        total_weight = sum(weights)

        def _weighted_avg(values: List[float]) -> float:
            return sum(v * w for v, w in zip(values, weights)) / total_weight

        diagnosis_success_rate = _weighted_avg(
            [1.0 if m.diagnosis_success else 0.0 for m in self.episodes]
        )
        avg_normalized_steps = _weighted_avg([m.normalized_steps for m in self.episodes])
        avg_steps = _weighted_avg([float(m.steps) for m in self.episodes])
        avg_total_reward = _weighted_avg([m.total_reward for m in self.episodes])
        avg_tool_cost = _weighted_avg([m.tool_cost_normalized for m in self.episodes])
        avg_topology_coverage = _weighted_avg([m.topology_coverage for m in self.episodes])
        avg_evidence = _weighted_avg([m.evidence_sufficiency for m in self.episodes])
        avg_composite = _weighted_avg([m.composite_episode_score for m in self.episodes])
        avg_wall_time = sum(m.wall_time_seconds for m in self.episodes) / len(self.episodes)
        avg_redundancy = _weighted_avg([m.redundancy_rate for m in self.episodes])

        confusion = _build_confusion_matrix(self.episodes)
        macro_f1 = _macro_f1(confusion)

        summary = {
            "episodes": len(self.episodes),
            "diagnosis_success_rate": diagnosis_success_rate,
            "fault_type_macro_f1": macro_f1,
            "avg_steps": avg_steps,
            "avg_total_reward": avg_total_reward,
            "normalized_steps": avg_normalized_steps,
            "composite_episode_score": avg_composite,
            "tool_cost_index": avg_tool_cost,
            "topology_coverage": avg_topology_coverage,
            "evidence_sufficiency": avg_evidence,
            "redundancy_rate": avg_redundancy,
            "avg_wall_time_seconds": avg_wall_time,
            "confusion_matrix": confusion,
        }

        return summary


def compute_episode_metrics(trace: EpisodeTrace) -> EpisodeMetrics:
    """Compute metrics for a single episode trace."""
    if trace.end_time is None:
        raise ValueError("EpisodeTrace.end_time must be set before computing metrics.")

    steps = len(trace.actions)
    normalized_steps = _safe_divide(steps, trace.max_episode_steps)
    total_reward = trace.total_reward
    tool_cost = sum(_estimate_action_cost(action) for action in trace.actions)
    tool_cost_normalized = min(
        1.0, _safe_divide(tool_cost, max(1, steps) * MAX_TOOL_COST_PER_STEP)
    )

    node_coverage = _safe_divide(trace.discovered_nodes, max(1, trace.network_size))
    edge_coverage = _safe_divide(trace.discovered_edges, max(1, len(trace.network_edges)))
    topology_coverage = 0.5 * (node_coverage + edge_coverage)

    diagnostic_actions = [a for a in trace.actions if a.category == "diagnostic"]

    evidence_sufficiency = _safe_divide(
        sum(_is_relevant_diagnostic(a, trace.ground_truth) for a in diagnostic_actions),
        len(diagnostic_actions),
    )

    redundancy_count, redundancy_rate = _redundancy_stats(diagnostic_actions)

    wall_time = trace.end_time - trace.start_time
    ground_truth_type = (trace.ground_truth or {}).get("type")
    ground_truth_location = (trace.ground_truth or {}).get("location")
    predicted_type, predicted_location = _extract_prediction(trace.actions)
    diagnosis_success = _diagnosis_success(
        predicted_type, predicted_location, trace.ground_truth, trace.final_info
    )

    # Composite score favors the environment reward; step penalties are already included.
    composite_score = total_reward

    return EpisodeMetrics(
        diagnosis_success=diagnosis_success,
        network_size=trace.network_size,
        steps=steps,
        normalized_steps=normalized_steps,
        total_reward=total_reward,
        tool_cost=tool_cost,
        tool_cost_normalized=tool_cost_normalized,
        topology_coverage=topology_coverage,
        node_coverage=node_coverage,
        edge_coverage=edge_coverage,
        evidence_sufficiency=evidence_sufficiency,
        redundancy_count=redundancy_count,
        redundancy_rate=redundancy_rate,
        wall_time_seconds=wall_time,
        ground_truth_type=ground_truth_type,
        ground_truth_location=ground_truth_location,
        predicted_type=predicted_type,
        predicted_location=predicted_location,
        composite_episode_score=composite_score,
    )


def _diagnosis_success(
    predicted_type: Optional[str],
    predicted_location: Optional[str],
    ground_truth: Optional[Dict[str, Any]],
    final_info: Optional[Dict[str, Any]],
) -> bool:
    """Determine if the diagnosis was fully successful (exact type + location match)."""
    if not ground_truth:
        return False

    # Check if environment already computed success via positive diagnosis_reward
    if final_info:
        reward_breakdown = final_info.get("reward_breakdown", {}) or {}
        if reward_breakdown.get("diagnosis_reward", 0.0) > 0:
            return True

    # Require exact match of type and location (order-insensitive for link faults)
    gt_type = ground_truth.get("type")
    gt_location = ground_truth.get("location")
    if predicted_type != gt_type:
        return False

    if gt_type in {"link_failure", "performance_degradation"}:
        return _link_locations_match(predicted_location, gt_location)

    return predicted_location == gt_location


def _link_locations_match(
    predicted_location: Optional[str], ground_truth_location: Optional[str]
) -> bool:
    if not predicted_location or not ground_truth_location:
        return False
    predicted = _normalize_link_location(predicted_location)
    ground_truth = _normalize_link_location(ground_truth_location)
    if predicted and ground_truth:
        return predicted == ground_truth
    return predicted_location == ground_truth_location


def _normalize_link_location(location: str) -> Optional[Tuple[str, str]]:
    if "->" not in location:
        return None
    parts = [part.strip() for part in location.split("->") if part.strip()]
    if len(parts) != 2:
        return None
    return tuple(sorted(parts))


def _estimate_action_cost(action: ActionRecord) -> float:
    """Estimate action cost from ToolResult.cost, falling back to default."""
    action_result = action.info.get("action_result")
    if action_result is not None:
        tool_result = getattr(action_result, "result", None)
        if tool_result is not None:
            cost = getattr(tool_result, "cost", None)
            if cost is not None:
                return float(cost)
    return DEFAULT_ACTION_COST


def _is_relevant_diagnostic(action: ActionRecord, ground_truth: Optional[Dict[str, Any]]) -> bool:
    if not ground_truth:
        return False
    if action.category != "diagnostic":
        return False

    fault_type = ground_truth.get("type")
    location = ground_truth.get("location", "")
    parameters = action.parameters or {}
    source = parameters.get("source")
    destination = parameters.get("destination")
    device = parameters.get("device")

    if not location:
        return False

    if fault_type in {"device_failure", "misconfiguration"}:
        if device and device == location:
            return True
        if source == location or destination == location:
            return True
        return False

    if "->" in location:
        endpoints = location.split("->")
    else:
        endpoints = [location]

    if source and destination and len(endpoints) == 2:
        return {source, destination} == set(endpoints)

    if device and device in endpoints:
        return True

    return False


def _redundancy_stats(actions: List[ActionRecord]) -> Tuple[int, float]:
    seen = set()
    redundant = 0
    for action in actions:
        params = tuple(sorted((action.parameters or {}).items()))
        signature = (action.action_type, params)
        if signature in seen:
            redundant += 1
        else:
            seen.add(signature)

    rate = _safe_divide(redundant, len(actions))
    return redundant, rate


def _extract_prediction(actions: List[ActionRecord]) -> Tuple[Optional[str], Optional[str]]:
    if not actions:
        return (None, None)
    final_action = actions[-1]
    if final_action.category != "diagnosis":
        return (None, None)

    action_type = final_action.action_type
    location = (final_action.parameters or {}).get("location")
    return (action_type, location)


def _build_confusion_matrix(episodes: List[EpisodeMetrics]) -> Dict[str, Dict[str, int]]:
    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for metrics in episodes:
        gt = metrics.ground_truth_type or "unknown"
        pred = metrics.predicted_type or "unknown"
        confusion[gt][pred] += 1
    return {gt: dict(preds) for gt, preds in confusion.items()}


def _macro_f1(confusion: Dict[str, Dict[str, int]]) -> float:
    scores = []
    labels = FAULT_TYPE_ORDER
    for label in labels:
        tp = confusion.get(label, {}).get(label, 0)
        fp = sum(confusion.get(other, {}).get(label, 0) for other in labels if other != label)
        fn = sum(confusion.get(label, {}).get(other, 0) for other in labels if other != label)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return sum(scores) / len(labels)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


__all__ = [
    "ActionRecord",
    "EpisodeTrace",
    "EpisodeMetrics",
    "CompetitionEvaluator",
    "compute_episode_metrics",
]

