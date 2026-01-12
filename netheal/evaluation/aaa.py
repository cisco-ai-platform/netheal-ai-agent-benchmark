# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
AAA (Agentified Agent Assessment) export helpers for NetHeal metrics.

Provides JSON payload generation compatible with AgentBeats leaderboards.
Output format is designed for DuckDB queries as described in:
https://docs.agentbeats.dev/tutorial/#appendix-a-writing-leaderboard-queries
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import json

from .metrics import CompetitionEvaluator, EpisodeMetrics


def build_aaa_payload(
    evaluator: CompetitionEvaluator,
    purple_agent_id: str,
    green_agent_name: str = "netheal_green_v1",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a JSON-serializable payload for AgentBeats leaderboards.

    The output format matches AgentBeats expectations:
    - `participants`: dict mapping role to agent ID (for DuckDB joins)
    - `results`: array of per-episode results (for DuckDB UNNEST)
    - `summary`: aggregate metrics for quick leaderboard display

    Args:
        evaluator: CompetitionEvaluator with recorded episodes.
        purple_agent_id: AgentBeats UUID or identifier for the solver agent.
        green_agent_name: Evaluator name/version.
        metadata: Additional metadata to include.

    Returns:
        Dict ready for JSON serialization and DuckDB querying.
    """
    summary = evaluator.compute_summary()
    timestamp = datetime.now(timezone.utc).isoformat()

    # Build per-episode results array for DuckDB UNNEST queries
    results: List[Dict[str, Any]] = []
    for episode in evaluator.episodes:
        results.append(_episode_to_result(episode))

    # AgentBeats-compatible structure
    payload = {
        # Participants mapping for DuckDB joins
        "participants": {
            "solver": purple_agent_id,
            "evaluator": green_agent_name,
        },
        # Per-episode results for DuckDB UNNEST
        "results": results,
        # Summary metrics for quick leaderboard display
        "summary": {
            "pass_rate": summary.get("diagnosis_success_rate", 0.0),
            "time_used": summary.get("avg_wall_time_seconds", 0.0),
            "max_score": 100.0,
            "total_episodes": summary.get("episodes", 0),
            "composite_score": summary.get("composite_episode_score", 0.0),
            "diagnosis_success_rate": summary.get("diagnosis_success_rate", 0.0),
            "fault_type_macro_f1": summary.get("fault_type_macro_f1", 0.0),
            "normalized_steps": summary.get("normalized_steps", 0.0),
            "tool_cost_index": summary.get("tool_cost_index", 0.0),
            "topology_coverage": summary.get("topology_coverage", 0.0),
        },
        # Full metrics for detailed analysis
        "metrics": summary,
        # Metadata
        "generated_at": timestamp,
        "metadata": metadata or {},
    }

    return payload


def _episode_to_result(episode: EpisodeMetrics) -> Dict[str, Any]:
    """Convert EpisodeMetrics to AgentBeats result format."""
    return {
        "pass": episode.diagnosis_success,
        "pass_rate": 1.0 if episode.diagnosis_success else 0.0,
        "time_used": episode.wall_time_seconds,
        "steps": episode.steps,
        "max_score": 100.0,
        "score": episode.composite_episode_score * 100.0,
        "network_size": episode.network_size,
        "tool_cost": episode.tool_cost,
        "tool_cost_normalized": episode.tool_cost_normalized,
        "topology_coverage": episode.topology_coverage,
        "evidence_sufficiency": episode.evidence_sufficiency,
        "ground_truth_type": episode.ground_truth_type,
        "ground_truth_location": episode.ground_truth_location,
        "predicted_type": episode.predicted_type,
        "predicted_location": episode.predicted_location,
    }


def dump_aaa_payload(payload: Dict[str, Any], path: str) -> None:
    """Write AAA payload to disk."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


# Backwards compatibility aliases
build_agentbeats_payload = build_aaa_payload
dump_agentbeats_payload = dump_aaa_payload


__all__ = [
    "build_aaa_payload",
    "dump_aaa_payload",
    "build_agentbeats_payload",
    "dump_agentbeats_payload",
]
