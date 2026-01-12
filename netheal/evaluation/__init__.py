# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Evaluation utilities for NetHeal.

Provides episode trace collection and benchmark metrics computation
without modifying the base environment.
"""

from .wrapper import MetricsCollectorWrapper
from .metrics import (
    EpisodeTrace,
    ActionRecord,
    EpisodeMetrics,
    CompetitionEvaluator,
    compute_episode_metrics,
)
from .aaa import build_aaa_payload, dump_aaa_payload

# Backwards compatibility
from .aaa import build_agentbeats_payload, dump_agentbeats_payload

__all__ = [
    "MetricsCollectorWrapper",
    "EpisodeTrace",
    "ActionRecord",
    "EpisodeMetrics",
    "CompetitionEvaluator",
    "compute_episode_metrics",
    "build_aaa_payload",
    "dump_aaa_payload",
    # Backwards compatibility
    "build_agentbeats_payload",
    "dump_agentbeats_payload",
]
