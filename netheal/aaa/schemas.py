# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Pydantic schemas for the NetHeal AAA protocol.

Defines data models for the A2A specification used across the FastAPI
server, MCP orchestration, and green/purple agent communication.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, AnyHttpUrl, NonNegativeInt, PositiveInt


class Participant(BaseModel):
    """Agent participant in an assessment."""

    role: str = Field(..., description="Role identifier (e.g., purple_agent).")
    endpoint: AnyHttpUrl = Field(..., description="Base URL for the participant agent.")
    card_url: Optional[AnyHttpUrl] = Field(
        default=None, description="Agent Card URL for capability discovery."
    )


class AssessmentConfig(BaseModel):
    """Configuration for a batch of NetHeal assessment episodes."""

    num_episodes: PositiveInt = Field(
        default=5, description="Number of scenarios to evaluate."
    )
    min_devices: PositiveInt = Field(
        default=3,
        ge=3,
        le=50,
        description="Minimum devices per topology.",
    )
    max_devices: PositiveInt = Field(
        default=15,
        ge=3,
        le=50,
        description="Maximum devices per topology.",
    )
    max_episode_steps: PositiveInt = Field(
        default=100, ge=5, le=200, description="Step budget per episode."
    )
    topology_types: List[str] = Field(
        default_factory=lambda: ["star", "mesh", "hierarchical"],
        description="Topology types to sample from.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Global seed for reproducibility.",
    )
    timeout_seconds: PositiveInt = Field(
        default=300,
        description="Max wall-clock time per episode.",
    )
    episode_concurrency: PositiveInt = Field(
        default=1,
        description="Number of episodes to run concurrently.",
    )
    episode_retry_limit: NonNegativeInt = Field(
        default=0,
        description="Retries per episode on timeout or error.",
    )
    fail_on_timeout: bool = Field(
        default=True,
        description="Fail assessment if any episode times out (unless max_timeouts is set).",
    )
    fail_on_error: bool = Field(
        default=True,
        description="Fail assessment if any episode errors (unless max_errors is set).",
    )
    max_timeouts: Optional[NonNegativeInt] = Field(
        default=None,
        description="Allow up to this many episode timeouts before failing.",
    )
    max_errors: Optional[NonNegativeInt] = Field(
        default=None,
        description="Allow up to this many episode errors before failing.",
    )
    enable_user_hints: bool = Field(
        default=True, description="Provide non-leaky hints to solver agents."
    )
    reward_scaling_factor: float = Field(
        default=10.0,
        ge=0.1,
        description="Scaling factor for sparse rewards.",
    )
    use_snapshots: bool = Field(
        default=False,
        description="Use pre-generated snapshots instead of random episodes.",
    )
    snapshot_path: Optional[str] = Field(
        default=None,
        description="Path to snapshot directory or JSONL file.",
    )
    snapshot_url: Optional[str] = Field(
        default=None,
        description="URL to snapshot archive (not yet supported).",
    )
    fault_sampling_strategy: str = Field(
        default="uniform",
        description="Fault sampling strategy (uniform, weighted, round_robin, stratified).",
    )
    fault_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Weights for fault types when using weighted sampling.",
    )
    latency_multiplier_range: Optional[List[float]] = Field(
        default=None,
        description="Latency multiplier range for performance degradation faults.",
    )
    extra_env_options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional environment kwargs.",
    )


class AssessmentRequest(BaseModel):
    """Request payload for the A2A /tasks endpoint."""

    task_id: Optional[str] = Field(
        default=None, description="Unique task identifier."
    )
    participants: Dict[str, Participant] = Field(
        default_factory=dict,
        description="Participant role to endpoint mapping.",
    )
    config: AssessmentConfig = Field(
        default_factory=AssessmentConfig, description="Assessment configuration."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Caller-provided metadata.",
    )


class EpisodeStart(BaseModel):
    """Message sent to purple agents announcing a new episode."""

    episode_index: NonNegativeInt = Field(
        ..., description="Zero-based episode index."
    )
    total_episodes: PositiveInt = Field(
        ..., description="Total episodes in the assessment."
    )
    mcp_server_url: AnyHttpUrl = Field(..., description="MCP tool server URL.")
    hint: Optional[str] = Field(
        default=None,
        description="Non-leaky hint describing network symptoms.",
    )
    network_size: Optional[int] = Field(
        default=None,
        description="Number of devices in this scenario.",
    )
    seed: Optional[int] = Field(
        default=None, description="Episode seed (if deterministic)."
    )
    max_steps: Optional[int] = Field(
        default=None,
        description="Step budget before episode ends.",
    )
    task_description: str = Field(
        default="Diagnose the network fault by exploring the topology, running diagnostics, and submitting your diagnosis using submit_diagnosis.",
        description="Task description for the solver.",
    )
    objective: str = Field(
        default="Identify the fault type and location, then call submit_diagnosis with your answer.",
        description="Goal the solver must achieve.",
    )
    fault_types: List[str] = Field(
        default_factory=lambda: ["device_failure", "link_failure", "misconfiguration", "performance_degradation"],
        description="Valid fault types for diagnosis.",
    )
    extra: Dict[str, Any] = Field(
        default_factory=dict, description="Additional scenario metadata."
    )


class DiagnosisSubmission(BaseModel):
    """Diagnosis submitted by a solver agent."""

    episode_index: NonNegativeInt = Field(
        ..., description="Episode this diagnosis applies to."
    )
    fault_type: str = Field(..., description="Predicted fault type.")
    location: str = Field(..., description="Predicted fault location.")
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1).",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Rationale or evidence summary.",
    )


class TaskStatus(str, Enum):
    """Task lifecycle states per A2A specification."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskUpdateLevel(str, Enum):
    """Severity level for task updates."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class TaskUpdate(BaseModel):
    """Streaming update emitted via SSE."""

    task_id: str = Field(..., description="Task identifier.")
    level: TaskUpdateLevel = Field(default=TaskUpdateLevel.INFO)
    message: str = Field(..., description="Status message.")
    payload: Dict[str, Any] = Field(
        default_factory=dict, description="Structured metadata."
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.utcnow(),
        description="UTC timestamp.",
    )


class Artifact(BaseModel):
    """Artifact in the assessment response."""

    label: str = Field(..., description="Artifact label.")
    mime_type: str = Field(
        default="application/json",
        description="MIME type.",
    )
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Artifact payload.",
    )


class AssessmentResult(BaseModel):
    """Final assessment response."""

    task_id: str = Field(..., description="Task identifier.")
    status: TaskStatus = Field(..., description="Terminal task status.")
    summary: Dict[str, Any] = Field(
        default_factory=dict, description="Aggregated metrics."
    )
    artifacts: List[Artifact] = Field(
        default_factory=list,
        description="Assessment artifacts.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata.",
    )


__all__ = [
    "AssessmentConfig",
    "AssessmentRequest",
    "EpisodeStart",
    "DiagnosisSubmission",
    "TaskStatus",
    "TaskUpdate",
    "TaskUpdateLevel",
    "Participant",
    "Artifact",
    "AssessmentResult",
]
