"""
NetHeal AAA (Agentified Agent Assessment) Protocol Implementation.

Provides A2A-compatible agents following the AAA format for standardized
agent-to-agent evaluation in network troubleshooting scenarios.

Components:
    - Green Agent: Assessment orchestrator that manages episodes and scoring
    - Purple Agent: Solver agent baseline implementation
    - MCP Server: Tool interface for diagnostic operations
    - Schemas: Pydantic models for the A2A protocol
"""

from .green_agent import NetHealGreenAgent
from .mcp_server import EpisodeRuntime, NetHealMCPServer
from .schemas import (
    AssessmentConfig,
    AssessmentRequest,
    AssessmentResult,
    Artifact,
    DiagnosisSubmission,
    EpisodeStart,
    Participant,
    TaskStatus,
    TaskUpdate,
    TaskUpdateLevel,
)
from .server import app as green_app
from .server import set_card_url
from .purple_server import app as purple_app
from .purple_server import set_config as set_purple_config

__all__ = [
    # Green agent
    "NetHealGreenAgent",
    "green_app",
    "set_card_url",
    # Purple agent
    "purple_app",
    "set_purple_config",
    # MCP server
    "NetHealMCPServer",
    "EpisodeRuntime",
    # Schemas
    "AssessmentConfig",
    "AssessmentRequest",
    "AssessmentResult",
    "Artifact",
    "DiagnosisSubmission",
    "EpisodeStart",
    "Participant",
    "TaskStatus",
    "TaskUpdate",
    "TaskUpdateLevel",
]
