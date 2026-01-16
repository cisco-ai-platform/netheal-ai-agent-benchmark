# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Pydantic request schemas for the NetHeal web API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ResetRequest(BaseModel):
    seed: Optional[int] = Field(default=None, description="Random seed")
    max_devices: int = Field(default=15, ge=3, le=50)
    max_episode_steps: int = Field(default=100, ge=1, le=200)
    topology_types: Optional[List[str]] = None
    enable_user_hints: bool = True
    hint_provider_mode: str = Field(default="auto")
    user_context: Optional[Dict[str, Any]] = None


class StepRequest(BaseModel):
    action_id: int


class ImportScenarioRequest(BaseModel):
    scenario_data: Dict[str, Any] = Field(..., description="Complete scenario state to import")
