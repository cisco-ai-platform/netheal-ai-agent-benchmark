# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Hint generation utilities for NetHeal.

Provides a configurable hint provider interface with:
- LLM-based natural language hints (Azure/OpenAI/Anthropic/Bedrock)
- Heuristic fallback hints for deterministic tests and offline use
"""

from .provider import (
    BaseHintProvider,
    LlmHintProvider,
    AzureGptHintProvider,
    HeuristicHintProvider,
    get_default_hint_provider,
)

__all__ = [
    "BaseHintProvider",
    "LlmHintProvider",
    "AzureGptHintProvider",
    "HeuristicHintProvider",
    "get_default_hint_provider",
]
