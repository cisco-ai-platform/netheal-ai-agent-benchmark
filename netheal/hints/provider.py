# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Hint provider implementations for NetHeal.

Providers:
- LlmHintProvider: uses a configured LLM backend (azure | openai | anthropic | bedrock)
- AzureGptHintProvider: Azure-specific convenience wrapper (backward-compatible)
- HeuristicHintProvider: offline/deterministic fallback based on ground truth fault

Selection helper:
- get_default_hint_provider(mode): 'auto' | 'heuristic' | 'azure' | 'openai' | 'anthropic' | 'bedrock'
  In 'auto' mode, the provider follows LLM_PROVIDER when set, otherwise selects
  the single configured backend; if ambiguous, it falls back to heuristic.

Environment variables (shared with solver LLM configuration):
- LLM_PROVIDER: azure | openai | anthropic | bedrock
- LLM_MODEL: model/deployment name (fallback for provider-specific model vars)

Provider-specific configuration:
- Azure: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT
- OpenAI: OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL, OPENAI_ORG_ID
- Anthropic: ANTHROPIC_API_KEY, ANTHROPIC_MODEL, ANTHROPIC_API_URL
- Bedrock: AWS_REGION/AWS_DEFAULT_REGION, BEDROCK_MODEL_ID, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List
import os
import re
import time

# OpenAI Python SDK (supports Azure via AzureOpenAI class)
try:
    from openai import AzureOpenAI, OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    AzureOpenAI = None  # type: ignore
    OpenAI = None  # type: ignore

try:
    from anthropic import Anthropic  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Anthropic = None  # type: ignore

try:
    import boto3  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    boto3 = None  # type: ignore

try:
    import weave  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    weave = None  # type: ignore


if weave is not None:  # pragma: no cover - weave tracing is optional

    @weave.op()
    def _record_hint_generation(**payload: Any) -> Dict[str, Any]:
        """Trace payload for Weave dashboards."""
        return payload

else:  # pragma: no cover - executed when weave is unavailable

    def _record_hint_generation(**payload: Any) -> Dict[str, Any]:
        return payload


# --------------------------- Base Interface ---------------------------

class BaseHintProvider:
    """Abstract interface for generating non-leaky user hints."""

    def generate_hint(self, context: Dict[str, Any]) -> str:
        raise NotImplementedError


# --------------------------- Heuristic Provider ---------------------------

@dataclass
class HeuristicHintProvider(BaseHintProvider):
    """Deterministic, offline hint generator.

    It uses the ground truth fault to produce a soft, directional hint without
    revealing fault type or exact device/link identifiers.
    """

    def generate_hint(self, context: Dict[str, Any]) -> str:
        fault_type = (context.get("ground_truth", {}) or {}).get("type", "")
        access_point = context.get("user_context", {}).get("access_point", "the office Wi‑Fi")
        # Generic client symptom to satisfy the spec request
        lead_in = f"I'm noticing network issues while connected to {access_point}. "

        # Compose per-fault but non-leaky language
        if fault_type == "device_failure":
            body = (
                "Some services suddenly became unreachable for multiple colleagues. "
                "Local resources still work, but anything past a common aggregation point times out."
            )
        elif fault_type == "link_failure":
            body = (
                "I can reach nearby devices, but anything in another part of the network seems isolated. "
                "Pings to resources across the building never return."
            )
        elif fault_type == "misconfiguration":
            body = (
                "Accessing one specific internal resource fails from my segment, while others are fine. "
                "Colleagues on a different segment can reach it without problems."
            )
        elif fault_type == "performance_degradation":
            body = (
                "Connectivity works, but everything is unusually slow—streaming stutters and file transfers crawl. "
                "It started recently without any local changes on my laptop."
            )
        else:
            body = (
                "Some destinations are intermittently unreachable while others are fine. "
                "It feels like the issue depends on where the traffic is going."
            )

        hint = (lead_in + body).strip()
        hint = _sanitize_hint(hint, context)
        return hint


# --------------------------- LLM Provider ---------------------------

_SUPPORTED_LLM_PROVIDERS = ("azure", "openai", "anthropic", "bedrock")


def _build_hint_prompts(context: Dict[str, Any]) -> Tuple[str, str]:
    system = (
        "You are an end user reporting network connectivity issues to IT support. "
        "Write a natural, user-centric problem description focusing ONLY on symptoms you observe. "
        "CRITICAL CONSTRAINTS:\n"
        "- Write from USER perspective (not technical/diagnostic perspective)\n"
        "- Describe SYMPTOMS only (what's broken/slow), NOT diagnostic steps\n"
        "- NO technical topology terms (no 'star', 'mesh', 'hierarchical', 'linear', etc.)\n"
        "- NO diagnostic instructions (no 'probe', 'verify', 'check', 'test', etc.)\n"
        "- NO device IDs, link names, or infrastructure details\n"
        "- NO fault type names ('device failure', 'link failure', 'misconfiguration', 'performance degradation')\n"
        "- Sound like a frustrated office worker, not a network engineer\n"
        "- VARY your phrasing significantly - use different sentence structures, different ways to describe location/scope\n"
        "- Keep under 40 words, 1-2 sentences\n"
    )
    fault_type = context.get("ground_truth", {}).get("type", "")
    symptom_guidance = ""
    if fault_type == "performance_degradation":
        symptom_guidance = (
            "CRITICAL: This is PERFORMANCE issue - use ONLY slowness terms: "
            "'slow', 'crawl', 'lag', 'stutters', 'barely responds', 'takes forever'. "
            "DO NOT use failure terms like 'timeout', 'won't load', 'unreachable', 'dead'."
        )
    elif fault_type == "device_failure":
        symptom_guidance = (
            "CRITICAL: This is DEVICE FAILURE - use complete failure terms: "
            "'unreachable', 'won't load', 'times out', 'dead', 'can't access'. "
            "DO NOT use slowness terms like 'slow', 'crawl', 'lag'."
        )
    elif fault_type == "link_failure":
        symptom_guidance = (
            "CRITICAL: This is LINK FAILURE - emphasize location/scope: "
            "'people on my floor', 'my side of office', 'our wing', 'folks near me'. "
            "Use failure terms: 'unreachable', 'times out', 'won't load'."
        )
    elif fault_type == "misconfiguration":
        symptom_guidance = (
            "CRITICAL: This is MISCONFIGURATION - show CLEAR asymmetry: "
            "'X works but Y doesn't', 'can access A but not B'. "
            "Use failure terms for affected services, but be selective (not all services fail)."
        )

    # Extract individual fields from context for cleaner prompt construction
    ground_truth = context.get("ground_truth", {}) or {}
    fault_location = ground_truth.get("location", "unknown")
    network_size = context.get("network_size", "unknown")
    topology_types = context.get("topology_types", [])
    user_context = context.get("user_context", {}) or {}
    access_point = user_context.get("access_point", "office Wi-Fi")
    department = user_context.get("department", "general office")

    # Format topology types for display
    topology_str = ", ".join(topology_types) if topology_types else "mixed"

    user = (
        "Ground truth context (for generating realistic symptoms, do NOT leak this info):\n"
        f"- Fault type: {fault_type}\n"
        f"- Fault location: {fault_location}\n"
        f"- Network size: {network_size} devices\n"
        f"- Topology: {topology_str}\n"
        f"- User access point: {access_point}\n"
        f"- User department: {department}\n\n"
        f"{symptom_guidance}\n\n"
        "Generate a diverse user complaint describing OBSERVABLE SYMPTOMS without revealing the technical cause or topology.\n\n"
        "VARY YOUR STYLE - use different structures like:\n"
        "- 'The file server times out, but email works. My whole team sees this.'\n"
        "- 'Can't load the intranet—just spins forever. Web browsing is fine though.'\n"
        "- 'Everything's super slow since this morning. Shared drives barely respond but Slack is normal.'\n"
        "- 'Getting timeouts on internal tools. External sites and email are fine. Others near me have it too.'\n"
        "- 'Half our apps won't connect—just hang—while the other half work fine.'\n\n"
        "Vary: sentence structure, how you mention location (or don't), which apps you name, how you describe the problem."
    )
    return system, user


def _maybe_log(level: str, message: str) -> None:
    import sys
    if hasattr(sys.modules.get("__main__"), "__file__"):
        try:
            from loguru import logger
            log_fn = getattr(logger, level, None)
            if callable(log_fn):
                log_fn(message)
        except Exception:
            pass


@dataclass
class LlmHintProvider(BaseHintProvider):
    """LLM-based hint generator using Azure, OpenAI, Anthropic, or Bedrock."""

    provider: Optional[str] = None
    model: Optional[str] = None
    # Azure overrides (optional)
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    endpoint: Optional[str] = None
    deployment: Optional[str] = None
    weave_project: Optional[str] = None
    _weave_ready: bool = field(default=False, init=False, repr=False)

    def _init_weave(self) -> bool:
        """Initialize W&B Weave when tracing is enabled."""
        if weave is None or self._weave_ready:
            return self._weave_ready
        project = self.weave_project or os.getenv("WEAVE_PROJECT")
        if not project:
            return False
        try:
            weave.init(project)
            self._weave_ready = True
        except Exception:  # pragma: no cover - tracing is best effort
            self._weave_ready = False
        return self._weave_ready

    def _resolve_provider(self) -> Optional[str]:
        return _resolve_llm_provider(self.provider)

    def _init_client(self, provider: str) -> Tuple[Any, str]:
        if provider == "azure":
            if AzureOpenAI is None:
                raise RuntimeError("AzureOpenAI SDK not available. Install 'openai>=1.0'.")
            api_key = self.api_key or os.getenv("AZURE_OPENAI_API_KEY")
            api_version = self.api_version or os.getenv("AZURE_OPENAI_API_VERSION")
            endpoint = self.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
            deployment = (
                self.deployment
                or self.model
                or os.getenv("AZURE_OPENAI_DEPLOYMENT")
                or os.getenv("LLM_MODEL")
                or "gpt-4o"
            )
            if not api_key or not api_version or not endpoint:
                raise RuntimeError("Azure OpenAI environment not configured.")
            return AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint), deployment

        if provider == "openai":
            if OpenAI is None:
                raise RuntimeError("OpenAI SDK not available. Install 'openai>=1.0'.")
            api_key = os.getenv("OPENAI_API_KEY")
            model = self.model or os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL")
            base_url = os.getenv("OPENAI_BASE_URL")
            org_id = os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION")
            if not api_key or not model:
                raise RuntimeError("OpenAI environment not configured.")
            return OpenAI(api_key=api_key, base_url=base_url, organization=org_id), model

        if provider == "anthropic":
            if Anthropic is None:
                raise RuntimeError("Anthropic SDK not available. Install 'anthropic>=0.37.0'.")
            api_key = os.getenv("ANTHROPIC_API_KEY")
            model = self.model or os.getenv("ANTHROPIC_MODEL") or os.getenv("LLM_MODEL")
            base_url = os.getenv("ANTHROPIC_API_URL") or os.getenv("ANTHROPIC_BASE_URL")
            if not api_key or not model:
                raise RuntimeError("Anthropic environment not configured.")
            return Anthropic(api_key=api_key, base_url=base_url), model

        if provider == "bedrock":
            if boto3 is None:
                raise RuntimeError("boto3 not available. Install 'boto3'.")
            region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
            model = self.model or os.getenv("BEDROCK_MODEL_ID") or os.getenv("LLM_MODEL")
            if not region or not model:
                raise RuntimeError("Bedrock environment not configured.")
            session_token = os.getenv("AWS_SESSION_TOKEN") or os.getenv("AWS_SESSION_ID")
            client_kwargs = {"service_name": "bedrock-runtime", "region_name": region}
            access_key = os.getenv("AWS_ACCESS_KEY_ID")
            secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            if access_key and secret_key:
                client_kwargs["aws_access_key_id"] = access_key
                client_kwargs["aws_secret_access_key"] = secret_key
                if session_token:
                    client_kwargs["aws_session_token"] = session_token
            return boto3.client(**client_kwargs), model

        raise RuntimeError(f"Unsupported LLM provider '{provider}'.")

    def _call_openai_chat(self, client: Any, model: str, system: str, user: str) -> Tuple[str, Optional[float]]:
        # Note: GPT-5 only supports temperature=1 (default)
        create_params: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "n": 1,
        }
        if "gpt-5" in model.lower():
            create_params["max_completion_tokens"] = 4096
        else:
            create_params["max_tokens"] = 256
            create_params["temperature"] = 0.7

        start = time.perf_counter()
        resp = client.chat.completions.create(**create_params)
        latency_ms = (time.perf_counter() - start) * 1000.0
        text = (resp.choices[0].message.content or "").strip()
        return text, latency_ms

    def _call_anthropic(self, client: Any, model: str, system: str, user: str) -> Tuple[str, Optional[float]]:
        create_params: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": 256,
            "temperature": 0.7,
        }
        if system:
            create_params["system"] = system
        start = time.perf_counter()
        resp = client.messages.create(**create_params)
        latency_ms = (time.perf_counter() - start) * 1000.0
        content_blocks = resp.content or []
        text_parts: List[str] = []
        for block in content_blocks:
            block_type = getattr(block, "type", None)
            if block_type is None and isinstance(block, dict):
                block_type = block.get("type")
            if block_type == "text":
                text_parts.append(getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else ""))
        return "".join(text_parts).strip(), latency_ms

    def _call_bedrock(self, client: Any, model: str, system: str, user: str) -> Tuple[str, Optional[float]]:
        request: Dict[str, Any] = {
            "modelId": model,
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": str(user)}],
                }
            ],
            "inferenceConfig": {
                "maxTokens": 256,
                "temperature": 0.7,
            },
        }
        if system:
            request["system"] = [{"text": system}]
        start = time.perf_counter()
        response = client.converse(**request)
        latency_ms = (time.perf_counter() - start) * 1000.0
        output_message = (response.get("output", {}) or {}).get("message", {}) or {}
        content_blocks = output_message.get("content", []) or []
        text_parts: List[str] = []
        for block in content_blocks:
            if "text" in block:
                text_parts.append(block.get("text", ""))
        return "".join(text_parts).strip(), latency_ms

    def generate_hint(self, context: Dict[str, Any]) -> str:
        system, user = _build_hint_prompts(context)
        provider = self._resolve_provider()
        weave_enabled = self._init_weave()
        latency_ms = None
        text = ""
        fallback_used = False
        model_name = None

        try:  # pragma: no cover - avoid network in unit tests
            if not provider:
                raise RuntimeError("No LLM provider configured for hints.")
            client, model_name = self._init_client(provider)
            if provider in ("azure", "openai"):
                text, latency_ms = self._call_openai_chat(client, model_name, system, user)
            elif provider == "anthropic":
                text, latency_ms = self._call_anthropic(client, model_name, system, user)
            elif provider == "bedrock":
                text, latency_ms = self._call_bedrock(client, model_name, system, user)
            else:
                raise RuntimeError(f"Unsupported provider '{provider}'.")
            _maybe_log("debug", f"Raw {provider} hint before sanitization: '{text}'")
        except Exception as e:
            fallback_used = True
            _maybe_log("warning", f"LLM hint generation failed ({provider or 'auto'}), using heuristic fallback: {e}")
            text = HeuristicHintProvider().generate_hint(context)

        sanitized = _sanitize_hint(text, context)

        if weave_enabled:
            _record_hint_generation(
                provider=provider or "heuristic",
                model=model_name,
                context=context,
                system_prompt=system,
                user_prompt=user,
                raw_response=text,
                sanitized_hint=sanitized,
                latency_ms=latency_ms,
                fallback_used=fallback_used,
            )

        return sanitized


class AzureGptHintProvider(LlmHintProvider):
    """Azure OpenAI GPT-based hint generator (backward-compatible wrapper)."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("provider", None)
        super().__init__(provider="azure", **kwargs)


# --------------------------- Helper selection ---------------------------

def get_default_hint_provider(mode: str = "auto") -> BaseHintProvider:
    mode = (mode or "auto").lower()
    if mode == "heuristic":
        return HeuristicHintProvider()
    if mode == "azure":
        return AzureGptHintProvider()
    if mode in ("openai", "anthropic", "bedrock"):
        return LlmHintProvider(provider=mode)
    # auto
    provider = _resolve_llm_provider("auto")
    if provider == "azure":
        return AzureGptHintProvider()
    if provider in ("openai", "anthropic", "bedrock"):
        return LlmHintProvider(provider=provider)
    return HeuristicHintProvider()


def _azure_env_present() -> bool:
    return all(
        os.getenv(k)
        for k in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_ENDPOINT")
    )


def _openai_env_present() -> bool:
    return bool(
        os.getenv("OPENAI_API_KEY")
        and (os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL"))
    )


def _anthropic_env_present() -> bool:
    return bool(
        os.getenv("ANTHROPIC_API_KEY")
        and (os.getenv("ANTHROPIC_MODEL") or os.getenv("LLM_MODEL"))
    )


def _bedrock_env_present() -> bool:
    return bool(
        (os.getenv("BEDROCK_MODEL_ID") or os.getenv("LLM_MODEL"))
        and (os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"))
    )


def _infer_provider_from_model(available: List[str]) -> Optional[str]:
    model = os.getenv("LLM_MODEL")
    if not model:
        return None
    lowered = model.lower()
    if "bedrock" in available:
        if lowered.startswith(("anthropic.", "amazon.", "meta.")):
            return "bedrock"
    if "anthropic" in available and "claude" in lowered:
        return "anthropic"
    if "openai" in available and "azure" not in available:
        return "openai"
    if "azure" in available and "openai" not in available:
        return "azure"
    return None


def _resolve_llm_provider(mode: Optional[str] = None) -> Optional[str]:
    if mode:
        normalized = mode.strip().lower()
        if normalized in _SUPPORTED_LLM_PROVIDERS:
            return normalized
        if normalized not in ("auto", ""):
            return None

    env_provider = os.getenv("LLM_PROVIDER")
    if env_provider:
        env_provider = env_provider.strip().lower()
        if env_provider in _SUPPORTED_LLM_PROVIDERS:
            return env_provider

    available: List[str] = []
    if _azure_env_present():
        available.append("azure")
    if _openai_env_present():
        available.append("openai")
    if _anthropic_env_present():
        available.append("anthropic")
    if _bedrock_env_present():
        available.append("bedrock")

    if not available:
        return None
    if len(available) == 1:
        return available[0]

    inferred = _infer_provider_from_model(available)
    return inferred


# --------------------------- Sanitization ---------------------------

_FORBIDDEN_TERMS = (
    "device_failure",
    "link_failure",
    "misconfiguration",
    "performance_degradation",
)


def _sanitize_hint(text: str, context: Dict[str, Any]) -> str:
    """Ensure the hint does not leak ground truth terms or internal IDs.

    - Removes specific device IDs like 'device_3' or link literals like 'device_1->device_2'
    - Removes explicit fault type terms
    """
    original_text = text
    
    if not text:
        return "I'm experiencing connectivity issues on the office Wi‑Fi; some internal resources are unreachable while others work."

    # Strip explicit fault terms (case-insensitive)
    for term in _FORBIDDEN_TERMS:
        text = re.sub(re.escape(term), "issue", text, flags=re.IGNORECASE)

    # Strip device IDs and connection arrows
    text = re.sub(r"device_\d+", "a device", text, flags=re.IGNORECASE)
    text = re.sub(r"\b([a-zA-Z0-9_]+)->([a-zA-Z0-9_]+)\b", "a connection", text)

    # Optionally strip the exact ground-truth location if present
    gt = (context.get("ground_truth", {}) or {}).get("location", "")
    if isinstance(gt, str) and gt:
        pattern = re.escape(gt)
        text = re.sub(pattern, "a location", text)

    # Normalize whitespace
    sanitized = re.sub(r"\s+", " ", text).strip()
    
    # Debug logging
    import sys
    if hasattr(sys.modules.get('__main__'), '__file__'):
        try:
            from loguru import logger
            if original_text != sanitized:
                logger.debug(f"Sanitization changed hint from: '{original_text}' to: '{sanitized}'")
        except:
            pass
    
    return sanitized
