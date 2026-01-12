"""Hint provider implementations for NetHeal.

Two providers are exposed:
- AzureGptHintProvider: uses Azure OpenAI (GPT-4.1) to generate a non-leaky hint
- HeuristicHintProvider: offline/deterministic fallback based on ground truth fault

Selection helper:
- get_default_hint_provider(mode): 'azure' | 'heuristic' | 'auto'
  In 'auto' mode, Azure provider is chosen when required environment variables
  are present; otherwise the heuristic provider is used.

Environment variables for Azure provider:
- AZURE_OPENAI_ENDPOINT
- AZURE_OPENAI_API_KEY
- AZURE_OPENAI_API_VERSION (e.g., '2024-02-15-preview')
- AZURE_OPENAI_DEPLOYMENT (deployment name for GPT-4.1 or compatible)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import os
import re
import time

# Azure OpenAI via OpenAI Python SDK (supports Azure via AzureOpenAI class)
try:
    from openai import AzureOpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    AzureOpenAI = None  # type: ignore

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


# --------------------------- Azure GPT Provider ---------------------------

@dataclass
class AzureGptHintProvider(BaseHintProvider):
    """Azure OpenAI GPT-based hint generator.

    Requires environment variables described above. If anything is missing,
    this provider should not be used (callers can fall back to Heuristic)."""

    api_key: Optional[str] = None
    api_version: Optional[str] = None
    endpoint: Optional[str] = None
    deployment: Optional[str] = None
    weave_project: Optional[str] = None
    _weave_ready: bool = field(default=False, init=False, repr=False)

    def _client(self):  # pragma: no cover - exercise network path only when configured
        if AzureOpenAI is None:
            raise RuntimeError("AzureOpenAI SDK not available. Install 'openai>=1.0' and try again.")
        api_key = self.api_key or os.getenv("AZURE_OPENAI_API_KEY")
        api_version = self.api_version or os.getenv("AZURE_OPENAI_API_VERSION")
        endpoint = self.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not api_key or not api_version or not endpoint:
            raise RuntimeError("Azure OpenAI environment not configured.")
        return AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)

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

    def generate_hint(self, context: Dict[str, Any]) -> str:
        deployment = self.deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
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
        ground_truth = context.get('ground_truth', {}) or {}
        fault_location = ground_truth.get('location', 'unknown')
        network_size = context.get('network_size', 'unknown')
        topology_types = context.get('topology_types', [])
        user_context = context.get('user_context', {}) or {}
        access_point = user_context.get('access_point', 'office Wi-Fi')
        department = user_context.get('department', 'general office')

        # Format topology types for display
        topology_str = ', '.join(topology_types) if topology_types else 'mixed'

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

        weave_enabled = self._init_weave()
        latency_ms = None
        text = ""

        try:  # pragma: no cover - avoid network in unit tests
            client = self._client()
            # Note: GPT-5 only supports temperature=1 (default)
            # GPT-5 is a reasoning model and uses many tokens for internal reasoning
            create_params = {
                "model": deployment,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "n": 1,
            }
            # GPT-5 needs much higher token limit for reasoning + output
            if "gpt-5" in deployment.lower():
                create_params["max_completion_tokens"] = 4096  # High limit for reasoning + output
            else:
                create_params["max_completion_tokens"] = 256
                create_params["temperature"] = 0.7

            start = time.perf_counter()
            resp = client.chat.completions.create(**create_params)
            latency_ms = (time.perf_counter() - start) * 1000.0

            # Debug logging
            import sys
            if hasattr(sys.modules.get('__main__'), '__file__'):
                try:
                    from loguru import logger
                    logger.debug(f"GPT Response object: {resp}")
                    logger.debug(f"GPT Response choices: {resp.choices}")
                    if resp.choices:
                        logger.debug(f"First choice message: {resp.choices[0].message}")
                        logger.debug(f"Message content: {resp.choices[0].message.content}")
                except:
                    pass
            
            text = (resp.choices[0].message.content or "").strip()
            
            # Debug logging
            if hasattr(sys.modules.get('__main__'), '__file__'):
                try:
                    from loguru import logger
                    logger.debug(f"Raw GPT hint before sanitization: '{text}'")
                except:
                    pass
        except Exception as e:
            # Graceful fallback to heuristic if Azure fails
            import sys
            if hasattr(sys.modules.get('__main__'), '__file__'):
                # Only log in non-test environments
                try:
                    from loguru import logger
                    logger.warning(f"Azure GPT hint generation failed, using heuristic fallback: {e}")
                except:
                    pass
            text = HeuristicHintProvider().generate_hint(context)

        sanitized = _sanitize_hint(text, context)

        if weave_enabled:
            _record_hint_generation(
                provider="AzureGptHintProvider",
                deployment=deployment,
                context=context,
                system_prompt=system,
                user_prompt=user,
                raw_response=text,
                sanitized_hint=sanitized,
                latency_ms=latency_ms,
            )

        return sanitized


# --------------------------- Helper selection ---------------------------

def get_default_hint_provider(mode: str = "auto") -> BaseHintProvider:
    mode = (mode or "auto").lower()
    if mode == "heuristic":
        return HeuristicHintProvider()
    if mode == "azure":
        return AzureGptHintProvider()
    # auto
    if _azure_env_present():
        return AzureGptHintProvider()
    return HeuristicHintProvider()


def _azure_env_present() -> bool:
    return all(
        os.getenv(k)
        for k in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_ENDPOINT")
    )


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
