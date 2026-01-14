# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Green Agent executor for NetHeal AAA assessments.

Orchestrates assessment episodes, manages MCP servers for tool access,
and produces the final scoring payload for the AAA protocol.

A2A Protocol Flow:
    1. External caller POSTs to /tasks with purple agent endpoints
    2. Green agent creates environment and starts MCP server per episode
    3. Green agent notifies purple agents via POST with episode context
    4. Purple agents use MCP tools to diagnose the network fault
    5. Green agent collects results, computes metrics, and returns summary
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

import httpx

from netheal.aaa.mcp_server import EpisodeRuntime, NetHealMCPServer
from netheal.aaa.schemas import (
    AssessmentRequest,
    AssessmentResult,
    AssessmentConfig,
    Artifact,
    EpisodeStart,
    Participant,
    TaskStatus,
    TaskUpdate,
    TaskUpdateLevel,
)
from netheal.environment.env import NetworkTroubleshootingEnv
from netheal.evaluation.aaa import build_aaa_payload
from netheal.evaluation.metrics import CompetitionEvaluator, EpisodeMetrics
from netheal.evaluation.wrapper import MetricsCollectorWrapper

LOGGER = logging.getLogger(__name__)

UpdateCallback = Callable[[TaskUpdate], Awaitable[None]]


@dataclass
class EpisodeOutcome:
    """Result of a completed assessment episode."""

    metrics: Optional[EpisodeMetrics]
    timed_out: bool = False
    error: Optional[str] = None


class NetHealGreenAgent:
    """Executes NetHeal assessments using dynamic MCP servers."""

    def __init__(
        self,
        assessment: AssessmentRequest,
        purple_role: str = "purple",
    ) -> None:
        self.assessment = assessment
        self.config: AssessmentConfig = assessment.config
        self.purple_role = purple_role

        self.evaluator = CompetitionEvaluator()
        self._task_id = assessment.task_id or "netheal-task"
        self._update_cb: Optional[UpdateCallback] = None

    async def run(
        self,
        update_callback: Optional[UpdateCallback] = None,
    ) -> AssessmentResult:
        """
        Execute all requested episodes and return the assessment result.

        Args:
            update_callback: Optional async callback invoked for each TaskUpdate.
        """
        self._update_cb = update_callback
        await self._emit_update(
            message="Starting NetHeal assessment.",
            payload={"config": self.config.model_dump()},
        )

        episode_results: Dict[int, EpisodeOutcome] = {}
        for episode_idx in range(self.config.num_episodes):
            await self._emit_update(
                message=f"Preparing episode {episode_idx + 1}/{self.config.num_episodes}",
                payload={"episode_index": episode_idx},
            )
            result = await self._run_single_episode(episode_idx)
            episode_results[episode_idx] = result

        payload = self._build_final_payload()
        artifacts = [
            Artifact(
                label="aaa_metrics",
                mime_type="application/json",
                data=payload,
            )
        ]
        summary = {
            "episodes": self.evaluator.compute_summary(),
            "timeouts": sum(1 for outcome in episode_results.values() if outcome.timed_out),
            "errors": [o.error for o in episode_results.values() if o.error],
        }

        status = (
            TaskStatus.COMPLETED
            if not summary["errors"] and summary["timeouts"] == 0
            else TaskStatus.FAILED
        )

        await self._emit_update(
            message="Assessment finished.",
            payload={"status": status.value, "summary": summary},
        )

        return AssessmentResult(
            task_id=self._task_id,
            status=status,
            summary=summary,
            artifacts=artifacts,
            metadata={"participants": list(self.assessment.participants.keys())},
        )

    async def _run_single_episode(self, episode_index: int) -> EpisodeOutcome:
        """
        Run a single assessment episode.

        Creates the environment and MCP server, notifies purple agents,
        waits for diagnosis submission, and collects metrics.
        """
        env = NetworkTroubleshootingEnv(
            max_devices=self.config.max_devices,
            max_episode_steps=self.config.max_episode_steps,
            topology_types=self.config.topology_types,
            reward_scaling_factor=self.config.reward_scaling_factor,
            enable_user_hints=self.config.enable_user_hints,
            **self.config.extra_env_options,
        )
        wrapped = MetricsCollectorWrapper(env, evaluator=self.evaluator)

        seed = self._seed_for_episode(episode_index)
        observation, info = wrapped.reset(seed=seed)
        runtime = EpisodeRuntime(env=wrapped, observation=observation, info=info)

        # Support Docker networking: bind to 0.0.0.0, advertise with container hostname
        mcp_host = os.environ.get("MCP_SERVER_HOST", "127.0.0.1")
        mcp_advertised_host = os.environ.get("MCP_SERVER_ADVERTISED_HOST", mcp_host)
        
        # Create callback for tool call notifications
        tool_call_queue: asyncio.Queue = asyncio.Queue()
        
        def on_tool_call(event: Dict[str, Any]) -> None:
            """Callback invoked when MCP server executes a tool."""
            try:
                tool_call_queue.put_nowait(event)
            except asyncio.QueueFull:
                LOGGER.warning("Tool call queue full, dropping event")
        
        mcp_server = NetHealMCPServer(
            runtime, 
            host=mcp_host, 
            advertised_host=mcp_advertised_host,
            on_tool_call=on_tool_call,
        )
        try:
            mcp_server.start()
        except Exception as exc:
            LOGGER.exception("Failed to start MCP server: %s", exc)
            return EpisodeOutcome(metrics=None, error=str(exc))

        # Build episode_start WITH ground_truth for dashboard visualization
        episode_start_for_dashboard = self._build_episode_start(
            episode_index, runtime, mcp_server, include_ground_truth=True
        )
        
        # Build episode_start WITHOUT ground_truth for purple agent (no leakage!)
        episode_start_for_solver = self._build_episode_start(
            episode_index, runtime, mcp_server, include_ground_truth=False
        )

        await self._emit_update(
            message="MCP server ready for episode.",
            payload={"episode_start": episode_start_for_dashboard.model_dump()},
        )

        await self._notify_purple_agents(episode_start_for_solver)

        try:
            metrics = await self._wait_for_completion(
                wrapped, 
                timeout=self.config.timeout_seconds,
                tool_call_queue=tool_call_queue,
                episode_index=episode_index,
            )
            if metrics is None:
                await self._emit_update(
                    message="Episode timed out waiting for solver diagnosis.",
                    level=TaskUpdateLevel.WARNING,
                    payload={"episode_index": episode_index},
                )
                return EpisodeOutcome(metrics=None, timed_out=True)

            await self._emit_update(
                message="Episode completed.",
                payload={
                    "episode_index": episode_index,
                    "metrics": {
                        "diagnosis_success": metrics.diagnosis_success,
                        "steps": metrics.steps,
                        "composite_episode_score": metrics.composite_episode_score,
                    },
                },
            )
            return EpisodeOutcome(metrics=metrics)
        except Exception as exc:
            LOGGER.exception("Episode %s encountered an error: %s", episode_index, exc)
            await self._emit_update(
                message="Episode failed.",
                level=TaskUpdateLevel.ERROR,
                payload={"episode_index": episode_index, "error": str(exc)},
            )
            return EpisodeOutcome(metrics=None, error=str(exc))
        finally:
            mcp_server.stop()
            wrapped.env.close()

    async def _wait_for_completion(
        self,
        wrapped_env: MetricsCollectorWrapper,
        timeout: float,
        tool_call_queue: Optional[asyncio.Queue] = None,
        episode_index: int = 0,
    ) -> Optional[EpisodeMetrics]:
        """Poll until episode finishes or timeout elapses, emitting tool call events."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        poll_interval = 0.1  # Faster polling for more responsive updates

        while loop.time() < deadline:
            # Process any pending tool call events
            if tool_call_queue:
                while not tool_call_queue.empty():
                    try:
                        tool_event = tool_call_queue.get_nowait()
                        await self._emit_update(
                            message=f"Tool call: {tool_event.get('tool_name', 'unknown')}",
                            payload={
                                "episode_index": episode_index,
                                "tool_call": tool_event,
                            },
                        )
                    except asyncio.QueueEmpty:
                        break
            
            metrics = wrapped_env.last_episode_metrics
            if metrics is not None:
                return metrics
            await asyncio.sleep(poll_interval)

        return None

    async def _notify_purple_agents(self, episode_start: EpisodeStart) -> None:
        """
        Notify purple agents about the episode via A2A POST.

        Purple agents receive episode context and connect to the MCP server.
        """
        if not self.assessment.participants:
            LOGGER.info("No purple agent endpoints configured; using SSE-only mode.")
            return

        async with httpx.AsyncClient(timeout=30.0) as client:
            for role, participant in self.assessment.participants.items():
                if role == "green":
                    continue

                endpoint = str(participant.endpoint).rstrip("/")
                tasks_url = f"{endpoint}/tasks"
                
                # Build the A2A request payload
                a2a_request = {
                    "task_id": f"{self._task_id}_ep{episode_start.episode_index}",
                    "episode_start": episode_start.model_dump(mode="json"),
                }
                
                # Emit A2A request event for dashboard visibility
                await self._emit_update(
                    message=f"A2A: Sending EpisodeStart to {role}",
                    payload={
                        "a2a_message": {
                            "type": "request",
                            "direction": "green_to_purple",
                            "endpoint": tasks_url,
                            "method": "POST",
                            "body": a2a_request,
                        }
                    },
                )

                try:
                    LOGGER.info("Notifying purple agent %s at %s", role, tasks_url)
                    response = await client.post(tasks_url, json=a2a_request)
                    
                    # Emit A2A response event
                    response_data = None
                    try:
                        response_data = response.json()
                    except:
                        response_data = {"raw": response.text[:500]}
                    
                    await self._emit_update(
                        message=f"A2A: Response from {role} (HTTP {response.status_code})",
                        payload={
                            "a2a_message": {
                                "type": "response",
                                "direction": "purple_to_green",
                                "status_code": response.status_code,
                                "body": response_data,
                            }
                        },
                    )
                    
                    if response.status_code < 400:
                        LOGGER.info("Purple agent %s acknowledged episode.", role)
                    else:
                        LOGGER.warning(
                            "Purple agent %s returned %s: %s",
                            role,
                            response.status_code,
                            response.text[:200],
                        )
                except httpx.RequestError as exc:
                    LOGGER.warning(
                        "Failed to notify purple agent %s: %s (continuing anyway)",
                        role,
                        exc,
                    )
                    await self._emit_update(
                        message=f"A2A: Failed to reach {role}",
                        level=TaskUpdateLevel.WARNING,
                        payload={
                            "a2a_message": {
                                "type": "error",
                                "direction": "green_to_purple",
                                "error": str(exc),
                            }
                        },
                    )

    async def _emit_update(
        self,
        message: str,
        level: TaskUpdateLevel = TaskUpdateLevel.INFO,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send a task update via the registered callback."""
        if self._update_cb is None:
            return

        update = TaskUpdate(
            task_id=self._task_id,
            level=level,
            message=message,
            payload=payload or {},
        )
        await self._update_cb(update)

    def _build_episode_start(
        self,
        episode_index: int,
        runtime: EpisodeRuntime,
        server: NetHealMCPServer,
        include_ground_truth: bool = False,
    ) -> EpisodeStart:
        """Construct the EpisodeStart message.
        
        Args:
            include_ground_truth: If True, include ground truth in extra dict.
                Should be False when sending to purple agent (solver),
                True when emitting to dashboard for visualization.
        """
        info = runtime.info or {}
        
        # Build extra dict - only include ground_truth for dashboard, NOT for purple agent
        extra_data = {
            "http_helper_url": server.http_helper_url,
            "tools_endpoint": f"{server.http_helper_url}/tools",
            "mcp_url": server.base_url,
        }
        
        # Only include ground truth for internal use (dashboard), never sent to solver
        if include_ground_truth:
            extra_data["ground_truth"] = info.get("ground_truth_fault")
        
        return EpisodeStart(
            episode_index=episode_index,
            total_episodes=self.config.num_episodes,
            mcp_server_url=server.http_helper_url,
            hint=info.get("user_hint"),
            network_size=info.get("network_size"),
            seed=self._seed_for_episode(episode_index),
            max_steps=self.config.max_episode_steps,
            task_description=(
                "Diagnose the network fault. The simulated network has an injected fault "
                "(device failure, link failure, misconfiguration, or performance degradation). "
                "Use the diagnostic tools to explore the network, identify the fault, "
                "and submit your diagnosis using the submit_diagnosis tool."
            ),
            objective=(
                f"Identify the fault type and location within {self.config.max_episode_steps} tool calls. "
                "Call submit_diagnosis(fault_type, location) with your answer. "
                "Each tool call (except list_actions) consumes one step from your budget."
            ),
            extra=extra_data,
        )

    def _seed_for_episode(self, episode_index: int) -> Optional[int]:
        if self.config.seed is None:
            return None
        return self.config.seed + episode_index

    def _build_final_payload(self) -> Dict[str, Any]:
        purple_agent_id = self._purple_agent_identifier()
        metadata = {
            "task_id": self._task_id,
            "config": self.config.model_dump(),
        }
        return build_aaa_payload(
            evaluator=self.evaluator,
            purple_agent_id=purple_agent_id,
            green_agent_name="netheal_green_mcp_v1",
            metadata=metadata,
        )

    def _purple_agent_identifier(self) -> str:
        participant = self.assessment.participants.get(self.purple_role)
        if participant:
            return participant.endpoint
        return f"{self.purple_role}_agent"


__all__ = ["NetHealGreenAgent"]
