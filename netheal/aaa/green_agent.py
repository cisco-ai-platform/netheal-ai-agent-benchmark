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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, List

import httpx
from tqdm import tqdm

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
from netheal.scenario import create_env_from_snapshot, load_snapshot_episodes

LOGGER = logging.getLogger(__name__)

UpdateCallback = Callable[[TaskUpdate], Awaitable[None]]


@dataclass
class EpisodeOutcome:
    """Result of a completed assessment episode."""

    metrics: Optional[EpisodeMetrics]
    timed_out: bool = False
    error: Optional[str] = None
    attempts: int = 1


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
        self._snapshots: Optional[List[Dict[str, Any]]] = None

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

        self._load_snapshots_if_needed()

        episode_results: Dict[int, EpisodeOutcome] = {}
        concurrency = max(1, min(self.config.episode_concurrency, self.config.num_episodes))
        await self._emit_update(
            message=f"Running {self.config.num_episodes} episodes with concurrency {concurrency}.",
            payload={"episode_concurrency": concurrency},
        )

        # Progress tracking counters
        completed_count = 0
        success_count = 0
        timeout_count = 0
        error_count = 0

        # Create progress bar for CI-friendly output
        progress_bar = tqdm(
            total=self.config.num_episodes,
            desc="Episodes",
            unit="ep",
            file=sys.stderr,
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

        async def update_progress(outcome: EpisodeOutcome) -> None:
            """Update progress bar and counters after episode completion."""
            nonlocal completed_count, success_count, timeout_count, error_count
            completed_count += 1
            if outcome.metrics and outcome.metrics.diagnosis_success:
                success_count += 1
            if outcome.timed_out:
                timeout_count += 1
            if outcome.error:
                error_count += 1
            progress_bar.update(1)
            progress_bar.set_postfix({
                "ok": success_count,
                "fail": completed_count - success_count,
            })

        try:
            if concurrency == 1:
                for episode_idx in range(self.config.num_episodes):
                    await self._emit_update(
                        message=f"Preparing episode {episode_idx + 1}/{self.config.num_episodes}",
                        payload={"episode_index": episode_idx},
                    )
                    result = await self._run_episode_with_retries(episode_idx)
                    episode_results[episode_idx] = result
                    await update_progress(result)
            else:
                semaphore = asyncio.Semaphore(concurrency)

                async def run_episode(episode_idx: int) -> None:
                    async with semaphore:
                        await self._emit_update(
                            message=f"Preparing episode {episode_idx + 1}/{self.config.num_episodes}",
                            payload={"episode_index": episode_idx},
                        )
                        result = await self._run_episode_with_retries(episode_idx)
                        episode_results[episode_idx] = result
                        await update_progress(result)

                tasks = [
                    asyncio.create_task(run_episode(episode_idx))
                    for episode_idx in range(self.config.num_episodes)
                ]
                await asyncio.gather(*tasks)
        finally:
            progress_bar.close()

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
            "retries_used": sum(max(0, o.attempts - 1) for o in episode_results.values()),
            "episodes_retried": sum(1 for o in episode_results.values() if o.attempts > 1),
        }

        # Print formatted summary to stderr for CI visibility
        ep_summary = summary["episodes"]
        print("\n" + "=" * 60, file=sys.stderr)
        print("ASSESSMENT SUMMARY", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"Episodes:    {self.config.num_episodes}", file=sys.stderr)
        print(f"Successful:  {ep_summary.get('successes', 0)}", file=sys.stderr)
        print(f"Failed:      {ep_summary.get('failures', 0)}", file=sys.stderr)
        print(f"Timeouts:    {summary['timeouts']}", file=sys.stderr)
        print(f"Errors:      {len(summary['errors'])}", file=sys.stderr)
        print(f"Retries:     {summary['retries_used']}", file=sys.stderr)
        if "success_rate" in ep_summary:
            print(f"Success Rate: {ep_summary['success_rate']:.1%}", file=sys.stderr)
        if "mean_composite_score" in ep_summary:
            print(f"Mean Score:   {ep_summary['mean_composite_score']:.3f}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        def _exceeds_limit(
            count: int, limit: Optional[int], fail_on_any: bool
        ) -> bool:
            if limit is not None:
                return count > limit
            if fail_on_any:
                return count > 0
            return False

        timeout_count = summary["timeouts"]
        error_count = len(summary["errors"])
        fail_timeouts = _exceeds_limit(
            timeout_count, self.config.max_timeouts, self.config.fail_on_timeout
        )
        fail_errors = _exceeds_limit(
            error_count, self.config.max_errors, self.config.fail_on_error
        )
        status = TaskStatus.FAILED if (fail_timeouts or fail_errors) else TaskStatus.COMPLETED

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

    async def _run_episode_with_retries(self, episode_index: int) -> EpisodeOutcome:
        """Run an episode with optional retries on timeout or error."""
        max_attempts = self.config.episode_retry_limit + 1
        last_outcome: Optional[EpisodeOutcome] = None

        for attempt in range(max_attempts):
            if attempt > 0:
                await self._emit_update(
                    message=(
                        f"Retrying episode {episode_index + 1}/{self.config.num_episodes} "
                        f"(attempt {attempt + 1}/{max_attempts})"
                    ),
                    level=TaskUpdateLevel.WARNING,
                    payload={
                        "episode_index": episode_index,
                        "attempt": attempt + 1,
                        "max_attempts": max_attempts,
                    },
                )

            outcome = await self._run_single_episode(episode_index)
            outcome.attempts = attempt + 1
            last_outcome = outcome

            if not outcome.timed_out and not outcome.error:
                # Success: add final metrics to evaluator
                if outcome.metrics is not None:
                    self.evaluator.add_episode_metrics(outcome.metrics)
                return outcome

        # All retries exhausted: add final attempt's metrics to evaluator
        final_outcome = last_outcome or EpisodeOutcome(metrics=None, error="Episode failed to run.")
        if final_outcome.metrics is not None:
            self.evaluator.add_episode_metrics(final_outcome.metrics)
        return final_outcome

    async def _run_single_episode(self, episode_index: int) -> EpisodeOutcome:
        """
        Run a single assessment episode.

        Creates the environment and MCP server, notifies purple agents,
        waits for diagnosis submission, and collects metrics.
        """
        snapshot = self._snapshot_for_episode(episode_index)
        if snapshot:
            env = create_env_from_snapshot(snapshot)
            # skip_evaluator=True: metrics added after retries complete
            wrapped = MetricsCollectorWrapper(
                env, evaluator=self.evaluator, skip_evaluator=True
            )
            observation = env.observation.to_dict()
            info = env._get_info()
            wrapped._start_new_trace(observation, info, seed=snapshot.get("seed"))
        else:
            env = NetworkTroubleshootingEnv(
                min_devices=self.config.min_devices,
                max_devices=self.config.max_devices,
                max_episode_steps=self.config.max_episode_steps,
                topology_types=self.config.topology_types,
                reward_scaling_factor=self.config.reward_scaling_factor,
                enable_user_hints=self.config.enable_user_hints,
                fault_sampling_strategy=self.config.fault_sampling_strategy,
                fault_weights=self.config.fault_weights or None,
                latency_multiplier_range=self._parse_latency_range(
                    self.config.latency_multiplier_range
                ) or (10.0, 20.0),
                **self.config.extra_env_options,
            )
            # skip_evaluator=True: metrics added after retries complete
            wrapped = MetricsCollectorWrapper(
                env, evaluator=self.evaluator, skip_evaluator=True
            )
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
        
        mcp_log_level = os.environ.get("MCP_SERVER_LOG_LEVEL", "warning")
        mcp_server = NetHealMCPServer(
            runtime,
            host=mcp_host,
            advertised_host=mcp_advertised_host,
            log_level=mcp_log_level,
            on_tool_call=on_tool_call,
        )
        try:
            mcp_server.start()
        except Exception as exc:
            LOGGER.exception("Failed to start MCP server: %s", exc)
            return EpisodeOutcome(metrics=None, error=str(exc))

        # Build episode_start WITH ground_truth for dashboard visualization
        episode_start_for_dashboard = self._build_episode_start(
            episode_index,
            runtime,
            mcp_server,
            include_ground_truth=True,
            seed_override=snapshot.get("seed") if snapshot else None,
        )
        
        # Build episode_start WITHOUT ground_truth for purple agent (no leakage!)
        episode_start_for_solver = self._build_episode_start(
            episode_index,
            runtime,
            mcp_server,
            include_ground_truth=False,
            seed_override=snapshot.get("seed") if snapshot else None,
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
                try:
                    wrapped._finalize_trace(
                        final_observation=runtime.observation,
                        final_info=runtime.info or {},
                        terminated=False,
                        truncated=True,
                    )
                except Exception:
                    pass
                return EpisodeOutcome(metrics=wrapped.last_episode_metrics, timed_out=True)

            await self._emit_update(
                message="Episode completed.",
                payload={
                    "episode_index": episode_index,
                    "metrics": {
                        "diagnosis_success": metrics.diagnosis_success,
                        "steps": metrics.steps,
                        "total_reward": metrics.total_reward,
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
        seed_override: Optional[int] = None,
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
            seed=seed_override if seed_override is not None else self._seed_for_episode(episode_index),
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

    def _load_snapshots_if_needed(self) -> None:
        if not self.config.use_snapshots:
            return
        if self._snapshots is not None:
            return
        if self.config.snapshot_url:
            raise RuntimeError("snapshot_url is not supported yet; use snapshot_path.")
        if not self.config.snapshot_path:
            raise RuntimeError("snapshot_path is required when use_snapshots is true.")

        snapshot_path = Path(self.config.snapshot_path)
        if not snapshot_path.is_absolute():
            snapshot_path = Path.cwd() / snapshot_path
        self._snapshots = load_snapshot_episodes(snapshot_path)
        if not self._snapshots:
            raise RuntimeError(f"No snapshots found in {snapshot_path}")

    def _snapshot_for_episode(self, episode_index: int) -> Optional[Dict[str, Any]]:
        if not self.config.use_snapshots:
            return None
        self._load_snapshots_if_needed()
        if not self._snapshots:
            return None
        return self._snapshots[episode_index % len(self._snapshots)]

    @staticmethod
    def _parse_latency_range(
        value: Optional[List[float]],
    ) -> Optional[tuple[float, float]]:
        if not value or len(value) < 2:
            return None
        return (float(value[0]), float(value[1]))

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
            return str(participant.endpoint)

        desired_roles = {self.purple_role, f"{self.purple_role}_agent"}
        for candidate in self.assessment.participants.values():
            if candidate.role in desired_roles:
                return str(candidate.endpoint)

        if len(self.assessment.participants) == 1:
            return str(next(iter(self.assessment.participants.values())).endpoint)

        return f"{self.purple_role}_agent"


__all__ = ["NetHealGreenAgent"]
