# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark runner for NetHeal assessments.

Runs multiple episodes with solver agents and collects metrics.

Usage:
    python -m netheal.aaa.benchmark --episodes 10 --solver gpt
    python -m netheal.aaa.benchmark --episodes 5 --solver dummy
    python -m netheal.aaa.benchmark --episodes 10 --seed 42 --solver gpt
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from netheal.aaa.mcp_server import EpisodeRuntime, NetHealMCPServer
from netheal.environment.env import NetworkTroubleshootingEnv
from netheal.evaluation.metrics import CompetitionEvaluator, EpisodeMetrics
from netheal.evaluation.wrapper import MetricsCollectorWrapper
from netheal.faults import FaultType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("netheal.benchmark")


@dataclass
class EpisodeResult:
    """Result of a single benchmark episode."""
    episode_index: int
    seed: Optional[int] = None
    ground_truth: Optional[Dict[str, Any]] = None
    diagnosis_submitted: bool = False
    diagnosis_correct: bool = False
    submitted_fault_type: Optional[str] = None
    submitted_location: Optional[str] = None
    turns_taken: int = 0
    time_seconds: float = 0.0
    total_reward: float = 0.0
    error: Optional[str] = None


@dataclass
class BenchmarkResults:
    """Aggregate results from a benchmark run."""
    solver_type: str
    total_episodes: int
    successful_diagnoses: int = 0
    correct_diagnoses: int = 0
    failed_episodes: int = 0
    total_turns: int = 0
    total_time_seconds: float = 0.0
    total_reward: float = 0.0
    episodes: List[EpisodeResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_episodes == 0:
            return 0.0
        return self.successful_diagnoses / self.total_episodes

    @property
    def accuracy(self) -> float:
        if self.successful_diagnoses == 0:
            return 0.0
        return self.correct_diagnoses / self.successful_diagnoses

    @property
    def avg_turns(self) -> float:
        if self.successful_diagnoses == 0:
            return 0.0
        return self.total_turns / self.successful_diagnoses

    @property
    def avg_time(self) -> float:
        if self.successful_diagnoses == 0:
            return 0.0
        return self.total_time_seconds / self.successful_diagnoses

    def to_dict(self) -> Dict[str, Any]:
        return {
            "solver_type": self.solver_type,
            "total_episodes": self.total_episodes,
            "successful_diagnoses": self.successful_diagnoses,
            "correct_diagnoses": self.correct_diagnoses,
            "failed_episodes": self.failed_episodes,
            "success_rate": f"{self.success_rate:.1%}",
            "accuracy": f"{self.accuracy:.1%}",
            "avg_turns": f"{self.avg_turns:.1f}",
            "avg_time_seconds": f"{self.avg_time:.2f}",
            "total_reward": f"{self.total_reward:.2f}",
            "episodes": [
                {
                    "index": ep.episode_index,
                    "success": ep.diagnosis_submitted,
                    "correct": ep.diagnosis_correct,
                    "turns": ep.turns_taken,
                    "time": f"{ep.time_seconds:.2f}s",
                    "ground_truth": ep.ground_truth,
                    "submitted": f"{ep.submitted_fault_type} @ {ep.submitted_location}",
                    "error": ep.error,
                }
                for ep in self.episodes
            ],
        }


async def run_single_episode(
    episode_index: int,
    solver_type: str,
    seed: Optional[int],
    max_devices: int,
    max_steps: int,
    max_turns: int,
    verbose: bool,
) -> EpisodeResult:
    """Run a single benchmark episode."""
    result = EpisodeResult(episode_index=episode_index, seed=seed)

    LOGGER.info("=" * 60)
    LOGGER.info("Episode %d (seed=%s)", episode_index + 1, seed)
    LOGGER.info("=" * 60)

    env = NetworkTroubleshootingEnv(
        max_devices=max_devices,
        max_episode_steps=max_steps,
        enable_user_hints=True,
    )
    wrapped = MetricsCollectorWrapper(env)

    try:
        obs, info = wrapped.reset(seed=seed)
        result.ground_truth = info.get("ground_truth_fault")
        hint = info.get("user_hint", "No hint")

        LOGGER.info("Ground truth: %s", result.ground_truth)
        LOGGER.info("Hint: %s", hint)

        runtime = EpisodeRuntime(env=wrapped, observation=obs, info=info)
        server = NetHealMCPServer(runtime)
        server.start()

        LOGGER.info("MCP server started at %s", server.base_url)

        start_time = time.time()

        try:
            if solver_type == "gpt":
                from netheal.aaa.gpt_agent import run_gpt_agent

                task_context = {
                    "max_steps": max_steps,
                    "task_description": (
                        "Diagnose the network fault. The simulated network has an injected fault "
                        "(device failure, link failure, misconfiguration, or performance degradation). "
                        "Use the diagnostic tools to explore the network, identify the fault, "
                        "and submit your diagnosis using the submit_diagnosis tool."
                    ),
                    "objective": (
                        f"Identify the fault type and location within {max_steps} tool calls. "
                        "Call submit_diagnosis(fault_type, location) with your answer. "
                        "Each tool call (except list_actions) consumes one step from your budget."
                    ),
                    "fault_types": ["device_failure", "link_failure", "misconfiguration", "performance_degradation"],
                }
                if verbose:
                    task_context["ground_truth"] = result.ground_truth

                agent_result = await run_gpt_agent(
                    mcp_url=server.base_url,
                    task_hint=hint,
                    task_context=task_context,
                    max_turns=max_turns,
                    verbose=verbose,
                )
                result.turns_taken = agent_result.get("turns", 0)
                result.diagnosis_submitted = agent_result.get("task_completed", False)

            elif solver_type == "dummy":
                from netheal.aaa.dummy_agent import run_dummy_agent

                agent_result = await run_dummy_agent(
                    base_url=server.http_helper_url,
                    min_steps=3,
                    max_steps=max_turns,
                )
                result.turns_taken = agent_result.get("steps", 0)
                result.diagnosis_submitted = agent_result.get("diagnosis_submitted", False)
            else:
                raise ValueError(f"Unknown solver type: {solver_type}")

            result.time_seconds = time.time() - start_time

            if server._diagnosis_submitted:
                result.diagnosis_submitted = True
                metrics = wrapped.last_episode_metrics
                if metrics:
                    result.diagnosis_correct = metrics.diagnosis_success
                    result.total_reward = metrics.total_reward

                    trace = wrapped.last_episode_trace
                    if trace and trace.actions:
                        last_action = trace.actions[-1]
                        result.submitted_fault_type = last_action.action_type
                        result.submitted_location = (last_action.parameters or {}).get('location')

        finally:
            server.stop()

    except Exception as e:
        LOGGER.error("Episode %d failed: %s", episode_index + 1, e)
        result.error = str(e)

    finally:
        env.close()

    status = "+" if result.diagnosis_correct else ("?" if result.diagnosis_submitted else "-")
    LOGGER.info(
        "Episode %d result: %s (turns=%d, time=%.1fs, reward=%.2f)",
        episode_index + 1, status, result.turns_taken, result.time_seconds, result.total_reward
    )

    return result


async def run_benchmark(
    solver_type: str,
    num_episodes: int,
    seed: Optional[int] = None,
    max_devices: int = 15,
    max_steps: int = 100,
    max_turns: int = 100,
    verbose: bool = False,
    parallel: bool = False,
) -> BenchmarkResults:
    """
    Run a benchmark with multiple episodes.

    Args:
        solver_type: "gpt" or "dummy"
        num_episodes: Number of episodes to run
        seed: Base random seed (each episode gets seed+i)
        max_devices: Max network devices per episode
        max_steps: Environment step budget
        max_turns: Agent turn limit
        verbose: Print detailed logs
        parallel: Run episodes concurrently
    """
    LOGGER.info("Starting NetHeal benchmark")
    LOGGER.info("  Solver: %s", solver_type)
    LOGGER.info("  Episodes: %d", num_episodes)
    LOGGER.info("  Base seed: %s", seed)
    LOGGER.info("  Max devices: %d", max_devices)
    LOGGER.info("  Max steps: %d", max_steps)
    LOGGER.info("  Max turns: %d", max_turns)
    LOGGER.info("  Parallel: %s", parallel)

    results = BenchmarkResults(
        solver_type=solver_type,
        total_episodes=num_episodes,
    )

    if parallel:
        tasks = [
            run_single_episode(
                episode_index=i,
                solver_type=solver_type,
                seed=(seed + i) if seed is not None else None,
                max_devices=max_devices,
                max_steps=max_steps,
                max_turns=max_turns,
                verbose=verbose,
            )
            for i in range(num_episodes)
        ]
        episode_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, ep_result in enumerate(episode_results):
            if isinstance(ep_result, Exception):
                LOGGER.error("Episode %d failed with exception: %s", i + 1, ep_result)
                ep_result = EpisodeResult(episode_index=i, error=str(ep_result))
            results.episodes.append(ep_result)
            _update_aggregate_results(results, ep_result)
    else:
        for i in range(num_episodes):
            episode_seed = (seed + i) if seed is not None else None

            episode_result = await run_single_episode(
                episode_index=i,
                solver_type=solver_type,
                seed=episode_seed,
                max_devices=max_devices,
                max_steps=max_steps,
                max_turns=max_turns,
                verbose=verbose,
            )

            results.episodes.append(episode_result)
            _update_aggregate_results(results, episode_result)

    return results


def _update_aggregate_results(results: BenchmarkResults, episode_result: EpisodeResult) -> None:
    """Update aggregate metrics from an episode result."""
    if episode_result.diagnosis_submitted:
        results.successful_diagnoses += 1
        results.total_turns += episode_result.turns_taken
        results.total_time_seconds += episode_result.time_seconds
        results.total_reward += episode_result.total_reward

        if episode_result.diagnosis_correct:
            results.correct_diagnoses += 1

    if episode_result.error:
        results.failed_episodes += 1


def print_results(results: BenchmarkResults) -> None:
    """Print formatted benchmark results."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Solver: {results.solver_type}")
    print(f"Total Episodes: {results.total_episodes}")
    print("-" * 70)
    print(f"Diagnoses Submitted: {results.successful_diagnoses}/{results.total_episodes} ({results.success_rate:.1%})")
    print(f"Correct Diagnoses: {results.correct_diagnoses}/{results.successful_diagnoses} ({results.accuracy:.1%})")
    print(f"Failed Episodes: {results.failed_episodes}")
    print("-" * 70)
    print(f"Average Turns: {results.avg_turns:.1f}")
    print(f"Average Time: {results.avg_time:.2f}s")
    print(f"Total Reward: {results.total_reward:.2f}")
    print("=" * 70)

    print("\nPer-Episode Results:")
    print("-" * 70)
    for ep in results.episodes:
        status = "+" if ep.diagnosis_correct else ("?" if ep.diagnosis_submitted else "-")
        gt = ep.ground_truth or {}
        gt_str = f"{gt.get('fault_type', '?')} @ {gt.get('location', '?')}"
        print(f"  Episode {ep.episode_index + 1}: {status} | "
              f"Turns: {ep.turns_taken:2d} | "
              f"Time: {ep.time_seconds:5.1f}s | "
              f"Truth: {gt_str}")
        if ep.error:
            print(f"    Error: {ep.error}")


def save_results(results: BenchmarkResults, output_path: Path) -> None:
    """Save results to JSON file."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "results": results.to_dict(),
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    LOGGER.info("Results saved to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run NetHeal benchmark"
    )
    parser.add_argument(
        "--episodes", "-n",
        type=int,
        default=10,
        help="Number of episodes (default: 10)",
    )
    parser.add_argument(
        "--solver", "-s",
        choices=["gpt", "dummy"],
        default="gpt",
        help="Solver type: 'gpt' or 'dummy'",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--max-devices",
        type=int,
        default=6,
        help="Max network devices (default: 6)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=25,
        help="Max environment steps (default: 25)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=25,
        help="Max agent turns (default: 25)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed agent outputs",
    )
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run episodes in parallel",
    )

    args = parser.parse_args()

    results = asyncio.run(run_benchmark(
        solver_type=args.solver,
        num_episodes=args.episodes,
        seed=args.seed,
        max_devices=args.max_devices,
        max_steps=args.max_steps,
        max_turns=args.max_turns,
        verbose=args.verbose,
        parallel=args.parallel,
    ))

    print_results(results)

    if args.output:
        save_results(results, Path(args.output))


if __name__ == "__main__":
    main()
