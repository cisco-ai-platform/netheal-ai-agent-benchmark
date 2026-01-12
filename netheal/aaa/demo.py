"""
Interactive demo for NetHeal AAA assessments.

Demonstrates the full A2A protocol flow with rich terminal output,
showing MCP tool calls, green/purple agent communication, and scoring.

Usage:
    python -m netheal.aaa.demo --seed 42 --devices 6
    python -m netheal.aaa.demo --seed 42 --devices 6 --step  # Step-by-step
    python -m netheal.aaa.demo --seed 42 --devices 6 --verbose
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from typing import Any, Dict, List, Optional

import httpx
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("netheal.aaa.mcp_server").setLevel(logging.WARNING)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.syntax import Syntax
    from rich.rule import Rule
    from rich.tree import Tree
    from rich import box
except ImportError:
    raise ImportError("Install 'rich' for the demo: pip install rich")

from netheal.aaa.mcp_server import EpisodeRuntime, NetHealMCPServer
from netheal.aaa.schemas import (
    AssessmentConfig,
    AssessmentRequest,
    AssessmentResult,
    Artifact,
    Participant,
    EpisodeStart,
    TaskStatus,
    TaskUpdate,
    TaskUpdateLevel,
)
from netheal.environment.env import NetworkTroubleshootingEnv
from netheal.evaluation.wrapper import MetricsCollectorWrapper
from netheal.evaluation.metrics import CompetitionEvaluator
from netheal.evaluation.aaa import build_aaa_payload

console = Console()


class DemoRunner:
    """Orchestrates the green/purple agent demo with rich output."""

    def __init__(
        self,
        seed: Optional[int] = None,
        max_devices: int = 6,
        max_episode_steps: int = 15,
        min_steps: int = 4,
        max_steps: int = 10,
        step_by_step: bool = False,
        delay: float = 0.5,
        verbose: bool = False,
    ) -> None:
        self.seed = seed
        self.max_devices = max_devices
        self.max_episode_steps = max_episode_steps
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.step_by_step = step_by_step
        self.delay = delay
        self.verbose = verbose

        self.total_reward: float = 0.0
        self.step_count: int = 0
        self.ground_truth: Optional[Dict[str, Any]] = None
        self.user_hint: Optional[str] = None
        self.evaluator: Optional[CompetitionEvaluator] = None
        self.mcp_calls: List[Dict[str, Any]] = []
        self.task_id: str = f"task-{seed or 'random'}-{int(time.time())}"
        self.task_updates: List[Dict[str, Any]] = []

    def run(self) -> None:
        """Run the demo."""
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        """Execute the demo flow."""
        if self.verbose:
            self._display_architecture()

        self._display_a2a_request()

        env = NetworkTroubleshootingEnv(
            max_devices=self.max_devices,
            max_episode_steps=self.max_episode_steps,
            enable_user_hints=True,
        )
        self.evaluator = CompetitionEvaluator()
        wrapped = MetricsCollectorWrapper(env, evaluator=self.evaluator)

        observation, info = wrapped.reset(seed=self.seed)
        self.ground_truth = info.get("ground_truth_fault")
        self.user_hint = info.get("user_hint")

        runtime = EpisodeRuntime(env=wrapped, observation=observation, info=info)
        mcp_server = NetHealMCPServer(runtime, log_level="error")

        try:
            mcp_server.start()
            self._display_episode_start(info, mcp_server)

            await self._run_dummy_agent(mcp_server.http_helper_url)

            metrics = wrapped.last_episode_metrics
            self._display_episode_end(metrics)

            self._display_aaa_payload()

        finally:
            mcp_server.stop()
            env.close()

    def _display_architecture(self) -> None:
        """Display the AAA architecture overview."""
        console.print()
        console.print(Rule("[bold cyan]AAA ARCHITECTURE[/bold cyan]", style="cyan"))
        console.print()

        tree = Tree("[bold]AAA Assessment Framework[/bold]")

        a2a = tree.add("[yellow]A2A Protocol Layer[/yellow]")
        a2a.add("POST /tasks - Submit assessment request")
        a2a.add("GET /tasks/{id} - Check task status")
        a2a.add("GET /tasks/{id}/stream - SSE updates")
        a2a.add("GET /.well-known/agent.json - Agent card")

        green = tree.add("[green]Green Agent (Environment Controller)[/green]")
        green.add("Creates NetworkTroubleshootingEnv")
        green.add("Injects random fault")
        green.add("Starts MCP server per episode")
        green.add("Collects metrics & builds payload")

        mcp = tree.add("[blue]MCP Server (Tool Interface)[/blue]")
        mcp.add("GET /state - Current observation")
        mcp.add("GET /actions - Valid actions list")
        mcp.add("POST /tools/scan_network")
        mcp.add("POST /tools/discover_neighbors?device=...")
        mcp.add("POST /tools/ping?source=...&destination=...")
        mcp.add("POST /tools/traceroute?source=...&destination=...")
        mcp.add("POST /tools/check_status?device=...")
        mcp.add("POST /tools/check_interfaces?device=...")
        mcp.add("POST /tools/submit_diagnosis?fault_type=...&location=...")

        purple = tree.add("[magenta]Purple Agent (Solver)[/magenta]")
        purple.add("Receives MCP server URL via A2A")
        purple.add("Calls diagnostic tools")
        purple.add("Submits final diagnosis")

        console.print(Panel(tree, border_style="cyan", box=box.ROUNDED))

        if self.step_by_step:
            console.print("[dim]Press Enter to continue...[/dim]")
            input()

    def _display_a2a_request(self) -> None:
        """Display the A2A assessment request."""
        console.print()
        console.print(Rule("[bold yellow]A2A: POST /tasks[/bold yellow]", style="yellow"))
        console.print()

        request = AssessmentRequest(
            task_id=self.task_id,
            participants={
                "purple": Participant(
                    role="purple_agent",
                    endpoint="http://localhost:9000/purple",  # type: ignore
                )
            },
            config=AssessmentConfig(
                num_episodes=1,
                max_devices=self.max_devices,
                max_episode_steps=self.max_episode_steps,
                seed=self.seed,
                enable_user_hints=True,
            ),
        )

        console.print("[magenta]Request Body:[/magenta]")
        request_json = json.dumps(request.model_dump(), indent=2, default=str)
        syntax = Syntax(request_json, "json", theme="monokai", line_numbers=False)
        console.print(Panel(syntax, title="[yellow]AssessmentRequest[/yellow]", border_style="yellow", box=box.ROUNDED))

        console.print()
        console.print("[green]Response (201 Created):[/green]")
        response = {"task_id": self.task_id, "status": "pending"}
        response_json = json.dumps(response, indent=2)
        console.print(Panel(
            Syntax(response_json, "json", theme="monokai", line_numbers=False),
            title="[green]Task Created[/green]",
            border_style="green",
            box=box.ROUNDED,
        ))

        console.print()
        console.print("[dim]Green Agent will now:[/dim]")
        console.print("  1. Create the environment")
        console.print("  2. Inject a random fault")
        console.print("  3. Start MCP server")
        console.print("  4. Notify Purple Agent via EpisodeStart")
        console.print()

        if self.step_by_step:
            console.print("[dim]Press Enter to continue...[/dim]")
            input()

    def _display_a2a_task_update(
        self,
        message: str,
        payload: Optional[Dict[str, Any]] = None,
        level: TaskUpdateLevel = TaskUpdateLevel.INFO,
        force_display: bool = False,
    ) -> None:
        """Display a TaskUpdate SSE message."""
        if not self.verbose and not force_display:
            return

        update = TaskUpdate(
            task_id=self.task_id,
            level=level,
            message=message,
            payload=payload or {},
        )
        self.task_updates.append(update.model_dump())

        level_color = {"info": "blue", "warning": "yellow", "error": "red"}.get(level.value, "white")

        console.print()
        console.print(f"[dim]---- A2A: SSE event=update ----[/dim]")
        console.print(f"[{level_color}]data:[/{level_color}]")
        update_json = json.dumps(update.model_dump(), indent=2, default=str)
        console.print(Syntax(update_json, "json", theme="monokai", line_numbers=False))

    def _display_episode_start(self, info: Dict[str, Any], mcp_server: NetHealMCPServer) -> None:
        """Display episode start panel."""
        gt = self.ground_truth or {}
        fault_type = gt.get("type", "unknown")
        fault_location = gt.get("location", "unknown")

        if self.verbose:
            console.print()
            console.print(Rule("[bold yellow]A2A: GET /tasks/{id}/stream (SSE)[/bold yellow]", style="yellow"))
            self._display_a2a_task_update(
                "Starting NetHeal assessment.",
                {"config": {"num_episodes": 1, "max_devices": self.max_devices}}
            )

        console.print()
        console.print(Rule("[bold green]GREEN AGENT: EPISODE START[/bold green]", style="green"))
        console.print()

        lines = [
            f"[bold cyan]NETHEAL EPISODE[/bold cyan]",
            "",
            f"[dim]Seed:[/dim] {self.seed or 'random'}  [dim]Devices:[/dim] {info.get('network_size', '?')}  [dim]Max Steps:[/dim] {self.max_episode_steps}",
            "",
            f"[yellow]Injected Fault:[/yellow] [bold]{fault_type}[/bold] @ [bold]{fault_location}[/bold]",
            "",
        ]

        if self.user_hint:
            hint_text = self.user_hint[:80] + "..." if len(self.user_hint) > 80 else self.user_hint
            lines.append(f"[green]User Hint:[/green] \"{hint_text}\"")
        else:
            lines.append("[dim]User Hint: (none)[/dim]")

        console.print(Panel(
            "\n".join(lines),
            border_style="green",
            box=box.ROUNDED,
            padding=(1, 2),
        ))

        if self.verbose:
            self._display_a2a_task_update(
                "Preparing episode 1/1",
                {"episode_index": 0}
            )

        console.print()
        console.print("[bold blue]MCP SERVER STARTED[/bold blue]")

        episode_start = EpisodeStart(
            episode_index=0,
            total_episodes=1,
            mcp_server_url=mcp_server.base_url,  # type: ignore
            hint=self.user_hint,
            network_size=info.get("network_size"),
            seed=self.seed,
            extra={"ground_truth": gt, "http_helper_url": mcp_server.http_helper_url},
        )

        if self.verbose:
            self._display_a2a_task_update(
                "MCP server ready for episode.",
                {"episode_start": episode_start.model_dump()}
            )

        console.print()
        console.print("[dim]A2A sends EpisodeStart to Purple Agent:[/dim]")
        start_json = json.dumps(episode_start.model_dump(), indent=2, default=str)
        syntax = Syntax(start_json, "json", theme="monokai", line_numbers=False)
        console.print(Panel(
            syntax,
            title="[blue]EpisodeStart Message[/blue]",
            border_style="blue",
            box=box.ROUNDED,
        ))

        console.print()
        console.print("[dim]Purple Agent connects to MCP server at:[/dim]")
        console.print(f"  [cyan]{mcp_server.http_helper_url}[/cyan]")
        console.print()

        if self.step_by_step:
            console.print("[dim]Press Enter to start troubleshooting...[/dim]")
            input()

    def _display_mcp_call(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        response: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Display an MCP server call."""
        if not self.verbose:
            return

        if params:
            param_str = "&".join(f"{k}={v}" for k, v in params.items())
            full_endpoint = f"{endpoint}?{param_str}"
        else:
            full_endpoint = endpoint

        console.print()
        console.print(f"[dim]---- MCP Call ----[/dim]")
        console.print(f"[magenta]{method}[/magenta] [blue]{full_endpoint}[/blue]")

        if response and self.verbose:
            condensed = {
                "reward": response.get("reward"),
                "terminated": response.get("terminated"),
                "info": {
                    k: v for k, v in response.get("info", {}).items()
                    if k in ["action_result", "discovered_devices", "episode_done"]
                }
            }
            resp_json = json.dumps(condensed, indent=2, default=str)
            console.print(f"[green]Response:[/green]")
            console.print(Syntax(resp_json, "json", theme="monokai", line_numbers=False))

    def _display_step(
        self,
        step_num: int,
        total_steps: int,
        tool_name: str,
        params: Dict[str, Any],
        result: Dict[str, Any],
        endpoint: str,
    ) -> None:
        """Display a single step."""
        reward = result.get("reward", 0.0)
        self.total_reward += reward
        self.step_count += 1

        self.mcp_calls.append({
            "step": step_num,
            "tool": tool_name,
            "endpoint": endpoint,
            "params": params,
            "reward": reward,
        })

        console.print()
        console.print(Rule(f"[bold]Step {step_num}/{total_steps}[/bold]", style="blue"))

        if self.verbose:
            self._display_mcp_call("POST", endpoint, params, result)

        tool_text = Text()
        tool_text.append("Tool: ", style="dim")
        tool_text.append(tool_name, style="bold cyan")
        console.print(tool_text)

        if params:
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            console.print(f"[dim]Params:[/dim] {param_str}")

        self._display_result_summary(tool_name, result)

        reward_color = "green" if reward >= 0 else "red"
        console.print(
            f"[dim]Reward:[/dim] [{reward_color}]{reward:+.2f}[/{reward_color}]  "
            f"[dim]Total:[/dim] [{reward_color}]{self.total_reward:+.2f}[/{reward_color}]"
        )

        if self.step_by_step:
            console.print()
            console.print("[dim]Press Enter to continue...[/dim]")
            input()
        else:
            time.sleep(self.delay)

    def _display_result_summary(self, tool_name: str, result: Dict[str, Any]) -> None:
        """Display summary of tool result."""
        info = result.get("info", {})

        if tool_name == "scan_network":
            discovered = info.get("discovered_devices", 0)
            console.print(f"[green]Discovered:[/green] {discovered} devices")

        elif tool_name == "discover_neighbors":
            console.print("[green]Neighbors discovered[/green]")

        elif tool_name == "ping":
            action_result = info.get("action_result", {})
            result_data = action_result.get("result", {})
            success = result_data.get("success", False)
            if success:
                data = result_data.get("data", {})
                latency = data.get("latency_ms", "?")
                console.print(f"[green]SUCCESS[/green] (latency: {latency}ms)")
            else:
                console.print("[red]FAILED[/red]")

        elif tool_name == "traceroute":
            action_result = info.get("action_result", {})
            result_data = action_result.get("result", {})
            success = result_data.get("success", False)
            if success:
                data = result_data.get("data", {})
                path = data.get("path", [])
                console.print(f"[green]Path:[/green] {' -> '.join(path) if path else '(empty)'}")
            else:
                console.print("[red]FAILED[/red]")

        elif tool_name == "check_status":
            action_result = info.get("action_result", {})
            result_data = action_result.get("result", {})
            data = result_data.get("data", {})
            status = data.get("status", "unknown")
            status_color = "green" if status == "active" else "red"
            console.print(f"[{status_color}]Status: {status}[/{status_color}]")

        elif tool_name == "check_interfaces":
            action_result = info.get("action_result", {})
            result_data = action_result.get("result", {})
            data = result_data.get("data", {})
            interfaces = data.get("interfaces", {})
            up_count = sum(1 for iface in interfaces.values() if iface.get("status") == "up")
            down_count = len(interfaces) - up_count
            console.print(f"Interfaces: [green]{up_count} up[/green], [red]{down_count} down[/red]")

        elif tool_name == "submit_diagnosis":
            terminated = result.get("terminated", False)
            if terminated:
                console.print("[bold]Episode terminated[/bold]")

    def _display_episode_end(self, metrics: Optional[Any]) -> None:
        """Display episode end panel."""
        gt = self.ground_truth or {}
        gt_type = gt.get("type", "unknown")
        gt_location = gt.get("location", "unknown")

        success = False
        composite_score = 0.0
        if metrics:
            success = getattr(metrics, "diagnosis_success", False)
            composite_score = getattr(metrics, "composite_episode_score", 0.0)

        if self.verbose:
            self._display_a2a_task_update(
                "Episode completed.",
                {
                    "episode_index": 0,
                    "metrics": {
                        "diagnosis_success": success,
                        "steps": self.step_count,
                        "composite_episode_score": composite_score,
                    },
                }
            )

        console.print()
        console.print(Rule("[bold]EPISODE COMPLETE[/bold]", style="cyan"))

        result_text = "[bold green]CORRECT[/bold green]" if success else "[bold red]INCORRECT[/bold red]"
        reward_color = "green" if self.total_reward > 0 else "red"

        lines = [
            "[bold cyan]EPISODE RESULTS[/bold cyan]",
            "",
            f"[dim]Steps taken:[/dim] {self.step_count}",
            "",
            f"[dim]Ground Truth:[/dim] [bold]{gt_type}[/bold] @ [bold]{gt_location}[/bold]",
            "",
            f"[dim]Result:[/dim] {result_text}",
            "",
            f"[dim]Final Reward:[/dim] [{reward_color}]{self.total_reward:+.2f}[/{reward_color}]",
        ]

        border_style = "green" if success else "red"
        console.print(Panel(
            "\n".join(lines),
            border_style=border_style,
            box=box.ROUNDED,
            padding=(1, 2),
        ))

        if self.verbose and self.mcp_calls:
            console.print()
            console.print("[bold]MCP Call Summary:[/bold]")
            table = Table(box=box.SIMPLE)
            table.add_column("Step", style="dim")
            table.add_column("Tool", style="cyan")
            table.add_column("Endpoint", style="blue")
            table.add_column("Reward", style="yellow")

            for call in self.mcp_calls:
                reward = call["reward"]
                reward_str = f"[green]{reward:+.2f}[/green]" if reward >= 0 else f"[red]{reward:+.2f}[/red]"
                table.add_row(
                    str(call["step"]),
                    call["tool"],
                    call["endpoint"],
                    reward_str,
                )
            console.print(table)

    def _display_aaa_payload(self) -> None:
        """Display the AAA assessment result and payload."""
        if self.evaluator is None:
            console.print("[red]No evaluator available[/red]")
            return

        payload = build_aaa_payload(
            evaluator=self.evaluator,
            purple_agent_id="demo_dummy_agent",
            green_agent_name="netheal_green_mcp_v1",
            metadata={
                "demo_seed": self.seed,
                "max_devices": self.max_devices,
            },
        )

        console.print()
        console.print(Rule("[bold yellow]A2A: GET /tasks/{id} (Final Result)[/bold yellow]", style="yellow"))
        console.print()

        summary = self.evaluator.compute_summary()
        assessment_result = AssessmentResult(
            task_id=self.task_id,
            status=TaskStatus.COMPLETED,
            summary=summary,
            artifacts=[
                Artifact(
                    label="aaa_metrics",
                    mime_type="application/json",
                    data=payload,
                )
            ],
            metadata={"participants": ["demo_dummy_agent"]},
        )

        console.print("[green]Response (200 OK):[/green]")
        result_json = json.dumps(assessment_result.model_dump(), indent=2, default=str)
        syntax = Syntax(result_json, "json", theme="monokai", line_numbers=False)
        console.print(Panel(
            syntax,
            title="[green]AssessmentResult[/green]",
            border_style="green",
            box=box.ROUNDED,
        ))

        console.print()
        console.print(Rule("[bold yellow]AAA LEADERBOARD PAYLOAD[/bold yellow]", style="yellow"))
        console.print()

        console.print("[dim]Extracted from artifacts[0].data:[/dim]")
        payload_json = json.dumps(payload, indent=2, default=str)
        syntax = Syntax(payload_json, "json", theme="monokai", line_numbers=False)

        console.print(Panel(
            syntax,
            title="[yellow]Leaderboard Submission[/yellow]",
            border_style="yellow",
            box=box.ROUNDED,
        ))

        console.print()
        console.print("[bold]Submission Fields:[/bold]")
        console.print()

        table = Table(box=box.SIMPLE, show_header=True)
        table.add_column("Field", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Required", style="yellow")

        table.add_row("green_agent", "Environment controller identifier", "Yes")
        table.add_row("purple_agent", "Solver being evaluated", "Yes")
        table.add_row("primary_score", "Main metric for ranking", "Yes")
        table.add_row("metrics", "Detailed breakdown", "Yes")
        table.add_row("episode_count", "Number of episodes", "Yes")
        table.add_row("generated_at", "ISO timestamp", "Yes")
        table.add_row("metadata", "Additional context", "No")

        console.print(table)
        console.print()

        if self.step_by_step:
            console.print("[dim]Press Enter to finish...[/dim]")
            input()

    async def _run_dummy_agent(self, base_url: str) -> None:
        """Run the dummy agent."""
        async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
            min_s = min(self.min_steps, self.max_steps)
            max_s = max(self.min_steps, self.max_steps)
            total_steps = random.randint(min_s, max_s)
            console.print(f"[dim]Purple Agent (dummy) will take {total_steps} steps[/dim]")
            console.print()

            for step_idx in range(total_steps):
                actions = await self._get_valid_actions(client)
                if not actions:
                    console.print("[yellow]No valid actions available[/yellow]")
                    break

                final_step = step_idx == total_steps - 1
                action = self._select_action(actions, prefer_diagnosis=final_step)
                if action is None:
                    console.print("[yellow]Unable to select action[/yellow]")
                    break

                tool_name, params, result, endpoint = await self._execute_action(
                    client, action, force_diagnosis=final_step
                )

                self._display_step(
                    step_num=step_idx + 1,
                    total_steps=total_steps,
                    tool_name=tool_name,
                    params=params,
                    result=result,
                    endpoint=endpoint,
                )

                if result.get("terminated", False) or result.get("truncated", False):
                    break

    async def _get_valid_actions(self, client: httpx.AsyncClient) -> List[Dict[str, Any]]:
        response = await client.get("/actions")
        response.raise_for_status()
        payload = response.json()
        return payload.get("valid_actions", [])

    def _select_action(
        self,
        actions: List[Dict[str, Any]],
        prefer_diagnosis: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if not actions:
            return None

        diagnosis_actions = [a for a in actions if a.get("category") == "diagnosis"]
        non_diagnosis = [a for a in actions if a.get("category") != "diagnosis"]

        if prefer_diagnosis and diagnosis_actions:
            return random.choice(diagnosis_actions)

        pool = non_diagnosis or actions
        return random.choice(pool)

    async def _execute_action(
        self,
        client: httpx.AsyncClient,
        action: Dict[str, Any],
        force_diagnosis: bool = False,
    ) -> tuple[str, Dict[str, Any], Dict[str, Any], str]:
        category = action.get("category")
        action_type = action.get("action_type")
        parameters = action.get("parameters", {})

        if category == "diagnosis" or force_diagnosis:
            fault_type = action.get("action_type")
            location = parameters.get("location") or "device_0"
            params = {"fault_type": fault_type, "location": location}
            endpoint = "/tools/submit_diagnosis"
            result = await self._post(client, endpoint, params=params)
            return "submit_diagnosis", params, result, endpoint

        if category == "topology_discovery":
            if action_type == "scan_network":
                endpoint = "/tools/scan_network"
                result = await self._post(client, endpoint)
                return "scan_network", {}, result, endpoint
            if action_type == "discover_neighbors":
                params = {"device": parameters["device"]}
                endpoint = "/tools/discover_neighbors"
                result = await self._post(client, endpoint, params=params)
                return "discover_neighbors", params, result, endpoint

        if category == "diagnostic":
            if action_type == "ping":
                params = {
                    "source": parameters["source"],
                    "destination": parameters["destination"],
                }
                endpoint = "/tools/ping"
                result = await self._post(client, endpoint, params=params)
                return "ping", params, result, endpoint
            if action_type == "traceroute":
                params = {
                    "source": parameters["source"],
                    "destination": parameters["destination"],
                }
                endpoint = "/tools/traceroute"
                result = await self._post(client, endpoint, params=params)
                return "traceroute", params, result, endpoint
            if action_type == "check_status":
                params = {"device": parameters["device"]}
                endpoint = "/tools/check_status"
                result = await self._post(client, endpoint, params=params)
                return "check_status", params, result, endpoint
            if action_type == "check_interfaces":
                params = {"device": parameters["device"]}
                endpoint = "/tools/check_interfaces"
                result = await self._post(client, endpoint, params=params)
                return "check_interfaces", params, result, endpoint

        console.print("[yellow]Unknown action type, submitting diagnosis[/yellow]")
        diag_actions = [a for a in [action] if a.get("category") == "diagnosis"]
        if diag_actions:
            chosen = random.choice(diag_actions)
            fault_type = chosen.get("action_type")
            location = chosen.get("parameters", {}).get("location", "device_0")
        else:
            fault_type = "device_failure"
            location = "device_0"
        params = {"fault_type": fault_type, "location": location}
        endpoint = "/tools/submit_diagnosis"
        result = await self._post(client, endpoint, params=params)
        return "submit_diagnosis", params, result, endpoint

    async def _post(
        self,
        client: httpx.AsyncClient,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        response = await client.post(path, params=params or {})
        response.raise_for_status()
        return response.json()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NetHeal AAA demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m netheal.aaa.demo --seed 42 --devices 6
  python -m netheal.aaa.demo --seed 42 --devices 6 --step
  python -m netheal.aaa.demo --seed 42 --devices 6 --verbose
  python -m netheal.aaa.demo --seed 42 --devices 6 --step --verbose
""",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--devices", type=int, default=6, help="Max devices (default: 6)")
    parser.add_argument("--max-steps", type=int, default=15, help="Max episode steps (default: 15)")
    parser.add_argument("--min-agent-steps", type=int, default=4, help="Min dummy agent steps (default: 4)")
    parser.add_argument("--max-agent-steps", type=int, default=10, help="Max dummy agent steps (default: 10)")
    parser.add_argument("--step", action="store_true", help="Step-by-step mode")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between steps (default: 0.5s)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show protocol details")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    runner = DemoRunner(
        seed=args.seed,
        max_devices=args.devices,
        max_episode_steps=args.max_steps,
        min_steps=args.min_agent_steps,
        max_steps=args.max_agent_steps,
        step_by_step=args.step,
        delay=args.delay,
        verbose=args.verbose,
    )
    runner.run()


if __name__ == "__main__":
    main()
