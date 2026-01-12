"""
Purple Agent A2A Server.

Implements the AAA protocol for solver agents:
    GET  /.well-known/agent.json - Agent capability card
    POST /tasks                  - Receive episode from green agent
    GET  /tasks/{id}             - Task status

When green agent POSTs an EpisodeStart, this agent:
    1. Connects to the MCP server URL provided
    2. Uses diagnostic tools to analyze the network
    3. Submits diagnosis via the MCP server

Usage:
    python -m netheal.aaa.purple_server --host 0.0.0.0 --port 9030
    python -m netheal.aaa.purple_server --host 0.0.0.0 --port 9030 --solver gpt
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

import httpx
import typer
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

LOGGER = logging.getLogger("netheal.purple_server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(title="NetHeal Purple Agent", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_CARD_URL: Optional[str] = None
_SOLVER_TYPE: str = "dummy"


def set_config(card_url: str, solver_type: str = "dummy") -> None:
    """Configure the purple agent server."""
    global _CARD_URL, _SOLVER_TYPE
    _CARD_URL = card_url
    _SOLVER_TYPE = solver_type


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EpisodeRequest(BaseModel):
    """Request from green agent with episode info."""
    task_id: Optional[str] = None
    episode_start: Dict[str, Any]


@dataclass
class TaskRecord:
    """Tracks purple agent task execution."""
    task_id: str
    episode_start: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    runner: Optional[asyncio.Task] = None


TASKS: Dict[str, TaskRecord] = {}


@app.get("/.well-known/agent.json")
async def agent_card() -> JSONResponse:
    """Return the agent capability card."""
    card = {
        "name": "netheal-purple-agent",
        "version": "0.1.0",
        "description": "NetHeal solver agent for network troubleshooting.",
        "capabilities": {
            "accepts_tasks": True,
            "solver_type": _SOLVER_TYPE,
        },
        "contact": {"repo": "https://github.com/cisco-aispg/netheal-rl-env"},
    }
    if _CARD_URL:
        card["url"] = _CARD_URL
    return JSONResponse(card)


@app.post("/tasks")
async def create_task(payload: EpisodeRequest) -> Dict[str, Any]:
    """Receive episode from green agent and start solving."""
    task_id = payload.task_id or str(uuid.uuid4())

    if task_id in TASKS:
        raise HTTPException(status_code=409, detail="Task ID already exists.")

    record = TaskRecord(
        task_id=task_id,
        episode_start=payload.episode_start,
        status=TaskStatus.PENDING,
    )
    TASKS[task_id] = record

    record.runner = asyncio.create_task(_run_solver(record))

    return {
        "task_id": task_id,
        "status": record.status.value,
        "message": "Episode received, solving started.",
    }


@app.get("/tasks/{task_id}")
async def get_task(task_id: str) -> Dict[str, Any]:
    """Get task status and result."""
    record = TASKS.get(task_id)
    if not record:
        raise HTTPException(status_code=404, detail="Task not found.")

    return {
        "task_id": task_id,
        "status": record.status.value,
        "created_at": record.created_at.isoformat(),
        "completed_at": record.completed_at.isoformat() if record.completed_at else None,
        "result": record.result,
    }


async def _run_solver(record: TaskRecord) -> None:
    """Execute the solver."""
    record.status = TaskStatus.RUNNING
    episode = record.episode_start

    mcp_url = episode.get("mcp_server_url") or episode.get("extra", {}).get("http_helper_url")
    if not mcp_url:
        record.status = TaskStatus.FAILED
        record.result = {"error": "No MCP server URL in episode_start"}
        record.completed_at = datetime.utcnow()
        return

    LOGGER.info("Starting solver for task %s with MCP URL: %s", record.task_id, mcp_url)
    LOGGER.info("  Episode hint: %s", episode.get("hint", "N/A"))

    try:
        if _SOLVER_TYPE == "gpt":
            result = await _run_gpt_solver(mcp_url, episode)
        else:
            result = await _run_dummy_solver(mcp_url, episode)

        record.result = result
        record.status = TaskStatus.COMPLETED
    except Exception as exc:
        LOGGER.exception("Solver failed: %s", exc)
        record.result = {"error": str(exc)}
        record.status = TaskStatus.FAILED
    finally:
        record.completed_at = datetime.utcnow()


async def _run_dummy_solver(mcp_url: str, episode: Dict[str, Any]) -> Dict[str, Any]:
    """Random policy solver."""
    import random

    async with httpx.AsyncClient(base_url=mcp_url, timeout=30.0) as client:
        try:
            await client.post("/tools/scan_network")
        except httpx.HTTPError:
            pass

        response = await client.get("/actions")
        actions = response.json().get("valid_actions", [])

        for _ in range(random.randint(2, 5)):
            non_diag = [a for a in actions if a.get("category") != "diagnosis"]
            if non_diag:
                action = random.choice(non_diag)
                await _execute_action(client, action)
                response = await client.get("/actions")
                actions = response.json().get("valid_actions", [])

        diag_actions = [a for a in actions if a.get("category") == "diagnosis"]
        if diag_actions:
            action = random.choice(diag_actions)
            result = await _execute_action(client, action)
            return {
                "diagnosis": {
                    "fault_type": action.get("action_type"),
                    "location": action.get("parameters", {}).get("location"),
                },
                "result": result,
            }

        return {"error": "No diagnosis actions available"}


async def _run_gpt_solver(mcp_url: str, episode: Dict[str, Any]) -> Dict[str, Any]:
    """GPT-powered solver."""
    from netheal.aaa.gpt_agent import GPTAgent

    max_turns = episode.get("max_steps", 20)

    agent = GPTAgent(mcp_url=mcp_url, max_turns=max_turns, verbose=True)
    result = await agent.run(
        task_hint=episode.get("hint"),
        task_context=episode,
    )
    return result


async def _execute_action(client: httpx.AsyncClient, action: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single action via HTTP."""
    category = action.get("category")
    action_type = action.get("action_type")
    params = action.get("parameters", {})

    if category == "topology_discovery":
        if action_type == "scan_network":
            response = await client.post("/tools/scan_network")
        elif action_type == "discover_neighbors":
            response = await client.post("/tools/discover_neighbors", params={"device": params.get("device")})
        else:
            return {"error": f"Unknown topology action: {action_type}"}
    elif category == "diagnostic":
        if action_type == "ping":
            response = await client.post("/tools/ping", params=params)
        elif action_type == "traceroute":
            response = await client.post("/tools/traceroute", params=params)
        elif action_type == "check_status":
            response = await client.post("/tools/check_status", params={"device": params.get("device")})
        elif action_type == "check_interfaces":
            response = await client.post("/tools/check_interfaces", params={"device": params.get("device")})
        else:
            return {"error": f"Unknown diagnostic action: {action_type}"}
    elif category == "diagnosis":
        response = await client.post(
            "/tools/submit_diagnosis",
            params={"fault_type": action_type, "location": params.get("location", "device_0")},
        )
    else:
        return {"error": f"Unknown action category: {category}"}

    return response.json()


cli = typer.Typer(
    name="netheal-purple-agent",
    help="NetHeal Purple Agent A2A Server",
    add_completion=False,
)


@cli.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Host address."),
    port: int = typer.Option(9030, "--port", help="Port number."),
    card_url: Optional[str] = typer.Option(None, "--card-url", help="URL for agent card."),
    solver: str = typer.Option("dummy", "--solver", help="Solver type: dummy or gpt"),
    log_level: str = typer.Option("info", "--log-level", help="Uvicorn log level."),
) -> None:
    """Start the purple agent A2A server."""
    load_dotenv()

    effective_card_url = card_url or f"http://{host}:{port}"
    set_config(effective_card_url, solver)

    LOGGER.info("Starting NetHeal purple agent A2A server")
    LOGGER.info("  Host: %s", host)
    LOGGER.info("  Port: %d", port)
    LOGGER.info("  Card URL: %s", effective_card_url)
    LOGGER.info("  Solver: %s", solver)

    uvicorn.run(app, host=host, port=port, log_level=log_level)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
