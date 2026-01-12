"""
Green Agent A2A Server for NetHeal assessments.

FastAPI server implementing the AAA (Agentified Agent Assessment) protocol
endpoints. Accepts assessment requests, streams progress via SSE, and
coordinates episode execution through the NetHealGreenAgent.

Endpoints:
    GET  /.well-known/agent.json  - Agent capability card
    POST /tasks                   - Create assessment task
    GET  /tasks/{id}              - Query task status
    GET  /tasks/{id}/stream       - SSE stream for real-time updates
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

LOGGER = logging.getLogger(__name__)
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/app/output"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from netheal.aaa.green_agent import NetHealGreenAgent
from netheal.aaa.schemas import (
    AssessmentRequest,
    AssessmentResult,
    TaskStatus,
    TaskUpdate,
    TaskUpdateLevel,
)

app = FastAPI(title="NetHeal Green Agent", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_CARD_URL: Optional[str] = None


def set_card_url(url: str) -> None:
    """Configure the URL advertised in the agent card."""
    global _CARD_URL
    _CARD_URL = url


def _build_agent_card() -> Dict[str, object]:
    """Construct the agent capability card."""
    card = {
        "name": "netheal-green-agent",
        "version": "0.1.0",
        "description": "NetHeal RL environment orchestrator with MCP tool support.",
        "capabilities": {
            "accepts_tasks": True,
            "streams_updates": True,
            "supports_mcp": True,
        },
        "contact": {"repo": "https://github.com/cisco-aispg/netheal-rl-env"},
    }
    if _CARD_URL:
        card["url"] = _CARD_URL
    return card


@dataclass
class TaskRecord:
    """Tracks state for a running or completed assessment task."""

    request: AssessmentRequest
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[AssessmentResult] = None
    created_at: datetime = field(default_factory=lambda: datetime.utcnow())
    completed_at: Optional[datetime] = None
    updates: list[TaskUpdate] = field(default_factory=list)
    update_queue: asyncio.Queue[TaskUpdate] = field(default_factory=asyncio.Queue)
    runner: Optional[asyncio.Task] = None


TASKS: Dict[str, TaskRecord] = {}
TASK_LOCK = asyncio.Lock()


@app.get("/.well-known/agent.json")
async def agent_card() -> JSONResponse:
    """Return the agent capability card."""
    return JSONResponse(_build_agent_card())


@app.post("/tasks")
async def create_task(payload: AssessmentRequest) -> Dict[str, str]:
    """Create and start a new assessment task."""
    async with TASK_LOCK:
        task_id = payload.task_id or str(uuid.uuid4())
        if task_id in TASKS:
            raise HTTPException(status_code=409, detail="Task ID already exists.")
        payload.task_id = task_id
        record = TaskRecord(request=payload, status=TaskStatus.PENDING)
        TASKS[task_id] = record

    record.runner = asyncio.create_task(_run_task(record))
    return {"task_id": task_id, "status": record.status.value}


@app.get("/tasks/{task_id}")
async def get_task(task_id: str) -> Dict[str, object]:
    """Retrieve task status and results."""
    record = _require_task(task_id)
    return {
        "task_id": task_id,
        "status": record.status.value,
        "created_at": record.created_at.isoformat(),
        "completed_at": record.completed_at.isoformat() if record.completed_at else None,
        "result": record.result.model_dump() if record.result else None,
        "updates": [update.model_dump() for update in record.updates],
    }


@app.get("/tasks/{task_id}/stream")
async def stream_updates(task_id: str) -> EventSourceResponse:
    """Stream task updates via Server-Sent Events."""
    record = _require_task(task_id)

    async def event_generator():
        for update in record.updates:
            yield _sse_payload("update", update)

        while True:
            if (
                record.status
                in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
                and record.update_queue.empty()
            ):
                if record.result:
                    yield _sse_payload("result", record.result)
                break
            try:
                update = await asyncio.wait_for(record.update_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            yield _sse_payload("update", update)

    return EventSourceResponse(event_generator())


def _require_task(task_id: str) -> TaskRecord:
    """Fetch task record or raise 404."""
    record = TASKS.get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return record


def _sse_payload(event: str, model) -> dict:
    """Format model as SSE payload."""
    if hasattr(model, "model_dump"):
        data = model.model_dump()
    else:
        data = model
    return {
        "event": event,
        "data": json.dumps(data, default=str),
    }


async def _run_task(record: TaskRecord) -> None:
    """Execute the assessment and capture results."""
    record.status = TaskStatus.RUNNING
    agent = NetHealGreenAgent(record.request)

    async def update_sink(update: TaskUpdate) -> None:
        record.updates.append(update)
        await record.update_queue.put(update)

    try:
        result = await agent.run(update_callback=update_sink)
        record.result = result
        record.status = result.status
        record.completed_at = datetime.utcnow()

        # Write results to output directory for AgentBeats collection
        _save_results(record)
    except Exception as exc:
        error_update = TaskUpdate(
            task_id=record.request.task_id or "unknown-task",
            message="Task execution failed.",
            level=TaskUpdateLevel.ERROR,
            payload={"error": str(exc)},
        )
        record.updates.append(error_update)
        await record.update_queue.put(error_update)
        record.status = TaskStatus.FAILED
        record.completed_at = datetime.utcnow()
        raise


def _save_results(record: TaskRecord) -> None:
    """Save assessment results to output directory for AgentBeats collection."""
    if record.result is None:
        return

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Save full assessment results
        results_path = OUTPUT_DIR / "assessment_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(record.result.model_dump(), f, indent=2, default=str)
        LOGGER.info("Saved assessment results to %s", results_path)

        # Extract and save metrics artifact for leaderboard
        for artifact in record.result.artifacts:
            if artifact.label == "aaa_metrics":
                metrics_path = OUTPUT_DIR / "metrics.json"
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(artifact.data, f, indent=2, default=str)
                LOGGER.info("Saved metrics artifact to %s", metrics_path)
                break
    except Exception as exc:
        LOGGER.warning("Failed to save results to output directory: %s", exc)


__all__ = ["app", "set_card_url"]
