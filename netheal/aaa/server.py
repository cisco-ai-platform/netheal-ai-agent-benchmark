# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

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
from typing import Dict, List, Optional

LOGGER = logging.getLogger(__name__)
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/app/output"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
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
    """Construct the agent capability card per A2A specification."""
    card = {
        # Required A2A fields
        "name": "netheal-green-agent",
        "description": "NetHeal RL environment orchestrator for network troubleshooting assessment. "
                       "Implements the AAA (Agentified Agent Assessment) protocol as a green agent "
                       "(evaluator) that creates diagnostic scenarios and scores solver agents.",
        "version": "0.1.0",
        # A2A capabilities
        "capabilities": {
            "streaming": True,  # Supports SSE streaming
            "pushNotifications": False,
            "stateTransitionHistory": True,
        },
        # A2A skills (what tasks this agent can handle)
        "skills": [
            {
                "id": "network-troubleshooting-assessment",
                "name": "Network Troubleshooting Assessment",
                "description": "Evaluate solver agents on network fault diagnosis tasks",
                "tags": ["assessment", "evaluation", "network", "troubleshooting", "rl"],
                "examples": [
                    "Run a 5-episode assessment with star topology networks",
                    "Evaluate solver agent on device failure scenarios",
                ],
            }
        ],
        # Default modes
        "defaultInputModes": ["application/json"],
        "defaultOutputModes": ["application/json", "text/event-stream"],
        # Protocol support
        "protocols": {
            "a2a": "1.0",
            "mcp": "1.0",
        },
        # Contact info
        "provider": {
            "organization": "Cisco AI SPG",
            "url": "https://github.com/cisco-aispg/netheal-rl-env",
        },
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


class SolverEvent(BaseModel):
    """Event from purple agent solver."""
    type: str
    turn: Optional[int] = None
    content: Optional[str] = None
    reasoning: Optional[str] = None
    tools: Optional[List[Dict]] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict] = None
    result: Optional[Dict] = None
    success: Optional[bool] = None
    has_tool_calls: Optional[bool] = None
    completed_by: Optional[str] = None
    total_turns: Optional[int] = None
    max_turns: Optional[int] = None


@app.post("/tasks/{task_id}/solver_event")
async def receive_solver_event(task_id: str, event: SolverEvent) -> Dict[str, str]:
    """Receive solver event from purple agent and emit via SSE."""
    # Find the task - solver events may use the episode task ID (with _ep suffix)
    record = None
    base_task_id = task_id.rsplit("_ep", 1)[0] if "_ep" in task_id else task_id
    
    # Try exact match first, then base task ID
    for tid in [task_id, base_task_id]:
        record = TASKS.get(tid)
        if record:
            break
    
    if not record:
        # Don't error - the task might have completed
        return {"status": "ignored", "reason": "task_not_found"}
    
    # Convert solver event to task update
    event_data = event.model_dump(exclude_none=True)
    event_type = event.type
    
    # Format message based on event type
    if event_type == "turn_start":
        message = f"Purple agent: Turn {event.turn}/{event.max_turns}"
    elif event_type == "llm_response":
        content_preview = (event.content or "")[:100]
        message = f"Purple agent reasoning: {content_preview}..."
    elif event_type == "tool_calls":
        tool_names = [t["name"] for t in (event.tools or [])]
        message = f"Purple agent calling: {', '.join(tool_names)}"
    elif event_type == "tool_result":
        status = "✓" if event.success else "✗"
        message = f"Purple agent: {event.tool_name} {status}"
    elif event_type == "task_complete":
        message = f"Purple agent completed via {event.completed_by}"
    elif event_type == "assistant_message":
        content_preview = (event.content or "")[:100]
        message = f"Purple agent: {content_preview}"
    else:
        message = f"Purple agent event: {event_type}"
    
    update = TaskUpdate(
        task_id=record.request.task_id or base_task_id,
        message=message,
        level=TaskUpdateLevel.INFO,
        payload={"solver_event": event_data},
    )
    
    record.updates.append(update)
    await record.update_queue.put(update)
    
    return {"status": "received"}


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
