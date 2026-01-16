# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Green Agent A2A Server for NetHeal assessments.

FastAPI server implementing the AAA (Agentified Agent Assessment) protocol
endpoints. Accepts assessment requests, streams progress via SSE, and
coordinates episode execution through the NetHealGreenAgent.

Endpoints:
    POST /                         - A2A JSON-RPC endpoint (streaming supported)
    GET  /.well-known/agent-card.json  - Agent capability card
    GET  /.well-known/agent.json       - Legacy agent card alias
    POST /tasks                    - Create assessment task (REST)
    GET  /tasks/{id}               - Query task status (REST)
    GET  /tasks/{id}/stream        - SSE stream for real-time updates (REST)
"""
from __future__ import annotations

import asyncio
import contextlib
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
from pydantic import BaseModel, ValidationError
from sse_starlette.sse import EventSourceResponse

from a2a.server.apps.jsonrpc import A2AFastAPIApplication
from a2a.server.context import ServerCallContext
from a2a.server.events.event_queue import Event
from a2a.server.request_handlers.request_handler import RequestHandler
from a2a.types import (
    AgentCapabilities as A2AAgentCapabilities,
    AgentCard as A2AAgentCard,
    AgentSkill as A2AAgentSkill,
    Artifact as A2AArtifact,
    DataPart as A2ADataPart,
    Message as A2AMessage,
    MessageSendParams as A2AMessageSendParams,
    Part as A2APart,
    Task as A2ATask,
    TaskArtifactUpdateEvent as A2ATaskArtifactUpdateEvent,
    TaskIdParams as A2ATaskIdParams,
    TaskPushNotificationConfig as A2ATaskPushNotificationConfig,
    TaskQueryParams as A2ATaskQueryParams,
    TaskState as A2ATaskState,
    TaskStatus as A2ATaskStatus,
    TaskStatusUpdateEvent as A2ATaskStatusUpdateEvent,
    TextPart as A2ATextPart,
    DeleteTaskPushNotificationConfigParams,
    ListTaskPushNotificationConfigParams,
    UnsupportedOperationError,
)
from a2a.utils.errors import ServerError

from netheal.aaa.green_agent import NetHealGreenAgent
from netheal.aaa.schemas import (
    AssessmentConfig,
    AssessmentRequest,
    AssessmentResult,
    Participant,
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
A2A_AGENT_CARD: Optional[A2AAgentCard] = None


def set_card_url(url: str) -> None:
    """Configure the URL advertised in the agent card."""
    global _CARD_URL, A2A_AGENT_CARD
    _CARD_URL = url
    if A2A_AGENT_CARD:
        A2A_AGENT_CARD.url = url


def _build_a2a_agent_card(url: Optional[str]) -> A2AAgentCard:
    """Construct the agent capability card per A2A specification."""
    card_url = url or "http://localhost:9020"
    return A2AAgentCard(
        name="netheal-green-agent",
        description=(
            "NetHeal RL environment orchestrator for network troubleshooting assessment. "
            "Implements the AAA (Agentified Agent Assessment) protocol as a green agent "
            "(evaluator) that creates diagnostic scenarios and scores solver agents."
        ),
        version="0.1.0",
        url=card_url,
        capabilities=A2AAgentCapabilities(
            streaming=True,
            push_notifications=False,
            state_transition_history=True,
        ),
        skills=[
            A2AAgentSkill(
                id="network-troubleshooting-assessment",
                name="Network Troubleshooting Assessment",
                description="Evaluate solver agents on network fault diagnosis tasks",
                tags=["assessment", "evaluation", "network", "troubleshooting", "rl"],
                examples=[
                    "Run a 5-episode assessment with star topology networks",
                    "Evaluate solver agent on device failure scenarios",
                ],
            )
        ],
        default_input_modes=["application/json"],
        default_output_modes=["application/json", "text/event-stream"],
    )


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


def _map_task_state(status: TaskStatus) -> A2ATaskState:
    mapping = {
        TaskStatus.PENDING: A2ATaskState.submitted,
        TaskStatus.RUNNING: A2ATaskState.working,
        TaskStatus.COMPLETED: A2ATaskState.completed,
        TaskStatus.FAILED: A2ATaskState.failed,
        TaskStatus.CANCELLED: A2ATaskState.canceled,
    }
    return mapping.get(status, A2ATaskState.unknown)


def _a2a_message(text: str, role: str = "agent") -> A2AMessage:
    return A2AMessage(
        message_id=str(uuid.uuid4()),
        role=role,
        parts=[A2APart(root=A2ATextPart(text=text))],
    )


def _a2a_task_from_record(record: TaskRecord, message: Optional[A2AMessage] = None) -> A2ATask:
    task_id = record.request.task_id or "unknown-task"
    status = A2ATaskStatus(
        state=_map_task_state(record.status),
        message=message,
        timestamp=(record.completed_at or record.created_at).isoformat(),
    )
    return A2ATask(
        id=task_id,
        context_id=task_id,
        status=status,
        metadata={"netheal_status": record.status.value},
    )


def _a2a_status_event(record: TaskRecord, message: Optional[A2AMessage], final: bool) -> A2ATaskStatusUpdateEvent:
    task_id = record.request.task_id or "unknown-task"
    status = A2ATaskStatus(
        state=_map_task_state(record.status),
        message=message,
        timestamp=datetime.utcnow().isoformat(),
    )
    return A2ATaskStatusUpdateEvent(
        task_id=task_id,
        context_id=task_id,
        status=status,
        final=final,
    )


def _a2a_artifact_event(record: TaskRecord) -> Optional[A2ATaskArtifactUpdateEvent]:
    if record.result is None:
        return None
    task_id = record.request.task_id or "unknown-task"
    artifact = A2AArtifact(
        artifact_id=str(uuid.uuid4()),
        name="assessment_results",
        description="NetHeal assessment results",
        parts=[A2APart(root=A2ADataPart(data=record.result.model_dump()))],
    )
    return A2ATaskArtifactUpdateEvent(
        task_id=task_id,
        context_id=task_id,
        artifact=artifact,
        append=False,
        last_chunk=True,
    )


def _extract_assessment_payload(params: A2AMessageSendParams) -> Dict[str, object]:
    payload: Dict[str, object] = {}

    def merge(data: object) -> None:
        if isinstance(data, dict):
            payload.update(data)

    if params.metadata:
        merge(params.metadata)
    if params.message.metadata:
        merge(params.message.metadata)

    for part in params.message.parts:
        root = part.root
        if isinstance(root, A2ADataPart):
            merge(root.data)
        elif isinstance(root, A2ATextPart):
            text = root.text.strip()
            if text.startswith("{") and text.endswith("}"):
                try:
                    merge(json.loads(text))
                except json.JSONDecodeError:
                    continue

    if "assessment_request" in payload and isinstance(payload["assessment_request"], dict):
        payload = payload["assessment_request"]  # type: ignore[assignment]

    participants = payload.get("participants")
    if isinstance(participants, (list, dict)):
        payload["participants"] = _normalize_participants(participants)

    if "task_id" not in payload:
        if params.message.task_id:
            payload["task_id"] = params.message.task_id
        elif params.message.context_id:
            payload["task_id"] = params.message.context_id

    return payload


def _build_assessment_request(params: A2AMessageSendParams) -> AssessmentRequest:
    payload = _extract_assessment_payload(params)
    if not payload:
        return AssessmentRequest()
    try:
        return AssessmentRequest.model_validate(payload)
    except ValidationError as exc:
        LOGGER.warning(
            "Failed to parse assessment payload, falling back to defaults: %s", exc
        )
        return _build_fallback_request(payload)


def _normalize_endpoint(endpoint: str) -> str:
    if "://" in endpoint:
        return endpoint
    return f"http://{endpoint}"


def _normalize_participants(participants: object) -> Dict[str, Dict[str, object]]:
    normalized: Dict[str, Dict[str, object]] = {}

    if isinstance(participants, dict):
        items = participants.items()
    else:
        items = []
        for participant in participants:
            if not isinstance(participant, dict):
                continue
            role = participant.get("role") or participant.get("name")
            items.append((role, participant))

    for role, participant in items:
        if not isinstance(participant, dict):
            continue
        resolved_role = role or participant.get("role") or participant.get("name")
        if not resolved_role:
            continue
        endpoint = participant.get("endpoint")
        if endpoint:
            endpoint = _normalize_endpoint(str(endpoint))
        else:
            endpoint = _normalize_endpoint(f"{resolved_role}:9009")
        enriched = dict(participant)
        enriched.setdefault("role", resolved_role)
        enriched.setdefault("endpoint", endpoint)
        normalized[str(resolved_role)] = enriched

    return normalized


def _build_fallback_request(payload: Dict[str, object]) -> AssessmentRequest:
    fallback_config = AssessmentConfig()
    config_payload = payload.get("config")
    if isinstance(config_payload, dict):
        with contextlib.suppress(Exception):
            fallback_config = AssessmentConfig.model_validate(config_payload)

    participants_payload = payload.get("participants")
    normalized_participants: Dict[str, Participant] = {}
    if isinstance(participants_payload, dict):
        for role, participant in participants_payload.items():
            if not isinstance(participant, dict):
                continue
            with contextlib.suppress(Exception):
                normalized_participants[str(role)] = Participant.model_validate(participant)

    return AssessmentRequest(
        task_id=payload.get("task_id") if isinstance(payload.get("task_id"), str) else None,
        participants=normalized_participants,
        config=fallback_config,
        metadata=payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {},
    )


async def _create_task_record(payload: AssessmentRequest) -> TaskRecord:
    """Create a task record and start its runner."""
    async with TASK_LOCK:
        task_id = payload.task_id or str(uuid.uuid4())
        if task_id in TASKS:
            raise HTTPException(status_code=409, detail="Task ID already exists.")
        payload.task_id = task_id
        record = TaskRecord(request=payload, status=TaskStatus.PENDING)
        TASKS[task_id] = record

    record.runner = asyncio.create_task(_run_task(record))
    return record


@app.post("/tasks")
async def create_task(payload: AssessmentRequest) -> Dict[str, str]:
    """Create and start a new assessment task."""
    record = await _create_task_record(payload)
    return {"task_id": record.request.task_id or "", "status": record.status.value}


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


async def _stream_a2a_events(record: TaskRecord, include_task: bool = True):
    """Yield A2A-compatible events for a task."""
    if include_task:
        yield _a2a_task_from_record(record)

    artifact_sent = False
    while True:
        if (
            record.status
            in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
            and record.update_queue.empty()
        ):
            if record.result and not artifact_sent:
                artifact_event = _a2a_artifact_event(record)
                if artifact_event:
                    yield artifact_event
                artifact_sent = True
            yield _a2a_status_event(record, message=None, final=True)
            break

        try:
            update = await asyncio.wait_for(record.update_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        message = _a2a_message(update.message)
        yield _a2a_status_event(record, message=message, final=False)


class NetHealA2ARequestHandler(RequestHandler):
    """A2A JSON-RPC handler backed by the NetHeal assessment engine."""

    async def on_get_task(
        self,
        params: A2ATaskQueryParams,
        context: ServerCallContext | None = None,
    ) -> Optional[A2ATask]:
        record = TASKS.get(params.id)
        if not record:
            return None
        return _a2a_task_from_record(record)

    async def on_cancel_task(
        self,
        params: A2ATaskIdParams,
        context: ServerCallContext | None = None,
    ) -> Optional[A2ATask]:
        record = TASKS.get(params.id)
        if not record:
            return None
        if record.runner and not record.runner.done():
            record.runner.cancel()
        record.status = TaskStatus.CANCELLED
        record.completed_at = datetime.utcnow()
        return _a2a_task_from_record(record)

    async def on_message_send(
        self,
        params: A2AMessageSendParams,
        context: ServerCallContext | None = None,
    ) -> A2ATask:
        record = await _create_task_record(_build_assessment_request(params))
        if record.runner:
            with contextlib.suppress(Exception):
                await record.runner
        return _a2a_task_from_record(record)

    async def on_message_send_stream(
        self,
        params: A2AMessageSendParams,
        context: ServerCallContext | None = None,
    ):
        record = await _create_task_record(_build_assessment_request(params))
        async for event in _stream_a2a_events(record, include_task=True):
            yield event

    async def on_set_task_push_notification_config(
        self,
        params: A2ATaskPushNotificationConfig,
        context: ServerCallContext | None = None,
    ) -> A2ATaskPushNotificationConfig:
        raise ServerError(error=UnsupportedOperationError())

    async def on_get_task_push_notification_config(
        self,
        params: A2ATaskIdParams,
        context: ServerCallContext | None = None,
    ) -> A2ATaskPushNotificationConfig:
        raise ServerError(error=UnsupportedOperationError())

    async def on_resubscribe_to_task(
        self,
        params: A2ATaskIdParams,
        context: ServerCallContext | None = None,
    ):
        record = TASKS.get(params.id)
        if not record:
            raise ServerError(error=UnsupportedOperationError())
        async for event in _stream_a2a_events(record, include_task=True):
            yield event

    async def on_list_task_push_notification_config(
        self,
        params: ListTaskPushNotificationConfigParams,
        context: ServerCallContext | None = None,
    ):
        raise ServerError(error=UnsupportedOperationError())

    async def on_delete_task_push_notification_config(
        self,
        params: DeleteTaskPushNotificationConfigParams,
        context: ServerCallContext | None = None,
    ) -> None:
        raise ServerError(error=UnsupportedOperationError())


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


A2A_AGENT_CARD = _build_a2a_agent_card(_CARD_URL)
A2A_HANDLER = NetHealA2ARequestHandler()
A2A_APP = A2AFastAPIApplication(agent_card=A2A_AGENT_CARD, http_handler=A2A_HANDLER)
A2A_APP.add_routes_to_app(app)


__all__ = ["app", "set_card_url"]
