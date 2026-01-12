# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""FastAPI application for the NetHeal web demo.

Endpoints:
- POST /api/env/reset: start a new episode
- GET  /api/env/state: current observation + info + valid actions
- GET  /api/env/actions: valid action IDs and descriptions
- POST /api/env/step: apply an action and return updated state

Also serves the static frontend from webapp/frontend/ at '/'.
"""
from __future__ import annotations

from typing import Any, Dict
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .manager import EnvManager
from .schemas import ResetRequest, StepRequest, ImportScenarioRequest


app = FastAPI(title="NetHeal Web Demo", version="0.1.0")

# CORS (allow localhost and file-based dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static frontend
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.isfile(index_path):
        raise HTTPException(status_code=404, detail="Frontend not built")
    return FileResponse(index_path)


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/api/env/reset")
async def api_reset(payload: ResetRequest) -> JSONResponse:
    mgr = EnvManager()
    state = mgr.reset(
        seed=payload.seed,
        max_devices=payload.max_devices,
        max_episode_steps=payload.max_episode_steps,
        topology_types=payload.topology_types,
        enable_user_hints=payload.enable_user_hints,
        hint_provider_mode=payload.hint_provider_mode,
        user_context=payload.user_context,
    )
    return JSONResponse(state)


@app.get("/api/env/state")
async def api_state() -> JSONResponse:
    mgr = EnvManager()
    try:
        state = mgr.get_state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(state)


@app.get("/api/env/actions")
async def api_actions() -> JSONResponse:
    mgr = EnvManager()
    try:
        data = mgr.get_actions()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(data)


@app.post("/api/env/step")
async def api_step(payload: StepRequest) -> JSONResponse:
    mgr = EnvManager()
    try:
        state = mgr.step(payload.action_id)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(state)


@app.get("/api/env/export")
async def api_export() -> JSONResponse:
    """Export the current scenario state."""
    mgr = EnvManager()
    try:
        scenario = mgr.export_scenario()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(scenario)


@app.post("/api/env/import")
async def api_import(payload: ImportScenarioRequest) -> JSONResponse:
    """Import a previously exported scenario."""
    mgr = EnvManager()
    try:
        state = mgr.import_scenario(payload.scenario_data)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (ValueError, KeyError, AttributeError) as e:
        # Likely a scenario format issue
        import traceback
        print("Import error (scenario format):")
        traceback.print_exc()
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid scenario format: {str(e)}. The scenario file may be corrupted or from an incompatible version."
        )
    except Exception as e:
        # Log the full error for debugging
        import traceback
        print("Unexpected import error:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")
    return JSONResponse(state)


# Convenience: run with `uvicorn webapp.backend.app.main:app --reload`
