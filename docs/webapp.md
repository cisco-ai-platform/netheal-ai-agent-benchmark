# Web App Guide

An interactive demo to run episodes, view hints, and step through actions.

## Run

```bash
source venv/bin/activate
uvicorn webapp.backend.app.main:app --reload
```
Open http://127.0.0.1:8000/

## Walkthrough

1. Reset: set seed, max devices/steps, hint mode; click Reset.
2. Observe: discovery matrix, device table, recent diagnostics, hint.
3. Act: choose category (discovery/diagnostics/diagnosis), then parameters, Step.
4. Diagnose: take a final diagnosis action to end the episode.

## API (served by backend)

- GET /api/health
- POST /api/env/reset
- GET /api/env/state
- GET /api/env/actions
- POST /api/env/step

See details: ./reference/web-api.md

## Common issues

- Port 8000 in use → run on another port: `uvicorn webapp.backend.app.main:app --port 8001` and open http://127.0.0.1:8001/
- 404 at root → ensure `webapp/frontend/index.html` exists (it should in this repo).
- Single environment instance: the demo manages one env at a time (singleton `EnvManager`).
