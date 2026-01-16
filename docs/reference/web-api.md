# REST API Reference

Backend module: `webapp.backend.app.main:app`

## Endpoints

- GET `/api/health` → `{ "status": "ok" }`
- POST `/api/env/reset` → starts a new episode
- GET `/api/env/state` → current observation, info, valid actions
- GET `/api/env/actions` → valid action IDs and descriptions
- POST `/api/env/step` → apply an action ID and return updated state

## Request: POST /api/env/reset

JSON body (see `webapp/backend/app/schemas.py`):

```json
{
  "seed": 42,
  "max_devices": 8,
  "max_episode_steps": 20,
  "topology_types": ["star", "mesh", "hierarchical"],
  "enable_user_hints": true,
  "hint_provider_mode": "auto",
  "user_context": {"access_point": "Guest-WiFi"}
}
```

`hint_provider_mode` accepts: `auto` | `heuristic` | `azure` | `openai` | `anthropic` | `bedrock`.

## Request: POST /api/env/step

```json
{ "action_id": 0 }
```

Responses include JSON-serializable observations (lists) and metadata suitable for the frontend.
