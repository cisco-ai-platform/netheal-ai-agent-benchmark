# Troubleshooting

- Port already in use: run uvicorn on another port, e.g. `uvicorn webapp.backend.app.main:app --port 8001`.
- CORS/404 issues: ensure you open http://127.0.0.1:8000/ and that `webapp/frontend/index.html` exists.
- Observation shape errors: remember observations are dicts; use dict-aware policies.
- Action errors: always call `env.get_valid_actions()` before stepping to avoid invalid IDs.
- Azure OpenAI hints: if not configured, heuristic hints are used automatically.
