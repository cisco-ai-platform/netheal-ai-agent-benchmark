# Architecture

- `netheal/network/`: Network graph (`graph.py`) and topology generation (`topology.py`).
- `netheal/faults/`: Fault injection (`injector.py`) for device/link/perf/misconfig faults.
- `netheal/tools/`: Diagnostic tool simulator (`simulator.py`).
- `netheal/environment/`: RL environment (`env.py`), actions (`actions.py`), observation (`observation.py`), rewards (`rewards.py`).
- `netheal/hints/`: Natural language hint providers.
- `webapp/backend/`: FastAPI app serving API and the static frontend.
- `webapp/frontend/`: Vanilla JS UI (index.html, app.js, style.css).
- `examples/`: Basic and interactive examples.
- `tests/`: Unit tests.
