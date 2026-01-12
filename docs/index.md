# CNTE Documentation

CNTE (Cisco Network Troubleshooting Environment) is a Gymnasium-compatible RL environment where agents discover network topology, run diagnostics, and make a final diagnosis to "heal" the network.

## Try CNTE in 60 seconds (Python)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

```python
from netheal import NetworkTroubleshootingEnv

env = NetworkTroubleshootingEnv(max_devices=6, max_episode_steps=15)
obs, info = env.reset(seed=42)

for step in range(10):
    valid = env.get_valid_actions()
    action = valid[0] if valid else env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

## Try the Web App in 60 seconds

```bash
source venv/bin/activate  # ensure your venv is active
uvicorn webapp.backend.app.main:app --reload
```
Open http://127.0.0.1:8000/ and use the UI to reset an episode, choose actions, and diagnose.

## Quick links

- Getting Started: ./getting-started.md
- Core Concepts: ./concepts.md
- Environment API: ./reference/environment.md
- Web App Guide: ./webapp.md
- Training with Stable Baselines3: ./guides/training-sb3.md
- REST API Reference: ./reference/web-api.md
- Examples: ./examples.md
- FAQ: ./faq.md
