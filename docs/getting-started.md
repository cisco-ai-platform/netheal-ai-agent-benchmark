# Getting Started

This guide gets you running quickly with NetHeal for Python experiments.

## Install

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Minimal loop

```python
from netheal import NetworkTroubleshootingEnv

env = NetworkTroubleshootingEnv(max_devices=6, max_episode_steps=15)
obs, info = env.reset(seed=0)

for step in range(20):
    valid = env.get_valid_actions()
    action = valid[0] if valid else env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

## Next steps

- Understand observations/actions: ./reference/environment.md
- Train an agent: ./guides/training-sb3.md
- Use the Web App: ./webapp.md
