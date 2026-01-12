# Training with Stable Baselines3 (SB3)

Minimal PPO example for dict observations.

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from netheal import NetworkTroubleshootingEnv

env = NetworkTroubleshootingEnv(max_devices=6, max_episode_steps=15)
check_env(env)

model = PPO(
    "MultiInputPolicy",  # required for dict observations
    env,
    verbose=1,
)

model.learn(total_timesteps=100_000)
```

Evaluation snippet:

```python
obs, info = env.reset()
for _ in range(20):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

Tips:
- Use `MultiInputPolicy` and feed the dict observation directly.
- Start small (max_devices=5–6), then scale.
- Always query `env.get_valid_actions()`.
