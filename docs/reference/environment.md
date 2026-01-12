# Environment API (Gymnasium)

Entry point: `netheal.environment.env.NetworkTroubleshootingEnv` (imported as `from netheal import NetworkTroubleshootingEnv`).

## Reset/Step

```python
obs, info = env.reset(seed=None)
obs, reward, terminated, truncated, info = env.step(action)
```

- `info` includes `reward_breakdown`, `user_hint` (if enabled), and other episode stats.
- Use `env.get_valid_actions()` and `env.get_action_meanings()` to interpret the current dynamic action set.

## Observation

A dictionary with 4 components:

- discovery_matrix: shape (max_devices, max_devices), dtype int8 in [-1,2]
- device_status: shape (max_devices, 10), float32
- recent_diagnostics: shape (10, 6), float32 (last N tool results)
- episode_metadata: shape (4,), float32

## Actions

Hierarchical, episode-specific. Categories:

- Topology discovery: `scan_network`, `discover_neighbors(<device>)`
- Diagnostics: `ping(<src>,<dst>)`, `traceroute(<src>,<dst>)`, `check_status(<device>)`, `check_interfaces(<device>)`
- Final diagnosis (terminating): device/link/performance/misconfiguration

Retrieve current valid action IDs: `env.get_valid_actions()`; meanings via `env.get_action_meanings()`.

## Rewards

- Step penalty: small negative per step.
- Final diagnosis: scaled positive (correct) or negative (incorrect), proportional to network size.

`info['reward_breakdown']` shows sources of reward at each step.

## Evaluation & Metrics

For benchmark/competition reporting wrap the environment with `netheal.evaluation.MetricsCollectorWrapper`. The wrapper captures every action (category, parameters, tool cost, reward breakdown) and exposes both per-episode metrics and aggregate summaries (Diagnosis Success Rate, fault-type macro F1, time-to-diagnosis, tool cost index, topology coverage, evidence sufficiency, action diversity, redundancy, discovery efficiency, composite episode score). Example:

```python
from netheal import NetworkTroubleshootingEnv
from netheal.evaluation import MetricsCollectorWrapper, build_aaa_payload

env = MetricsCollectorWrapper(NetworkTroubleshootingEnv())
obs, info = env.reset(seed=0)
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

summary = env.evaluator.compute_summary()
payload = build_aaa_payload(env.evaluator, purple_agent_id="solver_v1")
```

See `tests/test_evaluation_metrics.py` for regression coverage of the metrics calculator and wrapper.
