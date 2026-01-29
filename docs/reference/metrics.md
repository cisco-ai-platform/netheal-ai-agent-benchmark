# Metrics Reference

This document describes all metrics collected by the NetHeal benchmark for evaluating agent performance.

## Overview

The benchmark wraps the environment with `MetricsCollectorWrapper` to capture detailed episode-level metrics. These are aggregated by `CompetitionEvaluator` into summary statistics for leaderboard reporting.

## Episode-Level Metrics

### Primary Outcome

| Metric | Type | Description |
|--------|------|-------------|
| `diagnosis_success` | bool | Whether the agent correctly identified both fault type and location |

### Episode Context

| Metric | Type | Description |
|--------|------|-------------|
| `network_size` | int | Number of devices in the network |
| `steps` | int | Total actions taken in the episode |
| `normalized_steps` | float | `steps / max_episode_steps` (0-1 scale) |

### Reward

| Metric | Type | Description |
|--------|------|-------------|
| `total_reward` | float | Cumulative reward from the environment |

### Tool Usage

| Metric | Type | Description |
|--------|------|-------------|
| `tool_cost` | float | Sum of tool costs (each tool has an associated cost) |
| `tool_cost_normalized` | float | Normalized tool cost (0-1 scale) |
| `tool_error_count` | int | Number of invalid/failed tool calls |
| `tool_error_rate` | float | `tool_error_count / (tool_error_count + steps)` |

### Exploration

| Metric | Type | Description |
|--------|------|-------------|
| `topology_coverage` | float | Average of node and edge coverage (0-1 scale) |
| `node_coverage` | float | `discovered_nodes / network_size` |
| `edge_coverage` | float | `discovered_edges / total_edges` |

### Investigation Quality

| Metric | Type | Description |
|--------|------|-------------|
| `evidence_sufficiency` | float | Fraction of diagnostic actions relevant to the fault |
| `redundancy_count` | int | Number of repeated identical diagnostic calls |
| `redundancy_rate` | float | `redundancy_count / total_diagnostic_actions` |

### Partial Credit

| Metric | Type | Description |
|--------|------|-------------|
| `location_correct` | bool | Whether the predicted location matched ground truth |

### Efficiency

| Metric | Type | Description |
|--------|------|-------------|
| `steps_per_device` | float | `steps / network_size` - complexity-adjusted step count |
| `cost_efficiency` | float | `success / (1 + tool_cost_normalized)` - efficiency considering tool usage |

### Timing

| Metric | Type | Description |
|--------|------|-------------|
| `wall_time_seconds` | float | Real-world time taken for the episode |

### Ground Truth & Predictions

| Metric | Type | Description |
|--------|------|-------------|
| `ground_truth_type` | str | Actual fault type (device_failure, link_failure, etc.) |
| `ground_truth_location` | str | Actual fault location |
| `predicted_type` | str | Agent's predicted fault type |
| `predicted_location` | str | Agent's predicted fault location |

### Composite Score

| Metric | Type | Description |
|--------|------|-------------|
| `composite_episode_score` | float | Environment's total reward (includes success bonus and step penalties) |

## Aggregate Metrics (Summary)

These metrics are computed across all episodes by `CompetitionEvaluator.compute_summary()`.

### Primary Metrics

| Metric | Description |
|--------|-------------|
| `episodes` | Total number of episodes evaluated |
| `diagnosis_success_rate` | Weighted average of successful diagnoses |
| `fault_type_macro_f1` | Macro F1 score across fault types |

### Accuracy Metrics

| Metric | Description |
|--------|-------------|
| `location_accuracy` | Weighted average of correct location predictions |

### Step Metrics

| Metric | Description |
|--------|-------------|
| `avg_steps` | Weighted average of steps taken |
| `avg_steps_per_device` | Weighted average of complexity-adjusted steps |
| `normalized_steps` | Weighted average of normalized step count |

### Efficiency Metrics

| Metric | Description |
|--------|-------------|
| `cost_efficiency` | Weighted average of cost efficiency |
| `tool_cost_index` | Weighted average of normalized tool cost |
| `tool_error_rate` | Weighted average of tool error rate |

### Exploration Metrics

| Metric | Description |
|--------|-------------|
| `topology_coverage` | Weighted average of topology coverage |
| `evidence_sufficiency` | Weighted average of evidence sufficiency |
| `redundancy_rate` | Weighted average of redundancy rate |

### Other

| Metric | Description |
|--------|-------------|
| `avg_total_reward` | Weighted average of total reward |
| `composite_episode_score` | Weighted average of composite scores |
| `avg_wall_time_seconds` | Average wall clock time per episode |
| `confusion_matrix` | Fault type confusion matrix |
| `per_fault_type` | Per-fault-type breakdown of success rate, avg steps, and location accuracy |

## Weighting

Aggregate metrics use network-size weighting: larger networks contribute more to the averages. This prevents small networks from dominating the statistics.

```python
weights = [max(1, m.network_size) for m in episodes]
```

## Key Metrics for Comparison

When comparing agents, focus on:

1. **`diagnosis_success_rate`** - Primary measure of correctness
2. **`fault_type_macro_f1`** - Balanced performance across fault types
3. **`avg_steps_per_device`** - Complexity-adjusted efficiency (lower is better)
4. **`cost_efficiency`** - Success with minimal tool usage
5. **`location_accuracy`** - Partial credit for correct location

## Usage Example

```python
from netheal import NetworkTroubleshootingEnv
from netheal.evaluation import MetricsCollectorWrapper

env = MetricsCollectorWrapper(NetworkTroubleshootingEnv())
obs, info = env.reset(seed=0)

# Run episode...
done = False
while not done:
    action = your_agent.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

# Get episode metrics
episode_metrics = env.last_metrics

# Get aggregate summary after multiple episodes
summary = env.evaluator.compute_summary()
```
