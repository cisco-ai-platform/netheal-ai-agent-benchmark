# CNTE Gymnasium Interface Guide for RL Training

> Note: This material is consolidated in the canonical docs:
> - `docs/reference/environment.md` (Environment API)
> - `docs/guides/training-sb3.md` (Training with SB3)

This guide provides comprehensive documentation for using CNTE's Gymnasium-compatible interface to train reinforcement learning agents for network troubleshooting. CNTE implements the standard OpenAI Gymnasium API, making it compatible with popular RL libraries like Stable Baselines3, Ray RLlib, and custom training loops.

## Table of Contents

1. [Environment Overview](#environment-overview)
2. [Installation & Setup](#installation--setup)
3. [Basic Usage](#basic-usage)
4. [Observation Space](#observation-space)
5. [Action Space](#action-space)
6. [Reward System](#reward-system)
7. [Training Integration](#training-integration)
8. [Advanced Configuration](#advanced-configuration)
9. [Evaluation & Debugging](#evaluation--debugging)
10. [Best Practices](#best-practices)

## Environment Overview

CNTE's `NetworkTroubleshootingEnv` is a Gymnasium-compatible environment that simulates realistic network troubleshooting scenarios. Agents learn to:

- **Discover network topology** through systematic exploration.
- **Diagnose network faults** using realistic diagnostic tools.
- **Make an accurate diagnosis** to resolve the network issue.

### Key Features

- **Graph-aware observations**: Structured representation of network topology and diagnostic history
- **Hierarchical action space**: Actions are organized into 3 categories (topology discovery, diagnostics, diagnosis).
- **Sparse rewards**: An outcome-focused reward system that encourages efficiency and accuracy.
- **Realistic fault scenarios**: Device failures, link failures, performance degradation, misconfigurations
- **Episode management**: Configurable episode length and termination conditions

## Installation & Setup

### Prerequisites

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Note: The action space is dynamically rebuilt at the start of each episode using the
actual device IDs of the generated network. As you discover devices (e.g., via
`scan_network`), diagnostic and diagnosis actions become valid for those devices/links.

### Required Dependencies

```
networkx>=3.1          # Network graph representation
gymnasium>=0.29.0      # RL environment interface
numpy>=1.24.0         # Numerical computations
matplotlib>=3.7.0     # Visualization (optional)
pytest>=7.4.0         # Testing framework
```

### Import the Environment

```python
from netheal import NetworkTroubleshootingEnv
import gymnasium as gym
import numpy as np
```

## Basic Usage

### Creating the Environment

```python
# Basic environment creation
env = NetworkTroubleshootingEnv()

# Custom configuration
env = NetworkTroubleshootingEnv(
    max_devices=8,                    # Maximum network size
    max_episode_steps=20,             # Episode length limit
    topology_types=["star", "mesh"],  # Network topology types
    render_mode="text"                # Rendering mode
)
```

### Standard Gymnasium Interface

```python
# Reset environment for new episode
observation, info = env.reset(seed=42)

# Take actions
for step in range(20):
    # Get valid actions (recommended)
    valid_actions = env.get_valid_actions()
    
    # Choose action (replace with your agent's policy)
    action = valid_actions[0] if valid_actions else env.action_space.sample()
    
    # Execute step
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Check episode completion
    if terminated or truncated:
        print(f"Episode ended at step {step}")
        break

env.close()
```

## Observation Space

NetHeal provides structured observations as a dictionary with 4 components:

### Observation Components

```python
observation_space = spaces.Dict({
    'discovery_matrix': spaces.Box(low=-1, high=2, shape=(max_devices, max_devices), dtype=np.int8),
    'device_status': spaces.Box(low=0, high=1, shape=(max_devices, 10), dtype=np.float32),
    'recent_diagnostics': spaces.Box(low=0, high=1, shape=(10, 6), dtype=np.float32),
    'episode_metadata': spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
})
```

### 1. Discovery Matrix (`discovery_matrix`)
- **Shape**: `(max_devices, max_devices)`
- **Purpose**: Network topology adjacency matrix
- **Values**:
  - `-1`: Unknown connection
  - `0`: No connection
  - `1`: Active connection
  - `2`: Failed/problematic connection

### 2. Device Status (`device_status`)
- **Shape**: `(max_devices, 10)`
- **Purpose**: Device properties and operational status
- **Features per device**:
  - Operational status (0.0-1.0)
  - Device type encoding
  - Interface count
  - Response time metrics
  - Error indicators

### 3. Recent Diagnostics (`recent_diagnostics`)
- **Shape**: `(10, 6)`
- **Purpose**: Memory of recent diagnostic tool results
- **Features per result**:
  - Tool type (ping, traceroute, status check)
  - Success/failure indicator
  - Latency measurements
  - Timestamp information
  - Source/destination encoding


### 4. Episode Metadata (`episode_metadata`)
- **Shape**: `(4,)`
- **Purpose**: Episode progress and state information
- **Components**:
  - Current step / max steps
  - Discovery progress (0.0-1.0)
  - Number of discovered devices
  - Number of diagnostic actions taken

### Accessing Observations

```python
obs, info = env.reset()

# Access individual components
topology = obs['discovery_matrix']
devices = obs['device_status']
diagnostics = obs['recent_diagnostics']
metadata = obs['episode_metadata']

print(f"Network topology shape: {topology.shape}")
print(f"Episode progress: {metadata[0]:.2f}")
```

## Action Space

NetHeal provides a discrete action space organized into 3 hierarchical categories:

### Action Categories

1. **Topology Discovery**
2. **Diagnostic Actions**
3. **Final Diagnosis**

### 1. Topology Discovery Actions

```python
# Available topology actions
actions = [
    "scan_network",                    # Broad network discovery
    "discover_neighbors(<device_id>)", # Find connections from a specific discovered device
]
```

### 2. Diagnostic Actions

```python
# Diagnostic tools (for each device pair/device)
diagnostic_actions = [
    "ping(<src>, <dst>)",        # Test connectivity
    "traceroute(<src>, <dst>)",  # Path discovery
    "check_status(<device>)",    # Device health check
    "check_interfaces(<device>)"  # Interface inspection
]
```


### 3. Final Diagnosis Actions

```python
# Final diagnosis (episode terminating)
diagnosis_actions = [
    "device_failure(device_0)",
    "link_failure(device_0->device_1)",
    "performance_degradation(device_1)",
    "misconfiguration(device_0)"
]
```

### Getting Valid Actions

```python
# Get currently valid actions based on discovered topology
valid_actions = env.get_valid_actions()
print(f"Valid actions: {len(valid_actions)}")

# Get action descriptions
action_meanings = env.get_action_meanings()
for action_id in valid_actions[:5]:  # Show first 5
    print(f"Action {action_id}: {action_meanings[action_id]}")
```

## Reward System

NetHeal uses a sparse, dynamic reward system to encourage efficient, outcome-focused troubleshooting. The reward is scaled based on the complexity of the network.

- **Step Penalty**: A small, constant penalty (`-0.1`) is applied for every action to encourage efficiency.
- **Dynamic Final Diagnosis**: The reward for the final diagnosis is scaled based on the number of devices in the network. A correct diagnosis yields a positive reward, while an incorrect one yields a penalty. This ensures that solving more complex problems is appropriately incentivized. The formula is `base_reward * (1 + network_size / scaling_factor)`.

### Reward Interpretation

The `info` dictionary returned by `env.step()` contains a `reward_breakdown` that shows the source of the reward:

```python
obs, reward, terminated, truncated, info = env.step(action)

# Access detailed reward breakdown
reward_breakdown = info['reward_breakdown']
print(f"Total reward: {reward:.2f}")
print(f"Step Penalty: {reward_breakdown.get('step_penalty', 0):.2f}")
print(f"Diagnosis Reward: {reward_breakdown.get('diagnosis_reward', 0):.2f}")
```

### Success Criteria

- **Positive final reward** (>0): Indicates a correct diagnosis.
- **Negative final reward** (<0): Indicates an incorrect diagnosis or that the episode was truncated.

## Training Integration

### Stable Baselines3 Integration

```python
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback

# Verify environment compatibility
env = NetworkTroubleshootingEnv(max_devices=6, max_episode_steps=15)
check_env(env)

# Train with PPO (recommended for dict observations)
model = PPO(
    "MultiInputPolicy",  # Required for dict observation spaces
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    tensorboard_log="./netheal_tensorboard/"
)

# Train the model
model.learn(total_timesteps=100000)

# Save trained model
model.save("netheal_ppo_agent")
```

### Custom Training Loop

```python
import torch
import torch.nn as nn
from collections import deque

class NetHealAgent:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        # Initialize your neural network here
        
    def predict(self, observation):
        # Implement your policy here
        # Handle dict observation space
        discovery = observation['discovery_matrix']
        devices = observation['device_status']
        # ... process observations and return action
        return action

# Training loop
env = NetworkTroubleshootingEnv()
agent = NetHealAgent(env.observation_space, env.action_space)

for episode in range(1000):
    obs, info = env.reset()
    total_reward = 0
    
    for step in range(env.max_episode_steps):
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Update agent here
        
        if terminated or truncated:
            break
    
    print(f"Episode {episode}: Reward {total_reward:.2f}")
```

### Ray RLlib Integration

```python
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

# Initialize Ray
ray.init()

# Configure PPO for NetHeal
config = (
    PPOConfig()
    .environment(
        env="netheal_env",  # Register environment first
        env_config={
            "max_devices": 6,
            "max_episode_steps": 15,
            "topology_types": ["star", "mesh", "hierarchical"]
        }
    )
    .framework("torch")
    .training(
        lr=3e-4,
        train_batch_size=4000,
        sgd_minibatch_size=128,
        num_sgd_iter=10,
    )
    .rollouts(num_rollout_workers=4)
)

# Train the agent
tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=ray.air.RunConfig(stop={"training_iteration": 100})
)
results = tuner.fit()
```

## Advanced Configuration

### Environment Customization

```python
from netheal.faults.injector import FaultType

# Advanced environment configuration
env = NetworkTroubleshootingEnv(
    max_devices=10,                          # Network size range: 3-10 devices
    max_episode_steps=25,                    # Longer episodes for complex networks
    topology_types=["hierarchical", "mesh"], # Focus on complex topologies
    fault_types=[                            # Specific fault types
        FaultType.DEVICE_FAILURE,
        FaultType.LINK_FAILURE,
        FaultType.PERFORMANCE_DEGRADATION
    ],
    render_mode="text"                       # Enable text rendering
)
```

### Custom Reward Shaping

You can adjust how the reward scales with network size by setting the `reward_scaling_factor` when creating the environment. A smaller value will make the reward more sensitive to network size.

```python
# Create an environment with a custom reward scaling factor
env = NetworkTroubleshootingEnv(
    max_devices=10,
    reward_scaling_factor=5.0  # Increase reward sensitivity to network size
)
```

### Network Topology Control

```python
from netheal.network.topology import TopologyGenerator

# Generate specific network for testing
network = TopologyGenerator.generate_hierarchical_topology(
    num_layers=3,
    devices_per_layer=[2, 3, 4]
)

# You can inject this into episodes by modifying the environment
# (requires extending the base class)
```

## Evaluation & Debugging

### Episode Analysis

```python
def evaluate_agent(model, env, num_episodes=100):
    """Evaluate trained agent performance."""
    success_count = 0
    episode_lengths = []
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(env.max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                # Check if diagnosis was correct
                if info.get('reward_breakdown', {}).get('diagnosis_reward', 0) > 0:  # Correct diagnosis
                    success_count += 1
                break
        
        episode_lengths.append(steps)
        episode_rewards.append(total_reward)
    
    print(f"Success Rate: {success_count/num_episodes:.2%}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f}")
    print(f"Average Episode Reward: {np.mean(episode_rewards):.2f}")
    
    return {
        'success_rate': success_count/num_episodes,
        'avg_length': np.mean(episode_lengths),
        'avg_reward': np.mean(episode_rewards)
    }

# Evaluate your trained model
results = evaluate_agent(model, env, num_episodes=100)
```

### Debugging Tools

```python
# Enable detailed rendering
env = NetworkTroubleshootingEnv(render_mode="text")

# Step through episode with detailed output
obs, info = env.reset()
print(f"Initial network: {info['network_size']} devices")
print(f"Ground truth fault: {info['ground_truth_fault']}")

for step in range(10):
    valid_actions = env.get_valid_actions()
    action = valid_actions[0] if valid_actions else 0
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Detailed step information
    print(f"\nStep {step}:")
    print(f"  Action: {info.get('action_spec', {}).get('description', 'Unknown')}")
    print(f"  Reward: {reward:.2f}")
    print(f"  Reward breakdown: {info.get('reward_breakdown', {})}")
    print(f"  Discovered devices: {info['discovered_devices']}")
    
    # Render current state
    env.render()
    
    if terminated or truncated:
        print(f"Episode ended: {'SUCCESS' if info.get('reward_breakdown', {}).get('diagnosis_reward', 0) > 0 else 'FAILURE'}")
        break
```

### Performance Monitoring

```python
import time
from collections import defaultdict

def profile_environment():
    """Profile environment performance."""
    env = NetworkTroubleshootingEnv()
    
    reset_times = []
    step_times = []
    action_counts = defaultdict(int)
    
    for episode in range(100):
        # Time reset
        start_time = time.time()
        obs, info = env.reset()
        reset_times.append(time.time() - start_time)
        
        for step in range(env.max_episode_steps):
            valid_actions = env.get_valid_actions()
            action = valid_actions[0] if valid_actions else 0
            action_counts[action] += 1
            
            # Time step
            start_time = time.time()
            obs, reward, terminated, truncated, info = env.step(action)
            step_times.append(time.time() - start_time)
            
            if terminated or truncated:
                break
    
    print(f"Average reset time: {np.mean(reset_times)*1000:.2f}ms")
    print(f"Average step time: {np.mean(step_times)*1000:.2f}ms")
    print(f"Most common actions: {sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:5]}")

profile_environment()
```

## Best Practices

### 1. Environment Configuration

- **Start small**: Begin with `max_devices=5-6` and `max_episode_steps=15-20`
- **Gradual complexity**: Increase network size and episode length as agent improves
- **Topology variety**: Train on multiple topology types for generalization

### 2. Training Strategies

- **Multi-input policies**: Use `MultiInputPolicy` for dict observation spaces
- **Curriculum learning**: Start with simple scenarios, gradually increase difficulty
- **Reward monitoring**: Track the final reward to assess agent performance on correct vs. incorrect diagnoses.

### 3. Action Space Management

- **Valid actions**: Always use `env.get_valid_actions()` to avoid invalid actions
- **Action masking**: Consider implementing action masking for more efficient training

### 4. Observation Processing

```python
def preprocess_observation(obs):
    """Preprocess NetHeal observations for training."""
    # Flatten or reshape observations if needed
    discovery = obs['discovery_matrix'].flatten()
    devices = obs['device_status'].flatten()
    diagnostics = obs['recent_diagnostics'].flatten()
    metadata = obs['episode_metadata']
    
    # Concatenate all features (if using non-dict policies)
    # return np.concatenate([discovery, devices, diagnostics, metadata])
    
    # Or return dict for MultiInputPolicy
    return obs
```

### 5. Evaluation Metrics

Track these key metrics during training:

- **Success rate**: Percentage of episodes with correct diagnosis
- **Episode efficiency**: Average steps to successful diagnosis
- **Exploration coverage**: Percentage of network discovered before diagnosis

### 6. Common Pitfalls

- **Invalid actions**: Always check valid actions before selection
- **Observation space**: Ensure your model handles dict observation spaces correctly
- **Reward interpretation**: The only positive reward comes from a correct final diagnosis. All other steps will have a small negative reward.
- **Episode termination**: Remember that diagnosis actions terminate episodes

## Example Training Script

Here's a complete example for training an agent:

```python
#!/usr/bin/env python3
"""
Complete NetHeal RL training example with Stable Baselines3
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from netheal import NetworkTroubleshootingEnv

def main():
    # Create environment
    env = NetworkTroubleshootingEnv(
        max_devices=6,
        max_episode_steps=15,
        topology_types=["star", "mesh", "hierarchical"],
        render_mode=None  # Disable rendering for training
    )
    
    # Verify environment
    check_env(env)
    print("Environment check passed!")
    
    # Wrap environment with Monitor for logging
    env = Monitor(env, "./logs/")
    
    # Create evaluation environment
    eval_env = Monitor(
        NetworkTroubleshootingEnv(
            max_devices=6,
            max_episode_steps=15,
            topology_types=["star", "mesh", "hierarchical"]
        ),
        "./logs/eval/"
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/eval/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Create PPO model
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./tensorboard/"
    )
    
    # Train the model
    print("Starting training...")
    model.learn(
        total_timesteps=500000,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save("netheal_final_model")
    print("Training completed!")
    
    # Quick evaluation
    print("\nEvaluating trained agent...")
    obs, info = eval_env.reset()
    for step in range(20):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        print(f"Step {step}: Reward {reward:.2f}")
        if terminated or truncated:
            success = "SUCCESS" if info.get('reward_breakdown', {}).get('diagnosis_reward', 0) > 0 else "FAILURE"
            print(f"Episode result: {success}")
            break
    
    env.close()
    eval_env.close()

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("./logs/", exist_ok=True)
    os.makedirs("./logs/eval/", exist_ok=True)
    os.makedirs("./models/", exist_ok=True)
    os.makedirs("./tensorboard/", exist_ok=True)
    
    main()
```

This guide provides the necessary details to successfully train RL agents using NetHeal's Gymnasium interface. The simplified, sparse-reward environment is designed to be a challenging and realistic testbed for developing robust troubleshooting agents.
