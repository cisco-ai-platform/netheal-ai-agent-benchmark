# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Quick NetHeal Demo - Simple command-line demonstration

A streamlined version of the NetHeal demo for quick testing and validation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
from netheal import NetworkTroubleshootingEnv
from netheal.network.topology import TopologyGenerator
from netheal.faults.injector import FaultInjector, FaultType


def quick_demo():
    """Run a quick demonstration of NetHeal capabilities."""
    print("NetHeal Quick Demo")
    print("-" * 40)
    
    # 1. Create and show network
    print("\n1. Creating network topology...")
    network = TopologyGenerator.generate_star_topology(4)
    print(f"   Created star topology with {len(network)} devices")
    print(f"   Devices: {', '.join(network.get_all_devices())}")
    
    # 2. Inject fault
    print("\n2. Injecting network fault...")
    fault_injector = FaultInjector(network)
    fault = fault_injector.inject_random_fault([FaultType.LINK_FAILURE])
    print(f"   Injected: {fault}")
    
    # 3. Test RL environment
    print("\n3. Testing RL environment...")
    env = NetworkTroubleshootingEnv(max_devices=5, max_episode_steps=5)
    obs, info = env.reset(seed=42)
    
    print(f"   Network size: {info['network_size']} devices")
    print(f"   Ground truth: {info['ground_truth_fault']}")
    print(f"   Action space: {env.action_space.n} actions")
    
    # Take a few random actions
    for step in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   Step {step+1}: Action {action}, Reward {reward:.1f}")
        
        if terminated or truncated:
            break
    
    env.close()
    print("\n✅ Quick demo completed successfully!")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    quick_demo()
