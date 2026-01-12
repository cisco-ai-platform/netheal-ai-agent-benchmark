"""
Basic usage example for NetHeal environment.

This script demonstrates how to use the NetworkTroubleshootingEnv
for both manual interaction and random agent testing.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
from netheal import NetworkTroubleshootingEnv
from netheal.utils import NetworkVisualizer


def manual_interaction_example():
    """Example of manual interaction with the environment."""
    print("=== Manual Interaction Example ===")
    
    # Create environment
    env = NetworkTroubleshootingEnv(
        max_devices=6,
        max_episode_steps=15,
        render_mode="text"
    )
    
    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"Episode started with {info['network_size']} devices")
    print(f"Ground truth fault: {info['ground_truth_fault']}")
    if 'user_hint' in info and info['user_hint']:
        print(f"User hint: {info['user_hint']}")
    
    # Show available actions
    action_meanings = env.get_action_meanings()
    print(f"\nTotal available actions: {len(action_meanings)}")
    print("Sample actions:")
    for i, meaning in enumerate(action_meanings[:10]):
        print(f"  {i}: {meaning}")
    
    # Take some sample actions
    print("\n=== Taking Sample Actions ===")
    
    # Try a few diagnostic actions
    sample_actions = [0, 1, len(action_meanings)//4, len(action_meanings)//2]
    
    for action in sample_actions:
        if action < len(action_meanings):
            print(f"\nTaking action {action}: {action_meanings[action]}")
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Reward: {reward}")
            print(f"Terminated: {terminated}, Truncated: {truncated}")
            
            if 'action_result' in info and info['action_result'] and hasattr(info['action_result'], 'result') and info['action_result'].result:
                result = info['action_result'].result
                print(f"Tool result: Success={result.success}, Data={result.data}")
            
            env.render()
            
            if terminated or truncated:
                print("Episode ended!")
                break
    
    env.close()


def random_agent_example():
    """Example of a random agent interacting with the environment."""
    print("\n=== Random Agent Example ===")
    
    env = NetworkTroubleshootingEnv(
        max_devices=5,
        max_episode_steps=10
    )
    
    num_episodes = 3
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        
        obs, info = env.reset()
        total_reward = 0
        step_count = 0
        
        print(f"Network size: {info['network_size']} devices")
        print(f"Ground truth: {info['ground_truth_fault']}")
        if 'user_hint' in info and info['user_hint']:
            print(f"User hint: {info['user_hint']}")
        
        while True:
            # Take random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            print(f"Step {step_count}: Action {action}, Reward {reward}")
            
            if terminated or truncated:
                print(f"Episode ended! Total reward: {total_reward}")
                break
    
    env.close()


def visualization_example():
    """Example of network visualization."""
    print("\n=== Visualization Example ===")
    
    try:
        import matplotlib.pyplot as plt
        
        env = NetworkTroubleshootingEnv(max_devices=6)
        obs, info = env.reset(seed=123)
        
        # Get network and fault info
        network = env.network
        fault = env.ground_truth_fault
        
        # Create visualizer
        visualizer = NetworkVisualizer()
        
        # Plot network with fault
        fig = visualizer.plot_network(
            network, 
            faults=[fault],
            title="Network with Injected Fault"
        )
        
        # Save plot
        plt.savefig('/Users/askazemi/Desktop/windsurf projects/netheal/examples/network_example.png', 
                   dpi=150, bbox_inches='tight')
        print("Network visualization saved as 'network_example.png'")
        
        plt.close()
        
    except ImportError:
        print("Matplotlib not available for visualization")
    
    env.close()


def diagnostic_tools_example():
    """Example of using diagnostic tools directly."""
    print("\n=== Diagnostic Tools Example ===")
    
    from netheal.network.topology import TopologyGenerator
    from netheal.faults.injector import FaultInjector, FaultType
    from netheal.tools.simulator import ToolSimulator
    
    # Create a simple network
    network = TopologyGenerator.generate_linear_topology(4)
    print(f"Created network with {len(network)} devices")
    
    # Inject a fault
    fault_injector = FaultInjector(network)
    fault = fault_injector.inject_random_fault([FaultType.LINK_FAILURE])
    print(f"Injected fault: {fault}")
    
    # Use diagnostic tools
    tool_sim = ToolSimulator(network)
    
    devices = network.get_all_devices()
    
    # Try ping between first and last device
    if len(devices) >= 2:
        source, dest = devices[0], devices[-1]
        
        print(f"\nPinging from {source} to {dest}:")
        ping_result = tool_sim.ping(source, dest)
        print(f"Result: {ping_result}")
        
        print(f"\nTraceroute from {source} to {dest}:")
        trace_result = tool_sim.traceroute(source, dest)
        print(f"Result: {trace_result}")
        
        print(f"\nChecking status of {source}:")
        status_result = tool_sim.check_status(source)
        print(f"Result: {status_result}")
        
        print(f"\nChecking interfaces of {source}:")
        interface_result = tool_sim.check_interfaces(source)
        print(f"Result: {interface_result}")


if __name__ == "__main__":
    print("NetHeal - Network Troubleshooting RL Environment")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run examples
    manual_interaction_example()
    random_agent_example()
    diagnostic_tools_example()
    visualization_example()
    
    print("\n" + "=" * 50)
    print("Examples completed successfully!")
