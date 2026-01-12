# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Interactive NetHeal Demo - Fault Injection Workflow Visualization

This script provides a comprehensive demonstration of the NetHeal environment
with step-by-step fault injection, diagnostic workflows, and visual feedback.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import time
from typing import Dict, List, Any
import matplotlib.pyplot as plt

from netheal import NetworkTroubleshootingEnv
from netheal.network.topology import TopologyGenerator
from netheal.faults.injector import FaultInjector, FaultType
from netheal.tools.simulator import ToolSimulator
from netheal.utils import NetworkVisualizer


class NetHealDemo:
    """Interactive demonstration of NetHeal environment capabilities."""
    
    def __init__(self):
        """Initialize the demo."""
        self.visualizer = NetworkVisualizer()
        
    def print_header(self, title: str):
        """Print formatted section header."""
        print("\n" + "="*60)
        print(f"  {title}")
        print("="*60)
    
    def print_step(self, step: str, details: str = ""):
        """Print formatted step information."""
        print(f"\n🔹 {step}")
        if details:
            print(f"   {details}")
    
    def wait_for_input(self, prompt: str = "Press Enter to continue..."):
        """Wait for user input to proceed."""
        input(f"\n{prompt}")
    
    def demo_network_creation(self):
        """Demonstrate network topology generation."""
        self.print_header("STEP 1: Network Topology Generation")
        
        topologies = ["linear", "star", "mesh", "hierarchical"]
        
        for i, topology_type in enumerate(topologies, 1):
            self.print_step(f"Creating {topology_type} topology")
            
            if topology_type == "linear":
                network = TopologyGenerator.generate_linear_topology(5)
            elif topology_type == "star":
                network = TopologyGenerator.generate_star_topology(4)
            elif topology_type == "mesh":
                network = TopologyGenerator.generate_mesh_topology(4)
            elif topology_type == "hierarchical":
                network = TopologyGenerator.generate_hierarchical_topology(3, [2, 2, 2])
            
            print(f"   📊 Devices: {len(network.get_all_devices())}")
            print(f"   🔗 Connections: {len(network.get_all_connections())}")
            print(f"   📋 Device list: {', '.join(network.get_all_devices())}")
            
            if i < len(topologies):
                time.sleep(1)
    
    def demo_fault_injection_workflow(self):
        """Demonstrate the complete fault injection workflow."""
        self.print_header("STEP 2: Fault Injection Workflow")
        
        # Create a network for demonstration
        network = TopologyGenerator.generate_star_topology(4)
        fault_injector = FaultInjector(network)
        
        self.print_step("Initial Network State")
        self._print_network_state(network)
        
        self.wait_for_input("Ready to inject faults? Press Enter...")
        
        # Demonstrate each fault type
        fault_types = [
            (FaultType.DEVICE_FAILURE, "Device goes down"),
            (FaultType.LINK_FAILURE, "Connection breaks"),
            (FaultType.MISCONFIGURATION, "Firewall blocks traffic"),
            (FaultType.PERFORMANCE_DEGRADATION, "High latency")
        ]
        
        injected_faults = []
        
        for fault_type, description in fault_types:
            self.print_step(f"Injecting {fault_type.value}", description)
            
            try:
                if fault_type == FaultType.DEVICE_FAILURE:
                    fault = fault_injector.inject_device_failure()
                elif fault_type == FaultType.LINK_FAILURE:
                    fault = fault_injector.inject_link_failure()
                elif fault_type == FaultType.MISCONFIGURATION:
                    fault = fault_injector.inject_misconfiguration()
                elif fault_type == FaultType.PERFORMANCE_DEGRADATION:
                    fault = fault_injector.inject_performance_degradation()
                
                injected_faults.append(fault)
                
                print(f"   ❌ Fault injected: {fault}")
                print(f"   📍 Location: {fault.location}")
                print(f"   📝 Details: {fault.details}")
                
                self._print_network_state(network, highlight_fault=fault)
                
            except ValueError as e:
                print(f"   ⚠️  Could not inject {fault_type.value}: {e}")
            
            time.sleep(1)
        
        # Show fault restoration
        self.wait_for_input("\nReady to restore network? Press Enter...")
        
        self.print_step("Restoring Network to Healthy State")
        fault_injector.clear_all_faults()
        print("   ✅ All faults cleared")
        self._print_network_state(network)
        
        return network, injected_faults
    
    def demo_diagnostic_tools(self):
        """Demonstrate diagnostic tool usage."""
        self.print_header("STEP 3: Diagnostic Tools Demonstration")
        
        # Create network with a specific fault
        network = TopologyGenerator.generate_linear_topology(4)
        fault_injector = FaultInjector(network)
        tool_simulator = ToolSimulator(network)
        
        devices = network.get_all_devices()
        
        self.print_step("Network Before Fault")
        print(f"   📋 Devices: {', '.join(devices)}")
        
        # Inject a link failure
        fault = fault_injector.inject_link_failure()
        self.print_step(f"Injected Fault: {fault}")
        
        self.wait_for_input("Ready to run diagnostics? Press Enter...")
        
        # Demonstrate each diagnostic tool
        tools = [
            ("ping", "Test connectivity between devices"),
            ("traceroute", "Trace network path"),
            ("check_status", "Check device status"),
            ("check_interfaces", "Check device interfaces")
        ]
        
        for tool_name, description in tools:
            self.print_step(f"Using {tool_name}", description)
            
            if tool_name == "ping":
                result = tool_simulator.ping(devices[0], devices[-1])
            elif tool_name == "traceroute":
                result = tool_simulator.traceroute(devices[0], devices[-1])
            elif tool_name == "check_status":
                result = tool_simulator.check_status(devices[0])
            elif tool_name == "check_interfaces":
                result = tool_simulator.check_interfaces(devices[0])
            
            print(f"   📊 Success: {result.success}")
            print(f"   📄 Data: {result.data}")
            print(f"   💰 Cost: {result.cost:.1f}")
            
            time.sleep(1)
    
    def demo_rl_environment(self):
        """Demonstrate the RL environment in action."""
        self.print_header("STEP 4: RL Environment Demonstration")
        
        env = NetworkTroubleshootingEnv(
            max_devices=5,
            max_episode_steps=10,
            render_mode="text"
        )
        
        self.print_step("Initializing Environment")
        obs, info = env.reset(seed=42)
        
        print(f"   🌐 Network size: {info['network_size']} devices")
        print(f"   🎯 Ground truth fault: {info['ground_truth_fault']}")
        if 'user_hint' in info and info['user_hint']:
            print(f"   💡 User hint: {info['user_hint']}")
        print(f"   📊 Observation keys: {list(obs.keys())}")
        print(f"   🎮 Action space size: {env.action_space.n}")
        
        self.wait_for_input("Ready to see agent actions? Press Enter...")
        
        # Show some sample actions
        action_meanings = env.get_action_meanings()
        
        self.print_step("Sample Available Actions")
        for i in range(min(8, len(action_meanings))):
            print(f"   {i}: {action_meanings[i]}")
        
        # Simulate a few agent steps
        self.print_step("Simulating Agent Steps")
        
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"\n   Step {step + 1}:")
            print(f"   🎯 Action: {action} ({action_meanings[action] if action < len(action_meanings) else 'Invalid'})")
            print(f"   🏆 Reward: {reward}")
            print(f"   ✅ Terminated: {terminated}")
            
            if 'action_result' in info and info['action_result'] and hasattr(info['action_result'], 'result') and info['action_result'].result:
                result = info['action_result'].result
                print(f"   📊 Tool result: Success={result.success}, Data={result.data}")
            
            if terminated or truncated:
                print(f"   🏁 Episode ended!")
                break
            
            time.sleep(1)
        
        env.close()
    
    def demo_visualization(self):
        """Demonstrate network visualization capabilities."""
        self.print_header("STEP 5: Network Visualization")
        
        try:
            # Create network with fault
            network = TopologyGenerator.generate_star_topology(5)
            fault_injector = FaultInjector(network)
            fault = fault_injector.inject_random_fault()
            
            self.print_step("Creating Network Visualization")
            print(f"   📊 Network: {len(network)} devices")
            print(f"   ❌ Fault: {fault}")
            
            # Create visualization
            fig = self.visualizer.plot_network(
                network,
                faults=[fault],
                title="NetHeal Demo - Network with Fault",
                figsize=(10, 8)
            )
            
            # Save visualization
            save_path = "/Users/askazemi/Desktop/windsurf projects/netheal/examples/demo_network.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   💾 Visualization saved: {save_path}")
            
        except Exception as e:
            print(f"   ⚠️  Visualization error: {e}")
    
    def _print_network_state(self, network, highlight_fault=None):
        """Print current network state."""
        devices = network.get_all_devices()
        connections = network.get_all_connections()
        
        print(f"   📊 Network State:")
        print(f"      Devices: {len(devices)}")
        
        # Show device statuses
        for device in devices:
            status = "🟢" if network.is_device_up(device) else "🔴"
            device_info = network.get_device_info(device)
            print(f"        {status} {device} ({device_info['device_type'].value})")
        
        print(f"      Connections: {len(connections)}")
        
        # Show connection statuses
        for source, dest in connections:
            status = "🟢" if network.is_connection_up(source, dest) else "🔴"
            conn_info = network.get_connection_info(source, dest)
            latency = conn_info.get('latency', 0)
            
            highlight = ""
            if highlight_fault and f"{source}->{dest}" in str(highlight_fault.location):
                highlight = " ⚠️"
            
            print(f"        {status} {source} → {dest} ({latency:.1f}ms){highlight}")
    
    def run_complete_demo(self):
        """Run the complete demonstration."""
        print("🚀 Welcome to NetHeal Interactive Demo!")
        print("This demo will walk you through the fault injection workflow")
        
        self.wait_for_input("Ready to start? Press Enter...")
        
        # Run all demo sections
        self.demo_network_creation()
        self.demo_fault_injection_workflow()
        self.demo_diagnostic_tools()
        self.demo_rl_environment()
        self.demo_visualization()
        
        self.print_header("Demo Complete!")
        print("🎉 You've seen the complete NetHeal fault injection workflow!")
        print("📚 Key takeaways:")
        print("   • Networks are represented as directed graphs")
        print("   • Faults are injected programmatically with ground truth")
        print("   • Diagnostic tools simulate real network troubleshooting")
        print("   • RL agents learn to diagnose problems efficiently")
        print("   • Visualizations help understand network topology and faults")


def main():
    """Main demo function."""
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    demo = NetHealDemo()
    demo.run_complete_demo()


if __name__ == "__main__":
    main()
