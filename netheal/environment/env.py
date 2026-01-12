# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Enhanced RL environment for network troubleshooting simulation.

This module provides the NetworkTroubleshootingEnv class that implements
the OpenAI Gymnasium interface with graph-aware observations and structured actions.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
import time

from ..network.graph import NetworkGraph, DeviceType
from ..network.topology import TopologyGenerator
from ..faults.injector import FaultInjector, FaultType, FaultInfo
from ..tools.simulator import ToolSimulator, ToolResult
from netheal.environment.observation import StructuredObservation, DiagnosticResult, DeviceStatus, ConnectionStatus
from netheal.environment.actions import StructuredActionSpace, ActionSpec, ActionCategory, TopologyAction, DiagnosticAction
from netheal.environment.rewards import SparseRewardCalculator
from netheal.hints.provider import get_default_hint_provider, BaseHintProvider


# Legacy ActionType enum - kept for compatibility
class ActionType(Enum):
    """Legacy action types - use ActionCategory and specific action enums instead."""
    PING = "ping"
    TRACEROUTE = "traceroute"
    CHECK_STATUS = "check_status"
    CHECK_INTERFACES = "check_interfaces"
    DIAGNOSE = "diagnose"


class NetworkTroubleshootingEnv(gym.Env):
    """
    Enhanced RL environment for network troubleshooting simulation.
    
    Features:
    - Graph-aware observation space with network topology discovery
    - Structured action space with topology discovery and hypothesis testing
    - Information-gain based reward system
    - Diagnostic results memory and fault hypothesis tracking
    """
    
    metadata = {"render_modes": ["human", "text"], "render_fps": 1}
    
    def __init__(self, 
                 max_devices: int = 10,
                 max_episode_steps: int = 20,
                 topology_types: List[str] = None,
                 fault_types: List[FaultType] = None,
                 reward_scaling_factor: float = 10.0,
                 render_mode: Optional[str] = None,
                 enable_user_hints: bool = True,
                 hint_provider_mode: str = "auto",
                 user_context: Optional[Dict[str, Any]] = None,
                 hint_provider: Optional[BaseHintProvider] = None):
        """
        Initialize the enhanced network troubleshooting environment.
        
        Args:
            max_devices: Maximum number of devices in generated networks
            max_episode_steps: Maximum steps per episode
            topology_types: List of topology types to generate
            fault_types: List of fault types to inject
            reward_scaling_factor: Factor to scale rewards by network size.
            render_mode: Rendering mode
        """
        super().__init__()
        
        self.max_devices = max_devices
        self.max_episode_steps = max_episode_steps
        self.topology_types = topology_types or ["linear", "star", "mesh", "hierarchical", "random"]
        self.fault_types = fault_types or list(FaultType)
        
        # Environment state
        self.network: Optional[NetworkGraph] = None
        self.fault_injector: Optional[FaultInjector] = None
        self.tool_simulator: Optional[ToolSimulator] = None
        self.ground_truth_fault: Optional[FaultInfo] = None
        
        # Enhanced observation and action systems
        self.observation: Optional[StructuredObservation] = None
        self.previous_observation: Optional[StructuredObservation] = None
        self.action_space_manager = StructuredActionSpace(max_devices)
        self.reward_calculator = SparseRewardCalculator(scaling_factor=reward_scaling_factor)
        
        # Episode state
        self.step_count = 0
        self.episode_done = False
        self.episode_start_time = 0.0
        
        # User hint configuration/state
        self.enable_user_hints = enable_user_hints
        self.hint_provider_mode = hint_provider_mode
        self.user_context = user_context or {}
        self.hint_provider = hint_provider
        self.user_hint: Optional[str] = None
        
        # Action and observation spaces
        self._setup_action_space()
        self._setup_observation_space()
        
        self.render_mode = render_mode
        
    def _setup_action_space(self):
        """Setup the structured action space."""
        self.action_space = spaces.Discrete(self.action_space_manager.total_actions)
        
    def _setup_observation_space(self):
        """Setup the dictionary-based observation space."""
        # Simplified observation space for compatibility
        self.observation_space = spaces.Dict({
            'discovery_matrix': spaces.Box(
                low=-1, high=2, 
                shape=(self.max_devices, self.max_devices), 
                dtype=np.int8
            ),
            'device_status': spaces.Box(
                low=0, high=1, 
                shape=(self.max_devices, 10),  # 10 status features per device
                dtype=np.float32
            ),
            'recent_diagnostics': spaces.Box(
                low=0, high=1,
                shape=(10, 6),  # 10 recent results, 6 features each
                dtype=np.float32
            ),
            'episode_metadata': spaces.Box(
                low=0, high=1,
                shape=(4,),  # step, progress, discovered_devices, diagnostic_actions
                dtype=np.float32
            )
        })
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, Any], Dict]:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Initial structured observation and info dict
        """
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Generate new network topology
        self._generate_network()
        
        # Rebuild action space using real device IDs from the generated network
        try:
            device_ids = self.network.get_all_devices()
            self.action_space_manager.rebuild_for_network(device_ids)
            self.action_space = spaces.Discrete(self.action_space_manager.total_actions)
        except Exception:
            # Fallback to existing action space if any unexpected issue occurs
            self._setup_action_space()
        
        # Initialize components
        self.fault_injector = FaultInjector(self.network)
        self.tool_simulator = ToolSimulator(self.network)
        
        # Inject a random fault
        self.ground_truth_fault = self.fault_injector.inject_random_fault(self.fault_types)
        
        # Initialize structured observation
        self.observation = StructuredObservation(self.max_devices)
        self.observation.episode_step = 0
        self.observation.max_episode_steps = self.max_episode_steps
        self.previous_observation = None
        
        # Reset episode state
        self.step_count = 0
        self.episode_done = False
        self.episode_start_time = time.time()
        
        # Generate a user hint (non-leaky) if enabled
        self.user_hint = None
        if self.enable_user_hints:
            try:
                provider = self.hint_provider or get_default_hint_provider(self.hint_provider_mode)
                gt = {
                    'type': self.ground_truth_fault.fault_type.value if self.ground_truth_fault else None,
                    'location': self.ground_truth_fault.location if self.ground_truth_fault else None,
                }
                context = {
                    'ground_truth': gt,
                    'network_size': len(self.network.get_all_devices()),
                    'topology_types': self.topology_types,
                    'user_context': self.user_context,
                }
                self.user_hint = provider.generate_hint(context)
            except Exception:
                # Be resilient; hints are advisory
                self.user_hint = None
        
        # Get initial observation
        observation_dict = self.observation.to_dict()
        info = self._get_info()
        
        return observation_dict, info
    
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action ID to execute
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.episode_done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        
        # Store previous observation for reward calculation
        self.previous_observation = StructuredObservation(self.max_devices)
        self.previous_observation.discovery_matrix = self.observation.discovery_matrix
        self.previous_observation.device_status = self.observation.device_status
        self.previous_observation.diagnostic_memory = self.observation.diagnostic_memory
        
        self.step_count += 1
        self.observation.episode_step = self.step_count
        
        # Get action specification
        action_spec = self.action_space_manager.get_action_spec(action)
        if action_spec is None:
            # Invalid action
            return self._handle_invalid_action(action)
        
        # Execute action
        result = self._execute_action(action_spec)
        
        # Update observation with results
        if result:
            self.observation.update_from_diagnostic_result(result)
        
        # Calculate enhanced reward
        reward, reward_breakdown = self.reward_calculator.calculate_reward(
            action_spec, self.ground_truth_fault, len(self.network.get_all_devices())
        )
        
        # Check if episode is done
        terminated = self._is_terminated(action_spec)
        truncated = self.step_count >= self.max_episode_steps
        
        if terminated or truncated:
            self.episode_done = True
        
        # Get observation and info
        observation_dict = self.observation.to_dict()
        info = self._get_info()
        info.update({
            'action_spec': action_spec.to_dict() if action_spec else None,
            'action_result': result,
            'reward_breakdown': reward_breakdown,
            'valid_actions': self.get_valid_actions()
        })
        
        return observation_dict, reward, terminated, truncated, info
    
    def _generate_network(self):
        """Generate a random network topology."""
        num_devices = random.randint(3, self.max_devices)
        topology_type = random.choice(self.topology_types)
        
        if topology_type == "linear":
            self.network = TopologyGenerator.generate_linear_topology(num_devices)
        elif topology_type == "star":
            self.network = TopologyGenerator.generate_star_topology(num_devices - 1)
        elif topology_type == "mesh":
            self.network = TopologyGenerator.generate_mesh_topology(num_devices)
        elif topology_type == "hierarchical":
            layers = random.randint(2, 3)
            devices_per_layer = [random.randint(1, 3) for _ in range(layers)]
            self.network = TopologyGenerator.generate_hierarchical_topology(layers, devices_per_layer)
        elif topology_type == "random":
            self.network = TopologyGenerator.generate_random_topology(num_devices)
        else:
            # Default to random
            self.network = TopologyGenerator.generate_random_topology(num_devices)
    
    def _execute_action(self, action_spec: ActionSpec) -> Optional[DiagnosticResult]:
        """Execute an action and return diagnostic result."""
        if action_spec.category == ActionCategory.TOPOLOGY_DISCOVERY:
            return self._execute_topology_action(action_spec)
        elif action_spec.category == ActionCategory.DIAGNOSTIC:
            return self._execute_diagnostic_action(action_spec)
        elif action_spec.category == ActionCategory.DIAGNOSIS:
            return self._execute_diagnosis_action(action_spec)
        else:
            return None
    
    def _execute_topology_action(self, action_spec: ActionSpec) -> Optional[DiagnosticResult]:
        """Execute topology discovery actions."""
        if action_spec.action_type == TopologyAction.SCAN_NETWORK:
            # Broad discovery: add up to max_devices actual nodes and known connections among them
            all_devices = self.network.get_all_devices()
            discovered = []
            for dev in all_devices[: self.max_devices]:
                try:
                    self.observation.discovery_matrix.add_device(dev)
                    discovered.append(dev)
                except Exception:
                    # Stop if discovery matrix capacity is reached
                    break
            
            # Mark direct connections among discovered set
            discovered_set = set(discovered)
            for src, dst in self.network.get_all_connections():
                if src in discovered_set and dst in discovered_set and self.network.is_connection_up(src, dst):
                    self.observation.discovery_matrix.update_connection(src, dst, ConnectionStatus.CONNECTED)
            
            # Create diagnostic result
            tool_result = ToolResult(
                tool_name='scan_network',
                success=True,
                data={'discovered_devices': discovered},
                cost=2.0
            )
            
            return DiagnosticResult(
                tool_name='scan_network',
                source=discovered[0] if discovered else None,
                destination=None,
                result=tool_result,
                timestamp=time.time()
            )
        
        elif action_spec.action_type == TopologyAction.DISCOVER_NEIGHBORS:
            device = action_spec.parameters.get('device')
            if device and device in self.network.get_all_devices():
                connections = self.network.get_device_connections(device)
                neighbors = [dest for dest, _ in connections]
                # Ensure source is recorded (ignore capacity errors)
                try:
                    self.observation.discovery_matrix.add_device(device)
                except Exception:
                    pass
                for neighbor in neighbors:
                    added = True
                    try:
                        self.observation.discovery_matrix.add_device(neighbor)
                    except Exception:
                        added = False  # skip if capacity reached
                    if added:
                        self.observation.discovery_matrix.update_connection(device, neighbor, ConnectionStatus.CONNECTED)
                
                tool_result = ToolResult(
                    tool_name='discover_neighbors',
                    success=True,
                    data={'neighbors': neighbors},
                    cost=1.0
                )
                
                return DiagnosticResult(
                    tool_name='discover_neighbors',
                    source=device,
                    destination=None,
                    result=tool_result,
                    timestamp=time.time()
                )
        
        return None
    
    def _execute_diagnostic_action(self, action_spec: ActionSpec) -> Optional[DiagnosticResult]:
        """Execute diagnostic actions using tool simulator."""
        source = action_spec.parameters.get('source')
        destination = action_spec.parameters.get('destination')
        device = action_spec.parameters.get('device')
        
        if action_spec.action_type == DiagnosticAction.PING:
            if source and destination:
                result = self.tool_simulator.ping(source, destination)
                return DiagnosticResult(
                    tool_name='ping',
                    source=source,
                    destination=destination,
                    result=result,
                    timestamp=time.time()
                )
        
        elif action_spec.action_type == DiagnosticAction.TRACEROUTE:
            if source and destination:
                result = self.tool_simulator.traceroute(source, destination)
                return DiagnosticResult(
                    tool_name='traceroute',
                    source=source,
                    destination=destination,
                    result=result,
                    timestamp=time.time()
                )
        
        elif action_spec.action_type == DiagnosticAction.CHECK_STATUS:
            if device:
                result = self.tool_simulator.check_status(device)
                return DiagnosticResult(
                    tool_name='check_status',
                    source=device,
                    destination=None,
                    result=result,
                    timestamp=time.time()
                )
        
        elif action_spec.action_type == DiagnosticAction.CHECK_INTERFACES:
            if device:
                result = self.tool_simulator.check_interfaces(device)
                return DiagnosticResult(
                    tool_name='check_interfaces',
                    source=device,
                    destination=None,
                    result=result,
                    timestamp=time.time()
                )
        
        return None
    
    
    def _execute_diagnosis_action(self, action_spec: ActionSpec) -> Optional[DiagnosticResult]:
        """Execute final diagnosis action."""
        # This is handled in the reward calculation and termination logic
        return None
    
    def _handle_invalid_action(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict]:
        """Handle invalid action by returning current state with penalty."""
        observation_dict = self.observation.to_dict()
        reward = -5.0  # Penalty for invalid action
        terminated = False
        truncated = self.step_count >= self.max_episode_steps
        
        info = self._get_info()
        info.update({
            'action_spec': None,
            'action_result': None,
            'reward_breakdown': {'invalid_action_penalty': reward},
            'valid_actions': self.get_valid_actions(),
            'error': f'Invalid action ID: {action}'
        })
        
        return observation_dict, reward, terminated, truncated, info
    
    def get_valid_actions(self) -> List[int]:
        """Get list of currently valid action IDs."""
        return self.action_space_manager.get_valid_actions(
            self.observation.discovery_matrix.get_discovered_devices()
        )
    
    def _is_terminated(self, action_spec: ActionSpec) -> bool:
        """Check if episode should terminate."""
        return action_spec.category == ActionCategory.DIAGNOSIS
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about current state."""
        return {
            'step_count': self.step_count,
            'network_size': len(self.network.get_all_devices()),
            'ground_truth_fault': {
                'type': self.ground_truth_fault.fault_type.value,
                'location': self.ground_truth_fault.location
            } if self.ground_truth_fault else None,
            'discovered_devices': len(self.observation.discovery_matrix.get_discovered_devices()),
            'episode_done': self.episode_done,
            'episode_progress': self.step_count / self.max_episode_steps,
            'user_hint': self.user_hint
        }
    
    
    def render(self, mode: str = "human"):
        """Render the current state of the environment."""
        if mode == "human" or mode == "text":
            print(f"\n=== Enhanced Network Troubleshooting Environment ===")
            print(f"Step: {self.step_count}/{self.max_episode_steps}")
            print(f"Network: {len(self.network.get_all_devices())} devices, {len(self.network.get_all_connections())} connections")
            print(f"Discovered: {len(self.observation.discovery_matrix.get_discovered_devices())} devices")
            
            if self.ground_truth_fault:
                print(f"Ground Truth Fault: {self.ground_truth_fault}")
            
            
            # Show recent diagnostic results
            recent_results = self.observation.diagnostic_memory.get_recent_results(3)
            if recent_results:
                print("\nRecent Diagnostics:")
                for result in recent_results:
                    status = "✓" if result.result.success else "✗"
                    dest_str = f" -> {result.destination}" if result.destination else ""
                    print(f"  {status} {result.tool_name}: {result.source}{dest_str}")
    
    def close(self):
        """Clean up the environment."""
        pass
    
    def get_action_meanings(self) -> List[str]:
        """Get human-readable meanings for actions."""
        return self.action_space_manager.get_action_descriptions()
