# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Enhanced action space for network troubleshooting RL environment.

This module provides structured actions that enable topology discovery,
targeted diagnostics, and hypothesis testing.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from netheal.faults import FaultType


class ActionCategory(Enum):
    """Categories of actions available to the agent."""
    TOPOLOGY_DISCOVERY = "topology_discovery"
    DIAGNOSTIC = "diagnostic"
    DIAGNOSIS = "diagnosis"


class TopologyAction(Enum):
    """Topology discovery actions."""
    DISCOVER_NEIGHBORS = "discover_neighbors"
    SCAN_NETWORK = "scan_network"


class DiagnosticAction(Enum):
    """Diagnostic actions for testing network components."""
    PING = "ping"
    TRACEROUTE = "traceroute"
    CHECK_STATUS = "check_status"
    CHECK_INTERFACES = "check_interfaces"




@dataclass
class ActionSpec:
    """Specification for a structured action."""
    category: ActionCategory
    action_type: Enum
    parameters: Dict[str, Any]
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'category': self.category.value,
            'action_type': self.action_type.value,
            'parameters': self.parameters,
            'description': self.description
        }


class StructuredActionSpace:
    """Manages the structured action space for network troubleshooting."""
    
    def __init__(self, max_devices: int = 10):
        """Initialize structured action space."""
        self.max_devices = max_devices
        self.action_specs = []
        self.action_map = {}  # int -> ActionSpec
        self._build_action_space()
    
    def _build_action_space(self):
        """Build the complete action space."""
        action_id = 0
        
        # Topology discovery actions
        action_id = self._add_topology_actions(action_id)
        
        # Diagnostic actions
        action_id = self._add_diagnostic_actions(action_id)
        
        
        # Diagnosis actions
        action_id = self._add_diagnosis_actions(action_id)
        
        self.total_actions = action_id
    
    def _add_topology_actions(self, start_id: int) -> int:
        """Add topology discovery actions."""
        action_id = start_id
        
        # Discover neighbors for each potential device
        for device_idx in range(self.max_devices):
            device_id = f"device_{device_idx}"
            spec = ActionSpec(
                category=ActionCategory.TOPOLOGY_DISCOVERY,
                action_type=TopologyAction.DISCOVER_NEIGHBORS,
                parameters={'device': device_id},
                description=f"Discover neighbors of {device_id}"
            )
            self.action_map[action_id] = spec
            action_id += 1
        
        # Network scan action
        spec = ActionSpec(
            category=ActionCategory.TOPOLOGY_DISCOVERY,
            action_type=TopologyAction.SCAN_NETWORK,
            parameters={},
            description="Perform broad network discovery scan"
        )
        self.action_map[action_id] = spec
        action_id += 1
        
        return action_id
    
    def rebuild_for_network(self, device_ids: List[str]) -> None:
        """Rebuild the action space using real device IDs for the current network."""
        # Reset current mappings
        self.action_map = {}
        action_id = 0

        # Topology discovery actions for each known device
        for device_id in device_ids:
            spec = ActionSpec(
                category=ActionCategory.TOPOLOGY_DISCOVERY,
                action_type=TopologyAction.DISCOVER_NEIGHBORS,
                parameters={'device': device_id},
                description=f"Discover neighbors of {device_id}"
            )
            self.action_map[action_id] = spec
            action_id += 1

        # Network scan action (always available)
        self.action_map[action_id] = ActionSpec(
            category=ActionCategory.TOPOLOGY_DISCOVERY,
            action_type=TopologyAction.SCAN_NETWORK,
            parameters={},
            description="Perform broad network discovery scan"
        )
        action_id += 1

        # Diagnostic actions (ordered pairs)
        for src in device_ids:
            for dst in device_ids:
                if src != dst:
                    self.action_map[action_id] = ActionSpec(
                        category=ActionCategory.DIAGNOSTIC,
                        action_type=DiagnosticAction.PING,
                        parameters={'source': src, 'destination': dst},
                        description=f"Ping from {src} to {dst}"
                    )
                    action_id += 1

        for src in device_ids:
            for dst in device_ids:
                if src != dst:
                    self.action_map[action_id] = ActionSpec(
                        category=ActionCategory.DIAGNOSTIC,
                        action_type=DiagnosticAction.TRACEROUTE,
                        parameters={'source': src, 'destination': dst},
                        description=f"Traceroute from {src} to {dst}"
                    )
                    action_id += 1

        for device_id in device_ids:
            self.action_map[action_id] = ActionSpec(
                category=ActionCategory.DIAGNOSTIC,
                action_type=DiagnosticAction.CHECK_STATUS,
                parameters={'device': device_id},
                description=f"Check status of {device_id}"
            )
            action_id += 1

        for device_id in device_ids:
            self.action_map[action_id] = ActionSpec(
                category=ActionCategory.DIAGNOSTIC,
                action_type=DiagnosticAction.CHECK_INTERFACES,
                parameters={'device': device_id},
                description=f"Check interfaces of {device_id}"
            )
            action_id += 1

        # Diagnosis actions using actual device/link locations
        fault_types = list(FaultType)
        for device_id in device_ids:
            for fault_type in fault_types:
                if fault_type in [FaultType.DEVICE_FAILURE, FaultType.MISCONFIGURATION]:
                    self.action_map[action_id] = ActionSpec(
                        category=ActionCategory.DIAGNOSIS,
                        action_type=fault_type,
                        parameters={'location': device_id},
                        description=f"Diagnose {fault_type.value} at {device_id}"
                    )
                    action_id += 1

        for src in device_ids:
            for dst in device_ids:
                if src != dst:
                    for fault_type in fault_types:
                        if fault_type in [FaultType.LINK_FAILURE, FaultType.PERFORMANCE_DEGRADATION]:
                            location = f"{src}->{dst}"
                            self.action_map[action_id] = ActionSpec(
                                category=ActionCategory.DIAGNOSIS,
                                action_type=fault_type,
                                parameters={'location': location},
                                description=f"Diagnose {fault_type.value} at {location}"
                            )
                            action_id += 1

        self.total_actions = action_id
    
    def _add_diagnostic_actions(self, start_id: int) -> int:
        """Add diagnostic actions."""
        action_id = start_id
        
        # Ping actions between device pairs
        for src_idx in range(self.max_devices):
            for dst_idx in range(self.max_devices):
                if src_idx != dst_idx:
                    src_device = f"device_{src_idx}"
                    dst_device = f"device_{dst_idx}"
                    spec = ActionSpec(
                        category=ActionCategory.DIAGNOSTIC,
                        action_type=DiagnosticAction.PING,
                        parameters={'source': src_device, 'destination': dst_device},
                        description=f"Ping from {src_device} to {dst_device}"
                    )
                    self.action_map[action_id] = spec
                    action_id += 1
        
        # Traceroute actions
        for src_idx in range(self.max_devices):
            for dst_idx in range(self.max_devices):
                if src_idx != dst_idx:
                    src_device = f"device_{src_idx}"
                    dst_device = f"device_{dst_idx}"
                    spec = ActionSpec(
                        category=ActionCategory.DIAGNOSTIC,
                        action_type=DiagnosticAction.TRACEROUTE,
                        parameters={'source': src_device, 'destination': dst_device},
                        description=f"Traceroute from {src_device} to {dst_device}"
                    )
                    self.action_map[action_id] = spec
                    action_id += 1
        
        # Status check actions
        for device_idx in range(self.max_devices):
            device_id = f"device_{device_idx}"
            spec = ActionSpec(
                category=ActionCategory.DIAGNOSTIC,
                action_type=DiagnosticAction.CHECK_STATUS,
                parameters={'device': device_id},
                description=f"Check status of {device_id}"
            )
            self.action_map[action_id] = spec
            action_id += 1
        
        # Interface check actions
        for device_idx in range(self.max_devices):
            device_id = f"device_{device_idx}"
            spec = ActionSpec(
                category=ActionCategory.DIAGNOSTIC,
                action_type=DiagnosticAction.CHECK_INTERFACES,
                parameters={'device': device_id},
                description=f"Check interfaces of {device_id}"
            )
            self.action_map[action_id] = spec
            action_id += 1
        
        return action_id
    
    
    def _add_diagnosis_actions(self, start_id: int) -> int:
        """Add final diagnosis actions."""
        action_id = start_id
        
        fault_types = list(FaultType)
        
        # Device diagnosis actions
        for device_idx in range(self.max_devices):
            device_id = f"device_{device_idx}"
            for fault_type in fault_types:
                if fault_type in [FaultType.DEVICE_FAILURE, FaultType.MISCONFIGURATION]:
                    spec = ActionSpec(
                        category=ActionCategory.DIAGNOSIS,
                        action_type=fault_type,
                        parameters={'location': device_id},
                        description=f"Diagnose {fault_type.value} at {device_id}"
                    )
                    self.action_map[action_id] = spec
                    action_id += 1
        
        # Connection diagnosis actions
        for src_idx in range(self.max_devices):
            for dst_idx in range(self.max_devices):
                if src_idx != dst_idx:
                    connection = f"device_{src_idx}->device_{dst_idx}"
                    for fault_type in fault_types:
                        if fault_type in [FaultType.LINK_FAILURE, FaultType.PERFORMANCE_DEGRADATION]:
                            spec = ActionSpec(
                                category=ActionCategory.DIAGNOSIS,
                                action_type=fault_type,
                                parameters={'location': connection},
                                description=f"Diagnose {fault_type.value} at {connection}"
                            )
                            self.action_map[action_id] = spec
                            action_id += 1
        
        return action_id
    
    def get_action_spec(self, action_id: int) -> Optional[ActionSpec]:
        """Get action specification by ID."""
        return self.action_map.get(action_id)
    
    def get_valid_actions(self, discovered_devices: List[str]) -> List[int]:
        """Get list of valid action IDs based on discovered devices."""
        valid_actions = []
        
        for action_id, spec in self.action_map.items():
            if self._is_action_valid(spec, discovered_devices):
                valid_actions.append(action_id)
        
        return valid_actions
    
    def _is_action_valid(self, spec: ActionSpec, discovered_devices: List[str]) -> bool:
        """Check if an action is valid given discovered devices."""
        # Topology discovery actions are always valid
        if spec.category == ActionCategory.TOPOLOGY_DISCOVERY:
            if spec.action_type in [TopologyAction.SCAN_NETWORK]:
                return True
            # Discover neighbors only valid for known devices
            device = spec.parameters.get('device')
            return device in discovered_devices if device else False
        
        # Diagnostic actions require discovered devices
        if spec.category == ActionCategory.DIAGNOSTIC:
            source = spec.parameters.get('source')
            destination = spec.parameters.get('destination')
            device = spec.parameters.get('device')
            
            if source and destination:
                return source in discovered_devices and destination in discovered_devices
            elif device:
                return device in discovered_devices
            return False
        
        # Diagnosis actions require discovered locations
        if spec.category == ActionCategory.DIAGNOSIS:
            location = spec.parameters.get('location', '')

            # Device location
            if '->' not in location:
                return location in discovered_devices

            # Connection location
            if '->' in location:
                source, dest = location.split('->', 1)
                return source in discovered_devices and dest in discovered_devices
        
        return True
    
    def get_actions_by_category(self, category: ActionCategory) -> List[Tuple[int, ActionSpec]]:
        """Get all actions in a specific category."""
        return [(action_id, spec) for action_id, spec in self.action_map.items()
                if spec.category == category]
    
    def get_action_descriptions(self) -> List[str]:
        """Get human-readable descriptions of all actions."""
        descriptions = []
        for i in range(self.total_actions):
            spec = self.action_map.get(i)
            if spec:
                descriptions.append(f"{spec.category.value}:{spec.action_type.value}")
            else:
                descriptions.append(f"invalid_action_{i}")
        return descriptions


def validate_action_parameters(action_type, parameters: Dict[str, Any]) -> bool:
    """Validate parameters for a given action type."""
    if isinstance(action_type, DiagnosticAction):
        if action_type in [DiagnosticAction.PING, DiagnosticAction.TRACEROUTE]:
            return 'source' in parameters and 'destination' in parameters
        elif action_type in [DiagnosticAction.CHECK_STATUS, DiagnosticAction.CHECK_INTERFACES]:
            return 'device' in parameters
    
    elif isinstance(action_type, TopologyAction):
        if action_type == TopologyAction.SCAN_NETWORK:
            return True  # start_device is optional
        elif action_type == TopologyAction.DISCOVER_NEIGHBORS:
            return 'device' in parameters
    
    
    
    elif isinstance(action_type, FaultType):  # Diagnosis actions
        return 'location' in parameters
    
    return False
