# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Fault injection system for network troubleshooting simulation.

This module provides the FaultInjector class that can programmatically introduce
various types of network problems into a NetworkGraph.
"""

import random
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from ..network.graph import NetworkGraph


class FaultType(Enum):
    """Types of network faults that can be injected."""
    LINK_FAILURE = "link_failure"
    DEVICE_FAILURE = "device_failure" 
    MISCONFIGURATION = "misconfiguration"
    PERFORMANCE_DEGRADATION = "performance_degradation"


class FaultInfo:
    """Information about an injected fault."""
    
    def __init__(self, fault_type: FaultType, location: str, 
                 details: Dict[str, Any] = None):
        """
        Initialize fault information.
        
        Args:
            fault_type: Type of fault
            location: Location of fault (device ID or edge description)
            details: Additional fault details
        """
        self.fault_type = fault_type
        self.location = location
        self.details = details or {}
        
    def __str__(self) -> str:
        return f"{self.fault_type.value} at {self.location}"
    
    def __repr__(self) -> str:
        return f"FaultInfo({self.fault_type}, {self.location}, {self.details})"


class FaultSamplingStrategy(Enum):
    """Strategies for selecting fault types."""

    UNIFORM = "uniform"
    WEIGHTED = "weighted"
    ROUND_ROBIN = "round_robin"
    STRATIFIED = "stratified"


class FaultInjector:
    """
    Fault injection system for network graphs.
    
    This class can introduce various types of network problems into a NetworkGraph
    to create troubleshooting scenarios for RL training.
    """
    
    def __init__(
        self,
        network: NetworkGraph,
        rng: Optional[random.Random] = None,
        sampling_strategy: FaultSamplingStrategy = FaultSamplingStrategy.UNIFORM,
        fault_weights: Optional[Dict[Any, float]] = None,
        latency_multiplier_range: Tuple[float, float] = (10.0, 20.0),
    ):
        """
        Initialize fault injector.
        
        Args:
            network: NetworkGraph to inject faults into
        """
        self.network = network
        self.rng = rng or random
        self.active_faults: List[FaultInfo] = []
        self.sampling_strategy = (
            sampling_strategy
            if isinstance(sampling_strategy, FaultSamplingStrategy)
            else FaultSamplingStrategy(str(sampling_strategy))
        )
        self.fault_weights = self._normalize_fault_weights(fault_weights)
        self.latency_multiplier_range = latency_multiplier_range
        self._round_robin_index = 0
        
    def inject_random_fault(self, fault_types: List[FaultType] = None) -> FaultInfo:
        """
        Inject a random fault into the network.
        
        Args:
            fault_types: List of fault types to choose from (all types if None)
            
        Returns:
            FaultInfo object describing the injected fault
        """
        if fault_types is None:
            fault_types = list(FaultType)
            
        fault_type = self._select_fault_type(fault_types)
        
        if fault_type == FaultType.LINK_FAILURE:
            return self.inject_link_failure()
        elif fault_type == FaultType.DEVICE_FAILURE:
            return self.inject_device_failure()
        elif fault_type == FaultType.MISCONFIGURATION:
            return self.inject_misconfiguration()
        elif fault_type == FaultType.PERFORMANCE_DEGRADATION:
            return self.inject_performance_degradation()
        else:
            raise ValueError(f"Unknown fault type: {fault_type}")
    
    def inject_link_failure(self, source: str = None, destination: str = None) -> FaultInfo:
        """
        Inject a link failure by setting an edge status to 'down'.
        
        Args:
            source: Source device (random if None)
            destination: Destination device (random if None)
            
        Returns:
            FaultInfo object describing the injected fault
        """
        connections = self.network.get_all_connections()
        if not connections:
            raise ValueError("No connections available for link failure injection")
        
        if source is None or destination is None:
            # Choose random connection that is currently up
            up_connections = [(s, d) for s, d in connections 
                            if self.network.is_connection_up(s, d)]
            if not up_connections:
                raise ValueError("No active connections available for link failure")
            source, destination = self.rng.choice(up_connections)
        
        # Set connection status to down
        self.network.set_connection_status(source, destination, 'down')
        
        # If bidirectional, also set reverse direction to down
        if self.network.graph.has_edge(destination, source):
            self.network.set_connection_status(destination, source, 'down')
        
        fault_info = FaultInfo(
            FaultType.LINK_FAILURE,
            f"{source}->{destination}",
            {"source": source, "destination": destination}
        )
        self.active_faults.append(fault_info)
        
        return fault_info
    
    def inject_device_failure(self, device_id: str = None) -> FaultInfo:
        """
        Inject a device failure by setting a node status to 'down'.
        
        Args:
            device_id: Device to fail (random if None)
            
        Returns:
            FaultInfo object describing the injected fault
        """
        devices = self.network.get_all_devices()
        if not devices:
            raise ValueError("No devices available for device failure injection")
        
        if device_id is None:
            # Choose random device that is currently up
            up_devices = [d for d in devices if self.network.is_device_up(d)]
            if not up_devices:
                raise ValueError("No active devices available for device failure")
            device_id = self.rng.choice(up_devices)
        
        # Set device status to down
        self.network.set_device_status(device_id, 'down')
        
        fault_info = FaultInfo(
            FaultType.DEVICE_FAILURE,
            device_id,
            {"device_id": device_id}
        )
        self.active_faults.append(fault_info)
        
        return fault_info
    
    def inject_misconfiguration(self, device_id: str = None, 
                              blocked_destination: str = None) -> FaultInfo:
        """
        Inject a misconfiguration by simulating a blocked port on a firewall.
        This effectively acts like a selective link failure.
        
        Args:
            device_id: Device with misconfiguration (random firewall if None)
            blocked_destination: Destination to block (random if None)
            
        Returns:
            FaultInfo object describing the injected fault
        """
        devices = self.network.get_all_devices()
        
        if device_id is None:
            # Prefer firewall devices, but use any device if no firewalls
            from ..network.graph import DeviceType
            firewalls = [d for d in devices 
                        if self.network.get_device_info(d)['device_type'] == DeviceType.FIREWALL]
            
            if firewalls:
                device_id = self.rng.choice(firewalls)
            else:
                # Use any device with connections
                devices_with_connections = [d for d in devices 
                                          if self.network.get_device_connections(d)]
                if not devices_with_connections:
                    raise ValueError("No devices with connections for misconfiguration")
                device_id = self.rng.choice(devices_with_connections)
        
        # Get connections from this device
        connections = self.network.get_device_connections(device_id)
        if not connections:
            raise ValueError(f"Device {device_id} has no connections for misconfiguration")
        
        if blocked_destination is None:
            # Choose random connection that is currently up
            up_connections = [(dest, info) for dest, info in connections 
                            if info.get('status') == 'up']
            if not up_connections:
                raise ValueError(f"No active connections from {device_id} for misconfiguration")
            blocked_destination, _ = self.rng.choice(up_connections)
        
        # Block the connection (set to down)
        self.network.set_connection_status(device_id, blocked_destination, 'down')
        
        fault_info = FaultInfo(
            FaultType.MISCONFIGURATION,
            f"{device_id}",
            {
                "device_id": device_id,
                "blocked_destination": blocked_destination,
                "description": f"Blocked port to {blocked_destination}"
            }
        )
        self.active_faults.append(fault_info)
        
        return fault_info
    
    def inject_performance_degradation(
        self,
        source: str = None,
        destination: str = None,
        latency_multiplier: float = None,
    ) -> FaultInfo:
        """
        Inject performance degradation by increasing latency on a connection.
        
        Args:
            source: Source device (random if None)
            destination: Destination device (random if None)
            latency_multiplier: Factor to multiply latency by (random 10-20x if None)
            
        Returns:
            FaultInfo object describing the injected fault
        """
        connections = self.network.get_all_connections()
        if not connections:
            raise ValueError("No connections available for performance degradation")
        
        if source is None or destination is None:
            # Choose random connection that is currently up
            up_connections = [(s, d) for s, d in connections 
                            if self.network.is_connection_up(s, d)]
            if not up_connections:
                raise ValueError("No active connections for performance degradation")
            source, destination = self.rng.choice(up_connections)
        
        if latency_multiplier is None:
            min_mult, max_mult = self.latency_multiplier_range
            latency_multiplier = self.rng.uniform(min_mult, max_mult)
        
        # Get current latency and increase it
        current_info = self.network.get_connection_info(source, destination)
        original_latency = current_info.get('latency', 1.0)
        new_latency = original_latency * latency_multiplier
        
        self.network.set_connection_latency(source, destination, new_latency)
        
        # If bidirectional, also increase reverse direction
        if self.network.graph.has_edge(destination, source):
            self.network.set_connection_latency(destination, source, new_latency)
        
        fault_info = FaultInfo(
            FaultType.PERFORMANCE_DEGRADATION,
            f"{source}->{destination}",
            {
                "source": source,
                "destination": destination,
                "original_latency": original_latency,
                "new_latency": new_latency,
                "multiplier": latency_multiplier
            }
        )
        self.active_faults.append(fault_info)
        
        return fault_info

    def _select_fault_type(self, fault_types: List[FaultType]) -> FaultType:
        if not fault_types:
            raise ValueError("No fault types available for injection")

        if self.sampling_strategy == FaultSamplingStrategy.UNIFORM:
            return self.rng.choice(fault_types)

        if self.sampling_strategy == FaultSamplingStrategy.WEIGHTED:
            weights = [self.fault_weights.get(ft, 1.0) for ft in fault_types]
            if sum(weights) <= 0:
                weights = [1.0] * len(fault_types)
            return self.rng.choices(fault_types, weights=weights, k=1)[0]

        if self.sampling_strategy == FaultSamplingStrategy.ROUND_ROBIN:
            idx = self._round_robin_index % len(fault_types)
            self._round_robin_index += 1
            return fault_types[idx]

        if self.sampling_strategy == FaultSamplingStrategy.STRATIFIED:
            return self._select_stratified_fault(fault_types)

        return self.rng.choice(fault_types)

    def _select_stratified_fault(self, fault_types: List[FaultType]) -> FaultType:
        """Select a fault type based on network size buckets."""
        size = len(self.network.get_all_devices()) if self.network else 0
        if size <= 5:
            preferred = [FaultType.DEVICE_FAILURE, FaultType.MISCONFIGURATION]
        elif size <= 10:
            preferred = [FaultType.LINK_FAILURE, FaultType.MISCONFIGURATION]
        else:
            preferred = [FaultType.LINK_FAILURE, FaultType.PERFORMANCE_DEGRADATION]

        candidates = [ft for ft in fault_types if ft in preferred]
        return self.rng.choice(candidates or fault_types)

    @staticmethod
    def _normalize_fault_weights(
        fault_weights: Optional[Dict[Any, float]]
    ) -> Dict[FaultType, float]:
        if not fault_weights:
            return {}
        normalized: Dict[FaultType, float] = {}
        for key, value in fault_weights.items():
            fault_type = key if isinstance(key, FaultType) else FaultType(str(key))
            normalized[fault_type] = float(value)
        return normalized
    
    def clear_all_faults(self) -> None:
        """Clear all active faults and restore network to healthy state."""
        for fault in self.active_faults:
            self._restore_fault(fault)
        self.active_faults.clear()
    
    def clear_fault(self, fault_info: FaultInfo) -> bool:
        """
        Clear a specific fault.
        
        Args:
            fault_info: Fault to clear
            
        Returns:
            True if fault was cleared, False if not found
        """
        if fault_info in self.active_faults:
            self._restore_fault(fault_info)
            self.active_faults.remove(fault_info)
            return True
        return False
    
    def _restore_fault(self, fault_info: FaultInfo) -> None:
        """Restore network state for a specific fault."""
        if fault_info.fault_type == FaultType.LINK_FAILURE:
            source = fault_info.details["source"]
            destination = fault_info.details["destination"]
            self.network.set_connection_status(source, destination, 'up')
            if self.network.graph.has_edge(destination, source):
                self.network.set_connection_status(destination, source, 'up')
                
        elif fault_info.fault_type == FaultType.DEVICE_FAILURE:
            device_id = fault_info.details["device_id"]
            self.network.set_device_status(device_id, 'up')
            
        elif fault_info.fault_type == FaultType.MISCONFIGURATION:
            device_id = fault_info.details["device_id"]
            blocked_destination = fault_info.details["blocked_destination"]
            self.network.set_connection_status(device_id, blocked_destination, 'up')
            
        elif fault_info.fault_type == FaultType.PERFORMANCE_DEGRADATION:
            source = fault_info.details["source"]
            destination = fault_info.details["destination"]
            original_latency = fault_info.details["original_latency"]
            self.network.set_connection_latency(source, destination, original_latency)
            if self.network.graph.has_edge(destination, source):
                self.network.set_connection_latency(destination, source, original_latency)
    
    def get_active_faults(self) -> List[FaultInfo]:
        """Get list of currently active faults."""
        return self.active_faults.copy()
    
    def has_active_faults(self) -> bool:
        """Check if there are any active faults."""
        return len(self.active_faults) > 0
