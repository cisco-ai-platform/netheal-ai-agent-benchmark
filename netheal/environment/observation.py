"""
Graph-aware observation space for network troubleshooting RL environment.

This module provides structured observation components that enable agents to build
understanding of network topology and maintain diagnostic results memory.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import time

from ..network.graph import NetworkGraph, DeviceType
from ..tools.simulator import ToolResult


class ConnectionStatus(Enum):
    """Status of network connections from agent's perspective."""
    UNKNOWN = 0
    CONNECTED = 1
    DISCONNECTED = -1
    DEGRADED = 2


class DeviceStatus(Enum):
    """Status of network devices from agent's perspective."""
    UNKNOWN = 0
    UP = 1
    DOWN = -1
    DEGRADED = 2


@dataclass
class DiagnosticResult:
    """Single diagnostic result with metadata."""
    tool_name: str
    source: Optional[str]
    destination: Optional[str]
    result: ToolResult
    timestamp: float
    confidence: float = 1.0
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()




class NetworkDiscoveryMatrix:
    """Tracks discovered network topology from agent's perspective."""
    
    def __init__(self, max_devices: int):
        """Initialize discovery matrix."""
        self.max_devices = max_devices
        self.device_map = {}  # device_id -> index mapping
        self.reverse_map = {}  # index -> device_id mapping
        self.adjacency = np.zeros((max_devices, max_devices), dtype=np.int8)
        self.connection_properties = {}  # (i,j) -> properties dict
        self.next_index = 0
    
    def add_device(self, device_id: str) -> int:
        """Add device to discovery matrix, return its index."""
        if device_id not in self.device_map:
            if self.next_index >= self.max_devices:
                raise ValueError(f"Maximum devices ({self.max_devices}) exceeded")
            
            idx = self.next_index
            self.device_map[device_id] = idx
            self.reverse_map[idx] = device_id
            self.next_index += 1
            return idx
        return self.device_map[device_id]
    
    def update_connection(self, source: str, dest: str, status: ConnectionStatus, 
                         properties: Dict[str, Any] = None):
        """Update connection status between devices."""
        src_idx = self.add_device(source)
        dst_idx = self.add_device(dest)
        
        self.adjacency[src_idx, dst_idx] = status.value
        
        if properties:
            self.connection_properties[(src_idx, dst_idx)] = properties
    
    def get_connection_status(self, source: str, dest: str) -> ConnectionStatus:
        """Get connection status between devices."""
        if source not in self.device_map or dest not in self.device_map:
            return ConnectionStatus.UNKNOWN
        
        src_idx = self.device_map[source]
        dst_idx = self.device_map[dest]
        return ConnectionStatus(self.adjacency[src_idx, dst_idx])
    
    def get_discovered_devices(self) -> List[str]:
        """Get list of discovered device IDs."""
        return [self.reverse_map[i] for i in range(self.next_index)]
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Get the current adjacency matrix."""
        return self.adjacency[:self.next_index, :self.next_index].copy()
    
    def get_neighbors(self, device_id: str) -> List[str]:
        """Get discovered neighbors of a device."""
        if device_id not in self.device_map:
            return []
        
        idx = self.device_map[device_id]
        neighbors = []
        
        for i in range(self.next_index):
            if self.adjacency[idx, i] == ConnectionStatus.CONNECTED.value:
                neighbors.append(self.reverse_map[i])
        
        return neighbors


class DeviceStatusTable:
    """Tracks discovered device properties and status."""
    
    def __init__(self, max_devices: int):
        """Initialize device status table."""
        self.max_devices = max_devices
        self.devices = {}  # device_id -> properties dict
    
    def update_device(self, device_id: str, status: DeviceStatus = None,
                     device_type: DeviceType = None, ip_address: str = None,
                     interfaces: Dict[str, Any] = None, **kwargs):
        """Update device information."""
        if device_id not in self.devices:
            self.devices[device_id] = {
                'status': DeviceStatus.UNKNOWN,
                'device_type': None,
                'ip_address': None,
                'interfaces': {},
                'last_updated': time.time()
            }
        
        device_info = self.devices[device_id]
        
        if status is not None:
            device_info['status'] = status
        if device_type is not None:
            device_info['device_type'] = device_type
        if ip_address is not None:
            device_info['ip_address'] = ip_address
        if interfaces is not None:
            device_info['interfaces'].update(interfaces)
        
        device_info.update(kwargs)
        device_info['last_updated'] = time.time()
    
    def get_device_status(self, device_id: str) -> DeviceStatus:
        """Get device status."""
        return self.devices.get(device_id, {}).get('status', DeviceStatus.UNKNOWN)
    
    def get_device_info(self, device_id: str) -> Dict[str, Any]:
        """Get all device information."""
        return self.devices.get(device_id, {}).copy()
    
    def get_all_devices(self) -> List[str]:
        """Get all discovered device IDs."""
        return list(self.devices.keys())


class DiagnosticResultsMemory:
    """Maintains memory of diagnostic results and their implications."""
    
    def __init__(self, max_history: int = 100):
        """Initialize diagnostic results memory."""
        self.max_history = max_history
        self.results = []  # List of DiagnosticResult
        self.results_by_device = {}  # device_id -> List[DiagnosticResult]
        self.results_by_connection = {}  # (src, dst) -> List[DiagnosticResult]
    
    def add_result(self, result: DiagnosticResult):
        """Add a diagnostic result to memory."""
        self.results.append(result)
        
        # Maintain max history
        if len(self.results) > self.max_history:
            old_result = self.results.pop(0)
            self._remove_from_indices(old_result)
        
        # Index by device
        if result.source:
            if result.source not in self.results_by_device:
                self.results_by_device[result.source] = []
            self.results_by_device[result.source].append(result)
        
        if result.destination and result.destination != result.source:
            if result.destination not in self.results_by_device:
                self.results_by_device[result.destination] = []
            self.results_by_device[result.destination].append(result)
        
        # Index by connection
        if result.source and result.destination:
            conn_key = (result.source, result.destination)
            if conn_key not in self.results_by_connection:
                self.results_by_connection[conn_key] = []
            self.results_by_connection[conn_key].append(result)
    
    def _remove_from_indices(self, result: DiagnosticResult):
        """Remove result from device and connection indices."""
        if result.source and result.source in self.results_by_device:
            if result in self.results_by_device[result.source]:
                self.results_by_device[result.source].remove(result)
        
        if result.destination and result.destination in self.results_by_device:
            if result in self.results_by_device[result.destination]:
                self.results_by_device[result.destination].remove(result)
        
        if result.source and result.destination:
            conn_key = (result.source, result.destination)
            if conn_key in self.results_by_connection:
                if result in self.results_by_connection[conn_key]:
                    self.results_by_connection[conn_key].remove(result)
    
    def get_recent_results(self, count: int = 10) -> List[DiagnosticResult]:
        """Get most recent diagnostic results."""
        return self.results[-count:]
    
    def get_results_for_device(self, device_id: str) -> List[DiagnosticResult]:
        """Get all results involving a specific device."""
        return self.results_by_device.get(device_id, []).copy()
    
    def get_results_for_connection(self, source: str, dest: str) -> List[DiagnosticResult]:
        """Get all results for a specific connection."""
        return self.results_by_connection.get((source, dest), []).copy()
    
    def get_successful_pings(self) -> List[DiagnosticResult]:
        """Get all successful ping results."""
        return [r for r in self.results 
                if r.tool_name == 'ping' and r.result.success]
    
    def get_failed_connections(self) -> List[Tuple[str, str]]:
        """Get connections that have failed diagnostic tests."""
        failed = set()
        for result in self.results:
            if not result.result.success and result.source and result.destination:
                failed.add((result.source, result.destination))
        return list(failed)




class StructuredObservation:
    """Main observation class that combines all components."""
    
    def __init__(self, max_devices: int = 10, max_history: int = 100):
        """Initialize structured observation."""
        self.max_devices = max_devices
        self.discovery_matrix = NetworkDiscoveryMatrix(max_devices)
        self.device_status = DeviceStatusTable(max_devices)
        self.diagnostic_memory = DiagnosticResultsMemory(max_history)
        
        # Episode metadata
        self.episode_step = 0
        self.max_episode_steps = 20
        self.episode_start_time = time.time()
    
    def update_from_diagnostic_result(self, result: DiagnosticResult):
        """Update observation based on diagnostic tool result."""
        self.diagnostic_memory.add_result(result)
        
        # Update network discovery based on tool results
        if result.tool_name == 'ping':
            self._update_from_ping(result)
        elif result.tool_name == 'traceroute':
            self._update_from_traceroute(result)
        elif result.tool_name == 'check_status':
            self._update_from_status_check(result)
        elif result.tool_name == 'check_interfaces':
            self._update_from_interface_check(result)
    
    def _update_from_ping(self, result: DiagnosticResult):
        """Update observation from ping result."""
        source, dest = result.source, result.destination
        
        if result.result.success:
            # Successful ping indicates connectivity
            self.discovery_matrix.update_connection(
                source, dest, ConnectionStatus.CONNECTED,
                {'latency': result.result.data.get('latency_ms', 0)}
            )
            # Update device status
            self.device_status.update_device(source, DeviceStatus.UP)
            self.device_status.update_device(dest, DeviceStatus.UP)
        else:
            # Failed ping - could be device down or connection issue
            if 'device' in result.result.data.get('error', '').lower():
                self.device_status.update_device(dest, DeviceStatus.DOWN)
            else:
                self.discovery_matrix.update_connection(
                    source, dest, ConnectionStatus.DISCONNECTED
                )
    
    def _update_from_traceroute(self, result: DiagnosticResult):
        """Update observation from traceroute result."""
        if result.result.success and 'path' in result.result.data:
            path = result.result.data['path']
            # Update connections along the path
            for i in range(len(path) - 1):
                self.discovery_matrix.update_connection(
                    path[i], path[i+1], ConnectionStatus.CONNECTED
                )
                self.device_status.update_device(path[i], DeviceStatus.UP)
            
            if path:
                self.device_status.update_device(path[-1], DeviceStatus.UP)
    
    def _update_from_status_check(self, result: DiagnosticResult):
        """Update observation from device status check."""
        device_id = result.source or result.destination
        if device_id and result.result.success:
            data = result.result.data
            status = DeviceStatus.UP if data.get('status') == 'up' else DeviceStatus.DOWN
            
            device_type = None
            if 'device_type' in data:
                try:
                    device_type = DeviceType(data['device_type'])
                except ValueError:
                    pass
            
            self.device_status.update_device(
                device_id, 
                status=status,
                device_type=device_type,
                ip_address=data.get('ip_address')
            )
    
    def _update_from_interface_check(self, result: DiagnosticResult):
        """Update observation from interface check."""
        device_id = result.source or result.destination
        if device_id and result.result.success:
            interfaces = result.result.data.get('interfaces', [])
            interface_dict = {}
            
            for iface in interfaces:
                dest_device = iface.get('destination')
                if dest_device:
                    interface_dict[dest_device] = iface
                    
                    # Update connection status
                    if iface.get('status') == 'up':
                        self.discovery_matrix.update_connection(
                            device_id, dest_device, ConnectionStatus.CONNECTED,
                            {'latency': iface.get('latency_ms', 0),
                             'bandwidth': iface.get('bandwidth_mbps', 0)}
                        )
                    else:
                        self.discovery_matrix.update_connection(
                            device_id, dest_device, ConnectionStatus.DISCONNECTED
                        )
            
            self.device_status.update_device(device_id, interfaces=interface_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert observation to dictionary format compatible with Gymnasium spaces."""
        import numpy as np
        
        # Convert discovery matrix to numpy array
        discovery_matrix = self.discovery_matrix.adjacency.astype(np.int8)
        
        # Convert device status to fixed-size array
        device_status = np.zeros((self.max_devices, 10), dtype=np.float32)
        for i, (device_id, info) in enumerate(self.device_status.devices.items()):
            if i < self.max_devices:
                # Encode device status as features
                device_status[i, 0] = 1.0 if info.get('status') == 'active' else 0.0
                device_status[i, 1] = hash(info.get('device_type', '')) % 100 / 100.0  # Normalized hash
                device_status[i, 2] = len(info.get('interfaces', {})) / 10.0  # Normalized interface count
                device_status[i, 3] = min(info.get('last_updated', 0) / 1000.0, 1.0)  # Normalized timestamp
        
        # Convert recent diagnostics to fixed-size array
        recent_diagnostics = np.zeros((10, 6), dtype=np.float32)
        for i, result in enumerate(self.diagnostic_memory.get_recent_results(10)):
            if i < 10:
                recent_diagnostics[i, 0] = hash(result.tool_name) % 100 / 100.0  # Tool type
                recent_diagnostics[i, 1] = hash(result.source) % 100 / 100.0  # Source hash
                recent_diagnostics[i, 2] = hash(result.destination) % 100 / 100.0  # Dest hash
                recent_diagnostics[i, 3] = 1.0 if result.result.success else 0.0  # Success
                recent_diagnostics[i, 4] = min(result.timestamp / 1000.0, 1.0)  # Normalized timestamp
                recent_diagnostics[i, 5] = result.confidence  # Confidence
        
        
        # Convert episode metadata to fixed-size array
        episode_metadata = np.array([
            self.episode_step / max(self.max_episode_steps, 1),  # Normalized step
            self.episode_step / max(self.max_episode_steps, 1),  # Progress (same as step)
            len(self.discovery_matrix.get_discovered_devices()) / max(self.max_devices, 1),  # Normalized discovered devices
            min(len(self.diagnostic_memory.results) / 100.0, 1.0)  # Normalized diagnostic actions
        ], dtype=np.float32)
        
        return {
            'discovery_matrix': discovery_matrix,
            'device_status': device_status,
            'recent_diagnostics': recent_diagnostics,
            'episode_metadata': episode_metadata
        }
