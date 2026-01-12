# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Network graph representation using NetworkX.

This module provides the core NetworkGraph class that represents network topology
as a directed graph with device nodes and connection edges.
"""

import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import ipaddress
import random


class DeviceType(Enum):
    """Network device types."""
    ROUTER = "router"
    SWITCH = "switch"
    SERVER = "server"
    FIREWALL = "firewall"
    HOST = "host"


class NetworkGraph:
    """
    Network representation using NetworkX directed graph.
    
    Nodes represent network devices with attributes:
    - status: 'up' or 'down'
    - ip_address: IP address string
    - device_type: DeviceType enum
    
    Edges represent connections with attributes:
    - status: 'up' or 'down'  
    - bandwidth: bandwidth in Mbps
    - latency: latency in milliseconds
    """
    
    def __init__(self):
        """Initialize empty network graph."""
        self.graph = nx.DiGraph()
        self._ip_counter = 1
        
    def add_device(self, device_id: str, device_type: DeviceType, 
                   ip_address: Optional[str] = None, status: str = 'up') -> None:
        """
        Add a network device to the graph.
        
        Args:
            device_id: Unique identifier for the device
            device_type: Type of network device
            ip_address: IP address (auto-generated if None)
            status: Device status ('up' or 'down')
        """
        if ip_address is None:
            ip_address = f"192.168.1.{self._ip_counter}"
            self._ip_counter += 1
            
        self.graph.add_node(device_id, 
                           status=status,
                           ip_address=ip_address,
                           device_type=device_type)
    
    def add_connection(self, source: str, destination: str,
                      bandwidth: float = 100.0, latency: float = 1.0,
                      status: str = 'up', bidirectional: bool = True) -> None:
        """
        Add a connection between two devices.
        
        Args:
            source: Source device ID
            destination: Destination device ID
            bandwidth: Connection bandwidth in Mbps
            latency: Connection latency in milliseconds
            status: Connection status ('up' or 'down')
            bidirectional: If True, add connection in both directions
        """
        edge_attrs = {
            'status': status,
            'bandwidth': bandwidth,
            'latency': latency
        }
        
        self.graph.add_edge(source, destination, **edge_attrs)
        
        if bidirectional:
            self.graph.add_edge(destination, source, **edge_attrs)
    
    def get_device_info(self, device_id: str) -> Dict[str, Any]:
        """Get device information."""
        if device_id not in self.graph.nodes:
            raise ValueError(f"Device {device_id} not found")
        return dict(self.graph.nodes[device_id])
    
    def get_connection_info(self, source: str, destination: str) -> Dict[str, Any]:
        """Get connection information."""
        if not self.graph.has_edge(source, destination):
            raise ValueError(f"Connection {source}->{destination} not found")
        return dict(self.graph.edges[source, destination])
    
    def get_device_connections(self, device_id: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Get all connections from a device."""
        if device_id not in self.graph.nodes:
            raise ValueError(f"Device {device_id} not found")
        
        connections = []
        for neighbor in self.graph.neighbors(device_id):
            edge_data = dict(self.graph.edges[device_id, neighbor])
            connections.append((neighbor, edge_data))
        
        return connections
    
    def find_path(self, source: str, destination: str) -> Optional[List[str]]:
        """
        Find shortest path between two devices considering only 'up' connections.
        
        Args:
            source: Source device ID
            destination: Destination device ID
            
        Returns:
            List of device IDs representing the path, or None if no path exists
        """
        # Check if source and destination exist and are up
        if source not in self.graph.nodes or destination not in self.graph.nodes:
            return None
        
        if not self.is_device_up(source) or not self.is_device_up(destination):
            return None
        
        # Create subgraph with only 'up' nodes and edges
        up_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('status') == 'up']
        
        # Create a new graph with only up nodes and up edges
        subgraph = nx.DiGraph()
        subgraph.add_nodes_from(up_nodes)
        
        for u, v, d in self.graph.edges(data=True):
            if (d.get('status') == 'up' and 
                u in up_nodes and v in up_nodes):
                subgraph.add_edge(u, v)
        
        try:
            return nx.shortest_path(subgraph, source, destination)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def calculate_path_latency(self, path: List[str]) -> float:
        """Calculate total latency for a given path."""
        if len(path) < 2:
            return 0.0
            
        total_latency = 0.0
        for i in range(len(path) - 1):
            source, dest = path[i], path[i + 1]
            if self.graph.has_edge(source, dest):
                edge_data = self.graph.edges[source, dest]
                total_latency += edge_data.get('latency', 0.0)
        
        return total_latency
    
    def get_all_devices(self) -> List[str]:
        """Get list of all device IDs."""
        return list(self.graph.nodes())
    
    def get_all_connections(self) -> List[Tuple[str, str]]:
        """Get list of all connections as (source, destination) tuples."""
        return list(self.graph.edges())
    
    def is_device_up(self, device_id: str) -> bool:
        """Check if device is up."""
        return self.graph.nodes[device_id].get('status') == 'up'
    
    def is_connection_up(self, source: str, destination: str) -> bool:
        """Check if connection is up."""
        if not self.graph.has_edge(source, destination):
            return False
        return self.graph.edges[source, destination].get('status') == 'up'
    
    def set_device_status(self, device_id: str, status: str) -> None:
        """Set device status."""
        if device_id in self.graph.nodes:
            self.graph.nodes[device_id]['status'] = status
    
    def set_connection_status(self, source: str, destination: str, status: str) -> None:
        """Set connection status."""
        if self.graph.has_edge(source, destination):
            self.graph.edges[source, destination]['status'] = status
    
    def set_connection_latency(self, source: str, destination: str, latency: float) -> None:
        """Set connection latency."""
        if self.graph.has_edge(source, destination):
            self.graph.edges[source, destination]['latency'] = latency
    
    def copy(self) -> 'NetworkGraph':
        """Create a deep copy of the network graph."""
        new_graph = NetworkGraph()
        new_graph.graph = self.graph.copy()
        new_graph._ip_counter = self._ip_counter
        return new_graph
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert network graph to dictionary representation."""
        return {
            'nodes': dict(self.graph.nodes(data=True)),
            'edges': {f"{u}-{v}": data for u, v, data in self.graph.edges(data=True)}
        }
    
    def __len__(self) -> int:
        """Return number of devices in the network."""
        return len(self.graph.nodes())
    
    def __contains__(self, device_id: str) -> bool:
        """Check if device exists in the network."""
        return device_id in self.graph.nodes()
