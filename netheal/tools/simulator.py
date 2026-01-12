# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Diagnostic tools simulation for network troubleshooting.

This module provides the ToolSimulator class that simulates real-world network
diagnostic tools like ping, traceroute, etc., operating on NetworkGraph objects.
"""

import random
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from ..network.graph import NetworkGraph


@dataclass
class ToolResult:
    """Result of a diagnostic tool execution."""
    success: bool
    data: Dict[str, Any]
    cost: float
    tool_name: str
    
    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"{self.tool_name}: {status} (cost: {self.cost})"


class ToolSimulator:
    """
    Simulator for network diagnostic tools.
    
    This class provides methods that simulate real-world troubleshooting tools
    but operate on NetworkGraph objects. Each tool has an associated cost to
    encourage efficient troubleshooting.
    """
    
    def __init__(self, network: NetworkGraph, base_cost: float = 1.0):
        """
        Initialize tool simulator.
        
        Args:
            network: NetworkGraph to operate on
            base_cost: Base cost for tool operations
        """
        self.network = network
        self.base_cost = base_cost
        
        # Tool costs (relative to base_cost)
        self.tool_costs = {
            'ping': 1.0,
            'traceroute': 2.0,
            'check_status': 0.5,
            'check_interfaces': 1.5
        }
    
    def ping(self, source: str, destination: str) -> ToolResult:
        """
        Simulate ping command between two devices.
        
        Args:
            source: Source device ID
            destination: Destination device ID
            
        Returns:
            ToolResult with ping results
        """
        cost = self.base_cost * self.tool_costs['ping']
        
        # Check if devices exist
        if source not in self.network or destination not in self.network:
            return ToolResult(
                success=False,
                data={"error": "Device not found"},
                cost=cost,
                tool_name="ping"
            )
        
        # Check if source device is up
        if not self.network.is_device_up(source):
            return ToolResult(
                success=False,
                data={"error": f"Source device {source} is down"},
                cost=cost,
                tool_name="ping"
            )
        
        # Check if destination device is up
        if not self.network.is_device_up(destination):
            return ToolResult(
                success=False,
                data={"error": f"Destination device {destination} is unreachable"},
                cost=cost,
                tool_name="ping"
            )
        
        # Find path between devices
        path = self.network.find_path(source, destination)
        
        if path is None:
            # No path available
            return ToolResult(
                success=False,
                data={
                    "source": source,
                    "destination": destination,
                    "error": "No route to host",
                    "packets_transmitted": 4,
                    "packets_received": 0,
                    "packet_loss": "100%"
                },
                cost=cost,
                tool_name="ping"
            )
        
        # Calculate latency for successful ping
        latency = self.network.calculate_path_latency(path)
        
        # Add some random variation to simulate real network conditions
        jitter = random.uniform(-0.1, 0.1) * latency
        measured_latency = max(0.1, latency + jitter)
        
        return ToolResult(
            success=True,
            data={
                "source": source,
                "destination": destination,
                "latency_ms": round(measured_latency, 2),
                "packets_transmitted": 4,
                "packets_received": 4,
                "packet_loss": "0%"
            },
            cost=cost,
            tool_name="ping"
        )
    
    def traceroute(self, source: str, destination: str) -> ToolResult:
        """
        Simulate traceroute command between two devices.
        
        Args:
            source: Source device ID
            destination: Destination device ID
            
        Returns:
            ToolResult with traceroute results
        """
        cost = self.base_cost * self.tool_costs['traceroute']
        
        # Check if devices exist
        if source not in self.network or destination not in self.network:
            return ToolResult(
                success=False,
                data={"error": "Device not found"},
                cost=cost,
                tool_name="traceroute"
            )
        
        # Check if source device is up
        if not self.network.is_device_up(source):
            return ToolResult(
                success=False,
                data={"error": f"Source device {source} is down"},
                cost=cost,
                tool_name="traceroute"
            )
        
        # Find path between devices
        path = self.network.find_path(source, destination)
        
        if path is None or len(path) < 2:
            # Try to find partial path by checking each hop
            partial_path = [source]
            current = source
            
            # Check immediate neighbors
            connections = self.network.get_device_connections(current)
            up_connections = [(dest, info) for dest, info in connections 
                            if info.get('status') == 'up' and self.network.is_device_up(dest)]
            
            if up_connections:
                # Take first available connection for partial trace
                next_hop, _ = up_connections[0]
                partial_path.append(next_hop)
            
            return ToolResult(
                success=False,
                data={
                    "source": source,
                    "destination": destination,
                    "path": partial_path,
                    "hops": len(partial_path) - 1,
                    "error": "Network unreachable" if len(partial_path) == 1 else "Destination unreachable",
                    "failure_point": partial_path[-1] if len(partial_path) > 1 else source
                },
                cost=cost,
                tool_name="traceroute"
            )
        
        # Calculate hop-by-hop latencies
        hop_latencies = []
        cumulative_latency = 0.0
        
        for i in range(len(path) - 1):
            hop_source, hop_dest = path[i], path[i + 1]
            if self.network.graph.has_edge(hop_source, hop_dest):
                edge_data = self.network.get_connection_info(hop_source, hop_dest)
                hop_latency = edge_data.get('latency', 1.0)
                
                # Add jitter
                jitter = random.uniform(-0.05, 0.05) * hop_latency
                measured_hop_latency = max(0.1, hop_latency + jitter)
                cumulative_latency += measured_hop_latency
                
                hop_latencies.append({
                    "hop": i + 1,
                    "device": hop_dest,
                    "latency_ms": round(measured_hop_latency, 2),
                    "cumulative_latency_ms": round(cumulative_latency, 2)
                })
        
        return ToolResult(
            success=True,
            data={
                "source": source,
                "destination": destination,
                "path": path,
                "hops": len(path) - 1,
                "hop_details": hop_latencies,
                "total_latency_ms": round(cumulative_latency, 2)
            },
            cost=cost,
            tool_name="traceroute"
        )
    
    def check_status(self, device_id: str) -> ToolResult:
        """
        Check the status of a specific device.
        
        Args:
            device_id: Device ID to check
            
        Returns:
            ToolResult with device status
        """
        cost = self.base_cost * self.tool_costs['check_status']
        
        if device_id not in self.network:
            return ToolResult(
                success=False,
                data={"error": f"Device {device_id} not found"},
                cost=cost,
                tool_name="check_status"
            )
        
        device_info = self.network.get_device_info(device_id)
        
        return ToolResult(
            success=True,
            data={
                "device_id": device_id,
                "status": device_info.get('status', 'unknown'),
                "ip_address": device_info.get('ip_address', 'unknown'),
                "device_type": device_info.get('device_type', 'unknown').value if hasattr(device_info.get('device_type'), 'value') else str(device_info.get('device_type', 'unknown'))
            },
            cost=cost,
            tool_name="check_status"
        )
    
    def check_interfaces(self, device_id: str) -> ToolResult:
        """
        Check the status of all interfaces (connections) for a device.
        
        Args:
            device_id: Device ID to check interfaces for
            
        Returns:
            ToolResult with interface status information
        """
        cost = self.base_cost * self.tool_costs['check_interfaces']
        
        if device_id not in self.network:
            return ToolResult(
                success=False,
                data={"error": f"Device {device_id} not found"},
                cost=cost,
                tool_name="check_interfaces"
            )
        
        # Check if device is up
        if not self.network.is_device_up(device_id):
            return ToolResult(
                success=False,
                data={"error": f"Device {device_id} is down"},
                cost=cost,
                tool_name="check_interfaces"
            )
        
        connections = self.network.get_device_connections(device_id)
        interfaces = []
        
        for destination, connection_info in connections:
            interface_data = {
                "interface": f"eth-{destination}",
                "destination": destination,
                "status": connection_info.get('status', 'unknown'),
                "bandwidth_mbps": connection_info.get('bandwidth', 0),
                "latency_ms": connection_info.get('latency', 0)
            }
            interfaces.append(interface_data)
        
        return ToolResult(
            success=True,
            data={
                "device_id": device_id,
                "total_interfaces": len(interfaces),
                "interfaces": interfaces,
                "up_interfaces": len([i for i in interfaces if i['status'] == 'up']),
                "down_interfaces": len([i for i in interfaces if i['status'] == 'down'])
            },
            cost=cost,
            tool_name="check_interfaces"
        )
    
    def get_available_tools(self) -> List[str]:
        """Get list of available diagnostic tools."""
        return list(self.tool_costs.keys())
    
    def get_tool_cost(self, tool_name: str) -> float:
        """Get the cost of a specific tool."""
        return self.base_cost * self.tool_costs.get(tool_name, 1.0)
    
    def set_base_cost(self, base_cost: float) -> None:
        """Set the base cost for all tools."""
        self.base_cost = base_cost
