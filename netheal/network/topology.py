"""
Network topology generation utilities.

This module provides functions to generate various network topologies
for training and testing purposes.
"""

import random
from typing import List, Tuple, Dict, Any
from .graph import NetworkGraph, DeviceType


class TopologyGenerator:
    """Generator for various network topologies."""
    
    @staticmethod
    def generate_linear_topology(num_devices: int, device_types: List[DeviceType] = None) -> NetworkGraph:
        """
        Generate a linear topology (chain of devices).
        
        Args:
            num_devices: Number of devices in the chain
            device_types: List of device types to use (random if None)
            
        Returns:
            NetworkGraph with linear topology
        """
        if num_devices < 2:
            raise ValueError("Linear topology requires at least 2 devices")
        
        if device_types is None:
            device_types = [DeviceType.ROUTER, DeviceType.SWITCH, DeviceType.HOST]
        
        network = NetworkGraph()
        
        # Add devices
        for i in range(num_devices):
            device_id = f"device_{i}"
            device_type = random.choice(device_types)
            network.add_device(device_id, device_type)
        
        # Add connections in chain
        for i in range(num_devices - 1):
            source = f"device_{i}"
            dest = f"device_{i + 1}"
            bandwidth = random.uniform(10, 1000)  # 10-1000 Mbps
            latency = random.uniform(0.5, 5.0)    # 0.5-5ms
            network.add_connection(source, dest, bandwidth=bandwidth, latency=latency)
        
        return network
    
    @staticmethod
    def generate_star_topology(num_edge_devices: int, center_type: DeviceType = DeviceType.ROUTER) -> NetworkGraph:
        """
        Generate a star topology with central device.
        
        Args:
            num_edge_devices: Number of edge devices
            center_type: Type of central device
            
        Returns:
            NetworkGraph with star topology
        """
        if num_edge_devices < 1:
            raise ValueError("Star topology requires at least 1 edge device")
        
        network = NetworkGraph()
        
        # Add central device
        network.add_device("center", center_type)
        
        # Add edge devices and connect to center
        edge_types = [DeviceType.HOST, DeviceType.SERVER, DeviceType.SWITCH]
        
        for i in range(num_edge_devices):
            device_id = f"edge_{i}"
            device_type = random.choice(edge_types)
            network.add_device(device_id, device_type)
            
            bandwidth = random.uniform(10, 1000)
            latency = random.uniform(0.5, 5.0)
            network.add_connection("center", device_id, bandwidth=bandwidth, latency=latency)
        
        return network
    
    @staticmethod
    def generate_mesh_topology(num_devices: int, connection_probability: float = 0.3) -> NetworkGraph:
        """
        Generate a partial mesh topology.
        
        Args:
            num_devices: Number of devices
            connection_probability: Probability of connection between any two devices
            
        Returns:
            NetworkGraph with mesh topology
        """
        if num_devices < 2:
            raise ValueError("Mesh topology requires at least 2 devices")
        
        network = NetworkGraph()
        device_types = list(DeviceType)
        
        # Add devices
        for i in range(num_devices):
            device_id = f"device_{i}"
            device_type = random.choice(device_types)
            network.add_device(device_id, device_type)
        
        # Add random connections
        devices = [f"device_{i}" for i in range(num_devices)]
        for i in range(num_devices):
            for j in range(i + 1, num_devices):
                if random.random() < connection_probability:
                    bandwidth = random.uniform(10, 1000)
                    latency = random.uniform(0.5, 5.0)
                    network.add_connection(devices[i], devices[j], 
                                         bandwidth=bandwidth, latency=latency)
        
        # Ensure connectivity by adding minimum spanning connections
        TopologyGenerator._ensure_connectivity(network, devices)
        
        return network
    
    @staticmethod
    def generate_hierarchical_topology(num_layers: int = 3, devices_per_layer: List[int] = None) -> NetworkGraph:
        """
        Generate a hierarchical topology (e.g., enterprise network).
        
        Args:
            num_layers: Number of hierarchical layers
            devices_per_layer: Number of devices in each layer
            
        Returns:
            NetworkGraph with hierarchical topology
        """
        if num_layers < 2:
            raise ValueError("Hierarchical topology requires at least 2 layers")
        
        if devices_per_layer is None:
            devices_per_layer = [2] * num_layers
        
        if len(devices_per_layer) != num_layers:
            raise ValueError("devices_per_layer length must match num_layers")
        
        network = NetworkGraph()
        
        # Device types for each layer (top to bottom)
        layer_types = [
            [DeviceType.ROUTER],                    # Core layer
            [DeviceType.ROUTER, DeviceType.SWITCH], # Distribution layer  
            [DeviceType.SWITCH, DeviceType.HOST]    # Access layer
        ]
        
        # Extend layer types if needed
        while len(layer_types) < num_layers:
            layer_types.append([DeviceType.SWITCH, DeviceType.HOST])
        
        layers = []
        
        # Create devices for each layer
        for layer_idx in range(num_layers):
            layer_devices = []
            for device_idx in range(devices_per_layer[layer_idx]):
                device_id = f"L{layer_idx}_D{device_idx}"
                device_type = random.choice(layer_types[layer_idx])
                network.add_device(device_id, device_type)
                layer_devices.append(device_id)
            layers.append(layer_devices)
        
        # Connect adjacent layers
        for layer_idx in range(num_layers - 1):
            current_layer = layers[layer_idx]
            next_layer = layers[layer_idx + 1]
            
            # Connect each device in current layer to devices in next layer
            for current_device in current_layer:
                # Connect to 1-3 devices in next layer
                num_connections = min(random.randint(1, 3), len(next_layer))
                connected_devices = random.sample(next_layer, num_connections)
                
                for next_device in connected_devices:
                    bandwidth = random.uniform(100, 1000)  # Higher bandwidth for backbone
                    latency = random.uniform(0.1, 2.0)     # Lower latency for backbone
                    network.add_connection(current_device, next_device,
                                         bandwidth=bandwidth, latency=latency)
        
        return network
    
    @staticmethod
    def generate_random_topology(num_devices: int, min_connections: int = 1, 
                                max_connections: int = 4) -> NetworkGraph:
        """
        Generate a random topology ensuring connectivity.
        
        Args:
            num_devices: Number of devices
            min_connections: Minimum connections per device
            max_connections: Maximum connections per device
            
        Returns:
            NetworkGraph with random topology
        """
        if num_devices < 2:
            raise ValueError("Random topology requires at least 2 devices")
        
        network = NetworkGraph()
        device_types = list(DeviceType)
        
        # Add devices
        devices = []
        for i in range(num_devices):
            device_id = f"device_{i}"
            device_type = random.choice(device_types)
            network.add_device(device_id, device_type)
            devices.append(device_id)
        
        # Add random connections
        for device in devices:
            current_connections = len(network.get_device_connections(device))
            target_connections = random.randint(min_connections, max_connections)
            
            while current_connections < target_connections:
                # Choose random target that's not already connected
                available_targets = [d for d in devices if d != device and 
                                   not network.graph.has_edge(device, d)]
                
                if not available_targets:
                    break
                
                target = random.choice(available_targets)
                bandwidth = random.uniform(10, 1000)
                latency = random.uniform(0.5, 5.0)
                network.add_connection(device, target, bandwidth=bandwidth, latency=latency)
                current_connections += 1
        
        # Ensure connectivity
        TopologyGenerator._ensure_connectivity(network, devices)
        
        return network
    
    @staticmethod
    def _ensure_connectivity(network: NetworkGraph, devices: List[str]) -> None:
        """Ensure all devices are connected in the network."""
        # Use union-find to check connectivity and add minimum connections
        parent = {device: device for device in devices}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False
        
        # Mark existing connections
        for source, dest in network.get_all_connections():
            union(source, dest)
        
        # Add connections to ensure connectivity
        components = {}
        for device in devices:
            root = find(device)
            if root not in components:
                components[root] = []
            components[root].append(device)
        
        # Connect components
        component_list = list(components.values())
        for i in range(len(component_list) - 1):
            device1 = random.choice(component_list[i])
            device2 = random.choice(component_list[i + 1])
            
            if not network.graph.has_edge(device1, device2):
                bandwidth = random.uniform(10, 1000)
                latency = random.uniform(0.5, 5.0)
                network.add_connection(device1, device2, bandwidth=bandwidth, latency=latency)
