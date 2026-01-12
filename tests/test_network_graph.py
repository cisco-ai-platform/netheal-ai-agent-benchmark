# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for NetworkGraph class."""

import pytest
from netheal.network.graph import NetworkGraph, DeviceType


class TestNetworkGraph:
    """Test cases for NetworkGraph class."""
    
    def test_create_empty_graph(self):
        """Test creating an empty network graph."""
        network = NetworkGraph()
        assert len(network) == 0
        assert len(network.get_all_devices()) == 0
        assert len(network.get_all_connections()) == 0
    
    def test_add_device(self):
        """Test adding devices to the network."""
        network = NetworkGraph()
        
        # Add a router
        network.add_device("router1", DeviceType.ROUTER)
        assert len(network) == 1
        assert "router1" in network
        
        device_info = network.get_device_info("router1")
        assert device_info["status"] == "up"
        assert device_info["device_type"] == DeviceType.ROUTER
        assert "192.168.1." in device_info["ip_address"]
    
    def test_add_connection(self):
        """Test adding connections between devices."""
        network = NetworkGraph()
        
        # Add two devices
        network.add_device("device1", DeviceType.ROUTER)
        network.add_device("device2", DeviceType.SWITCH)
        
        # Add bidirectional connection
        network.add_connection("device1", "device2", bandwidth=100.0, latency=1.0)
        
        connections = network.get_all_connections()
        assert len(connections) == 2  # Bidirectional
        
        # Check connection info
        conn_info = network.get_connection_info("device1", "device2")
        assert conn_info["status"] == "up"
        assert conn_info["bandwidth"] == 100.0
        assert conn_info["latency"] == 1.0
    
    def test_find_path(self):
        """Test path finding between devices."""
        network = NetworkGraph()
        
        # Create linear topology: A -> B -> C
        network.add_device("A", DeviceType.ROUTER)
        network.add_device("B", DeviceType.SWITCH)
        network.add_device("C", DeviceType.HOST)
        
        network.add_connection("A", "B")
        network.add_connection("B", "C")
        
        # Test path finding
        path = network.find_path("A", "C")
        assert path == ["A", "B", "C"]
        
        # Test reverse path
        path_reverse = network.find_path("C", "A")
        assert path_reverse == ["C", "B", "A"]
    
    def test_path_with_failed_device(self):
        """Test path finding with failed device."""
        network = NetworkGraph()
        
        # Create linear topology: A -> B -> C
        network.add_device("A", DeviceType.ROUTER)
        network.add_device("B", DeviceType.SWITCH)
        network.add_device("C", DeviceType.HOST)
        
        network.add_connection("A", "B")
        network.add_connection("B", "C")
        
        # Fail middle device
        network.set_device_status("B", "down")
        
        # Path should not be found
        path = network.find_path("A", "C")
        assert path is None
    
    def test_path_with_failed_connection(self):
        """Test path finding with failed connection."""
        network = NetworkGraph()
        
        # Create linear topology: A -> B -> C
        network.add_device("A", DeviceType.ROUTER)
        network.add_device("B", DeviceType.SWITCH)
        network.add_device("C", DeviceType.HOST)
        
        network.add_connection("A", "B")
        network.add_connection("B", "C")
        
        # Fail connection
        network.set_connection_status("B", "C", "down")
        
        # Path should not be found
        path = network.find_path("A", "C")
        assert path is None
    
    def test_calculate_path_latency(self):
        """Test path latency calculation."""
        network = NetworkGraph()
        
        # Create path with known latencies
        network.add_device("A", DeviceType.ROUTER)
        network.add_device("B", DeviceType.SWITCH)
        network.add_device("C", DeviceType.HOST)
        
        network.add_connection("A", "B", latency=2.0)
        network.add_connection("B", "C", latency=3.0)
        
        path = ["A", "B", "C"]
        total_latency = network.calculate_path_latency(path)
        assert total_latency == 5.0
    
    def test_device_connections(self):
        """Test getting device connections."""
        network = NetworkGraph()
        
        network.add_device("router", DeviceType.ROUTER)
        network.add_device("switch1", DeviceType.SWITCH)
        network.add_device("switch2", DeviceType.SWITCH)
        
        network.add_connection("router", "switch1")
        network.add_connection("router", "switch2")
        
        connections = network.get_device_connections("router")
        assert len(connections) == 2
        
        connected_devices = [conn[0] for conn in connections]
        assert "switch1" in connected_devices
        assert "switch2" in connected_devices
    
    def test_copy_network(self):
        """Test copying network graph."""
        network = NetworkGraph()
        
        network.add_device("device1", DeviceType.ROUTER)
        network.add_device("device2", DeviceType.SWITCH)
        network.add_connection("device1", "device2")
        
        # Copy network
        network_copy = network.copy()
        
        # Verify copy
        assert len(network_copy) == len(network)
        assert network_copy.get_all_devices() == network.get_all_devices()
        assert network_copy.get_all_connections() == network.get_all_connections()
        
        # Verify independence
        network_copy.add_device("device3", DeviceType.HOST)
        assert len(network_copy) != len(network)
    
    def test_invalid_operations(self):
        """Test invalid operations raise appropriate errors."""
        network = NetworkGraph()
        
        # Test getting info for non-existent device
        with pytest.raises(ValueError):
            network.get_device_info("nonexistent")
        
        # Test getting connection info for non-existent connection
        network.add_device("device1", DeviceType.ROUTER)
        with pytest.raises(ValueError):
            network.get_connection_info("device1", "nonexistent")
