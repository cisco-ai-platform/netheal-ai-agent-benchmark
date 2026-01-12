"""Unit tests for FaultInjector class."""

import pytest
from netheal.network.graph import NetworkGraph, DeviceType
from netheal.network.topology import TopologyGenerator
from netheal.faults.injector import FaultInjector, FaultType


class TestFaultInjector:
    """Test cases for FaultInjector class."""
    
    def setup_method(self):
        """Set up test network for each test."""
        self.network = TopologyGenerator.generate_linear_topology(4)
        self.fault_injector = FaultInjector(self.network)
    
    def test_inject_link_failure(self):
        """Test link failure injection."""
        # Get initial connections
        initial_connections = self.network.get_all_connections()
        up_connections = [(s, d) for s, d in initial_connections 
                         if self.network.is_connection_up(s, d)]
        
        assert len(up_connections) > 0, "Should have some up connections"
        
        # Inject link failure
        fault = self.fault_injector.inject_link_failure()
        
        assert fault.fault_type == FaultType.LINK_FAILURE
        assert fault.location in [f"{s}->{d}" for s, d in initial_connections]
        
        # Verify fault is active
        assert len(self.fault_injector.get_active_faults()) == 1
        assert self.fault_injector.has_active_faults()
    
    def test_inject_device_failure(self):
        """Test device failure injection."""
        # Get initial devices
        devices = self.network.get_all_devices()
        up_devices = [d for d in devices if self.network.is_device_up(d)]
        
        assert len(up_devices) > 0, "Should have some up devices"
        
        # Inject device failure
        fault = self.fault_injector.inject_device_failure()
        
        assert fault.fault_type == FaultType.DEVICE_FAILURE
        assert fault.location in devices
        
        # Verify device is down
        failed_device = fault.details["device_id"]
        assert not self.network.is_device_up(failed_device)
    
    def test_inject_misconfiguration(self):
        """Test misconfiguration injection."""
        fault = self.fault_injector.inject_misconfiguration()
        
        assert fault.fault_type == FaultType.MISCONFIGURATION
        assert "blocked_destination" in fault.details
        
        # Verify connection is blocked
        device_id = fault.details["device_id"]
        blocked_dest = fault.details["blocked_destination"]
        assert not self.network.is_connection_up(device_id, blocked_dest)
    
    def test_inject_performance_degradation(self):
        """Test performance degradation injection."""
        # Get initial connection latency
        connections = self.network.get_all_connections()
        if connections:
            source, dest = connections[0]
            initial_latency = self.network.get_connection_info(source, dest)["latency"]
            
            # Inject performance degradation
            fault = self.fault_injector.inject_performance_degradation(source, dest)
            
            assert fault.fault_type == FaultType.PERFORMANCE_DEGRADATION
            
            # Verify latency increased
            new_latency = self.network.get_connection_info(source, dest)["latency"]
            assert new_latency > initial_latency
            
            # Check fault details
            assert fault.details["original_latency"] == initial_latency
            assert fault.details["new_latency"] == new_latency
    
    def test_inject_random_fault(self):
        """Test random fault injection."""
        fault = self.fault_injector.inject_random_fault()
        
        assert fault.fault_type in [FaultType.LINK_FAILURE, FaultType.DEVICE_FAILURE, 
                                   FaultType.MISCONFIGURATION, FaultType.PERFORMANCE_DEGRADATION]
        assert len(self.fault_injector.get_active_faults()) == 1
    
    def test_clear_all_faults(self):
        """Test clearing all faults."""
        # Inject multiple faults
        fault1 = self.fault_injector.inject_device_failure()
        fault2 = self.fault_injector.inject_link_failure()
        
        assert len(self.fault_injector.get_active_faults()) == 2
        
        # Clear all faults
        self.fault_injector.clear_all_faults()
        
        assert len(self.fault_injector.get_active_faults()) == 0
        assert not self.fault_injector.has_active_faults()
        
        # Verify network is restored
        devices = self.network.get_all_devices()
        for device in devices:
            assert self.network.is_device_up(device)
    
    def test_clear_specific_fault(self):
        """Test clearing a specific fault."""
        fault1 = self.fault_injector.inject_device_failure()
        fault2 = self.fault_injector.inject_link_failure()
        
        assert len(self.fault_injector.get_active_faults()) == 2
        
        # Clear specific fault
        success = self.fault_injector.clear_fault(fault1)
        assert success
        
        active_faults = self.fault_injector.get_active_faults()
        assert len(active_faults) == 1
        assert fault1 not in active_faults
        assert fault2 in active_faults
    
    def test_fault_with_empty_network(self):
        """Test fault injection with empty network."""
        empty_network = NetworkGraph()
        empty_injector = FaultInjector(empty_network)
        
        # Should raise errors for empty network
        with pytest.raises(ValueError):
            empty_injector.inject_link_failure()
        
        with pytest.raises(ValueError):
            empty_injector.inject_device_failure()
    
    def test_fault_info_representation(self):
        """Test FaultInfo string representations."""
        fault = self.fault_injector.inject_device_failure()
        
        fault_str = str(fault)
        assert fault.fault_type.value in fault_str
        assert fault.location in fault_str
        
        fault_repr = repr(fault)
        assert "FaultInfo" in fault_repr
