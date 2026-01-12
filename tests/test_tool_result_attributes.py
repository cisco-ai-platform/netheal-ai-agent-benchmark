"""
Unit tests to verify ToolResult attributes and prevent AttributeError regression.

This test suite ensures that ToolResult objects have the expected attributes
and that demo scripts don't try to access non-existent attributes.
"""

import unittest
import sys
import os

# Add the parent directory to the path to import netheal
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netheal.tools.simulator import ToolResult, ToolSimulator
from netheal.network.topology import TopologyGenerator
from netheal.faults.injector import FaultInjector, FaultType


class TestToolResultAttributes(unittest.TestCase):
    """Test ToolResult class attributes and methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.network = TopologyGenerator.generate_linear_topology(3)
        self.tool_simulator = ToolSimulator(self.network)
        self.devices = self.network.get_all_devices()
    
    def test_tool_result_has_required_attributes(self):
        """Test that ToolResult has all required attributes."""
        # Create a sample ToolResult
        result = ToolResult(
            success=True,
            data={"test": "data"},
            cost=1.0,
            tool_name="test_tool"
        )
        
        # Verify all required attributes exist
        self.assertTrue(hasattr(result, 'success'))
        self.assertTrue(hasattr(result, 'data'))
        self.assertTrue(hasattr(result, 'cost'))
        self.assertTrue(hasattr(result, 'tool_name'))
        
        # Verify attribute types
        self.assertIsInstance(result.success, bool)
        self.assertIsInstance(result.data, dict)
        self.assertIsInstance(result.cost, (int, float))
        self.assertIsInstance(result.tool_name, str)
    
    def test_tool_result_does_not_have_execution_time(self):
        """Test that ToolResult does NOT have execution_time attribute (regression test)."""
        result = ToolResult(
            success=True,
            data={"test": "data"},
            cost=1.0,
            tool_name="test_tool"
        )
        
        # This is the specific bug we're preventing
        self.assertFalse(hasattr(result, 'execution_time'))
        
        # Verify accessing execution_time raises AttributeError
        with self.assertRaises(AttributeError):
            _ = result.execution_time
    
    def test_ping_result_attributes(self):
        """Test that ping results have correct attributes."""
        if len(self.devices) >= 2:
            result = self.tool_simulator.ping(self.devices[0], self.devices[1])
            
            # Verify ToolResult attributes
            self.assertIsInstance(result, ToolResult)
            self.assertTrue(hasattr(result, 'success'))
            self.assertTrue(hasattr(result, 'data'))
            self.assertTrue(hasattr(result, 'cost'))
            self.assertTrue(hasattr(result, 'tool_name'))
            self.assertEqual(result.tool_name, 'ping')
            
            # Verify no execution_time attribute
            self.assertFalse(hasattr(result, 'execution_time'))
    
    def test_traceroute_result_attributes(self):
        """Test that traceroute results have correct attributes."""
        if len(self.devices) >= 2:
            result = self.tool_simulator.traceroute(self.devices[0], self.devices[-1])
            
            # Verify ToolResult attributes
            self.assertIsInstance(result, ToolResult)
            self.assertTrue(hasattr(result, 'success'))
            self.assertTrue(hasattr(result, 'data'))
            self.assertTrue(hasattr(result, 'cost'))
            self.assertTrue(hasattr(result, 'tool_name'))
            self.assertEqual(result.tool_name, 'traceroute')
            
            # Verify no execution_time attribute
            self.assertFalse(hasattr(result, 'execution_time'))
    
    def test_check_status_result_attributes(self):
        """Test that check_status results have correct attributes."""
        if self.devices:
            result = self.tool_simulator.check_status(self.devices[0])
            
            # Verify ToolResult attributes
            self.assertIsInstance(result, ToolResult)
            self.assertTrue(hasattr(result, 'success'))
            self.assertTrue(hasattr(result, 'data'))
            self.assertTrue(hasattr(result, 'cost'))
            self.assertTrue(hasattr(result, 'tool_name'))
            self.assertEqual(result.tool_name, 'check_status')
            
            # Verify no execution_time attribute
            self.assertFalse(hasattr(result, 'execution_time'))
    
    def test_check_interfaces_result_attributes(self):
        """Test that check_interfaces results have correct attributes."""
        if self.devices:
            result = self.tool_simulator.check_interfaces(self.devices[0])
            
            # Verify ToolResult attributes
            self.assertIsInstance(result, ToolResult)
            self.assertTrue(hasattr(result, 'success'))
            self.assertTrue(hasattr(result, 'data'))
            self.assertTrue(hasattr(result, 'cost'))
            self.assertTrue(hasattr(result, 'tool_name'))
            self.assertEqual(result.tool_name, 'check_interfaces')
            
            # Verify no execution_time attribute
            self.assertFalse(hasattr(result, 'execution_time'))
    
    def test_tool_result_with_fault_injection(self):
        """Test ToolResult attributes when network has faults."""
        # Inject a fault
        fault_injector = FaultInjector(self.network)
        fault = fault_injector.inject_random_fault([FaultType.LINK_FAILURE])
        
        if len(self.devices) >= 2:
            # Test ping on faulty network
            result = self.tool_simulator.ping(self.devices[0], self.devices[-1])
            
            # Should still have correct attributes even with network faults
            self.assertIsInstance(result, ToolResult)
            self.assertTrue(hasattr(result, 'success'))
            self.assertTrue(hasattr(result, 'data'))
            self.assertTrue(hasattr(result, 'cost'))
            self.assertTrue(hasattr(result, 'tool_name'))
            
            # Should not have execution_time
            self.assertFalse(hasattr(result, 'execution_time'))
    
    def test_tool_result_string_representation(self):
        """Test that ToolResult string methods work correctly."""
        result = ToolResult(
            success=True,
            data={"latency_ms": 5.2},
            cost=1.5,
            tool_name="ping"
        )
        
        # Test __str__ method
        str_repr = str(result)
        self.assertIn("ping", str_repr)
        self.assertIn("SUCCESS", str_repr)
        self.assertIn("1.5", str_repr)
        
        # Ensure no execution_time in string representation
        self.assertNotIn("execution_time", str_repr)


class TestDemoScriptCompatibility(unittest.TestCase):
    """Test that demo scripts are compatible with ToolResult attributes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.network = TopologyGenerator.generate_star_topology(4)
        self.tool_simulator = ToolSimulator(self.network)
        self.devices = self.network.get_all_devices()
    
    def test_demo_compatible_attribute_access(self):
        """Test that demo-style attribute access works correctly."""
        if len(self.devices) >= 2:
            result = self.tool_simulator.ping(self.devices[0], self.devices[1])
            
            # These should work (attributes that exist)
            success = result.success
            data = result.data
            cost = result.cost
            tool_name = result.tool_name
            
            self.assertIsInstance(success, bool)
            self.assertIsInstance(data, dict)
            self.assertIsInstance(cost, (int, float))
            self.assertIsInstance(tool_name, str)
            
            # This should fail (the bug we're preventing)
            with self.assertRaises(AttributeError):
                _ = result.execution_time
    
    def test_safe_attribute_checking(self):
        """Test safe ways to check for attributes."""
        result = self.tool_simulator.ping(self.devices[0], self.devices[1])
        
        # Safe ways to check for attributes
        self.assertTrue(hasattr(result, 'success'))
        self.assertTrue(hasattr(result, 'cost'))
        self.assertFalse(hasattr(result, 'execution_time'))
        
        # Using getattr with default
        cost = getattr(result, 'cost', 0.0)
        self.assertGreater(cost, 0)
        
        execution_time = getattr(result, 'execution_time', None)
        self.assertIsNone(execution_time)


if __name__ == '__main__':
    unittest.main()
