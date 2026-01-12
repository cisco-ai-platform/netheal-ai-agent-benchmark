"""
Integration tests for demo scripts to prevent attribute access bugs.

This test suite verifies that demo scripts can run without AttributeError
exceptions and that they only access valid ToolResult attributes.
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock
from io import StringIO

# Add the parent directory to the path to import netheal
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netheal.tools.simulator import ToolResult, ToolSimulator
from netheal.network.topology import TopologyGenerator
from netheal.faults.injector import FaultInjector, FaultType


class TestDemoIntegration(unittest.TestCase):
    """Test demo script integration to prevent attribute access bugs."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.network = TopologyGenerator.generate_linear_topology(4)
        self.tool_simulator = ToolSimulator(self.network)
        self.devices = self.network.get_all_devices()
    
    def test_demo_style_tool_result_usage(self):
        """Test the pattern used in demo scripts for displaying tool results."""
        if len(self.devices) >= 2:
            result = self.tool_simulator.ping(self.devices[0], self.devices[-1])
            
            # This is the pattern used in the fixed demo
            try:
                success_output = f"Success: {result.success}"
                data_output = f"Data: {result.data}"
                cost_output = f"Cost: {result.cost:.1f}"
                
                # These should all work without errors
                self.assertIn("Success:", success_output)
                self.assertIn("Data:", data_output)
                self.assertIn("Cost:", cost_output)
                
            except AttributeError as e:
                self.fail(f"Demo-style attribute access failed: {e}")
    
    def test_all_diagnostic_tools_have_valid_attributes(self):
        """Test that all diagnostic tools return ToolResult with valid attributes."""
        tools_and_methods = [
            ("ping", lambda: self.tool_simulator.ping(self.devices[0], self.devices[-1])),
            ("traceroute", lambda: self.tool_simulator.traceroute(self.devices[0], self.devices[-1])),
            ("check_status", lambda: self.tool_simulator.check_status(self.devices[0])),
            ("check_interfaces", lambda: self.tool_simulator.check_interfaces(self.devices[0]))
        ]
        
        for tool_name, tool_method in tools_and_methods:
            with self.subTest(tool=tool_name):
                result = tool_method()
                
                # Verify all required attributes exist and are accessible
                self.assertIsInstance(result.success, bool)
                self.assertIsInstance(result.data, dict)
                self.assertIsInstance(result.cost, (int, float))
                self.assertIsInstance(result.tool_name, str)
                
                # Verify the problematic attribute doesn't exist
                with self.assertRaises(AttributeError):
                    _ = result.execution_time
    
    def test_demo_output_formatting_safety(self):
        """Test that demo output formatting doesn't access invalid attributes."""
        result = self.tool_simulator.ping(self.devices[0], self.devices[1])
        
        # Safe formatting patterns that should work
        safe_formats = [
            f"📊 Success: {result.success}",
            f"📄 Data: {result.data}",
            f"💰 Cost: {result.cost:.1f}",
            f"🔧 Tool: {result.tool_name}"
        ]
        
        for format_str in safe_formats:
            self.assertIsInstance(format_str, str)
            self.assertGreater(len(format_str), 0)
        
        # Unsafe formatting that should fail
        with self.assertRaises(AttributeError):
            _ = f"⏱️ Execution time: {result.execution_time:.3f}s"
    
    def test_fault_injection_demo_compatibility(self):
        """Test that fault injection doesn't break tool result attributes."""
        # Inject various fault types
        fault_injector = FaultInjector(self.network)
        
        fault_types = [FaultType.LINK_FAILURE, FaultType.DEVICE_FAILURE]
        
        for fault_type in fault_types:
            with self.subTest(fault_type=fault_type):
                # Clear previous faults
                fault_injector.clear_all_faults()
                
                # Inject new fault
                try:
                    if fault_type == FaultType.LINK_FAILURE:
                        fault = fault_injector.inject_link_failure()
                    elif fault_type == FaultType.DEVICE_FAILURE:
                        fault = fault_injector.inject_device_failure()
                    
                    # Test tool results with fault present
                    result = self.tool_simulator.ping(self.devices[0], self.devices[-1])
                    
                    # Should still have valid attributes
                    self.assertTrue(hasattr(result, 'success'))
                    self.assertTrue(hasattr(result, 'data'))
                    self.assertTrue(hasattr(result, 'cost'))
                    self.assertTrue(hasattr(result, 'tool_name'))
                    
                    # Should not have execution_time
                    self.assertFalse(hasattr(result, 'execution_time'))
                    
                except ValueError:
                    # Some fault types might not be injectable on this network
                    pass
    
    def test_mock_demo_script_execution(self):
        """Test a mock version of demo script execution patterns."""
        # Mock the interactive parts
        with patch('builtins.input', return_value=''):
            with patch('time.sleep'):
                with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                    
                    # Simulate the demo pattern
                    tools = [
                        ("ping", "Test connectivity between devices"),
                        ("check_status", "Check device status"),
                    ]
                    
                    for tool_name, description in tools:
                        print(f"🔹 Using {tool_name}")
                        print(f"   {description}")
                        
                        if tool_name == "ping":
                            result = self.tool_simulator.ping(self.devices[0], self.devices[-1])
                        elif tool_name == "check_status":
                            result = self.tool_simulator.check_status(self.devices[0])
                        
                        # This is the fixed pattern from the demo
                        print(f"   📊 Success: {result.success}")
                        print(f"   📄 Data: {result.data}")
                        print(f"   💰 Cost: {result.cost:.1f}")
                    
                    # Verify output was generated without errors
                    output = mock_stdout.getvalue()
                    self.assertIn("Using ping", output)
                    self.assertIn("Success:", output)
                    self.assertIn("Cost:", output)
                    
                    # Verify no execution_time in output
                    self.assertNotIn("execution_time", output)
                    self.assertNotIn("Execution time:", output)


if __name__ == '__main__':
    unittest.main()
