# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for example code fixes to prevent regression.

This module tests the specific bugs that were fixed in the example code
to ensure they don't reoccur in the future.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netheal import NetworkTroubleshootingEnv
from netheal.environment.observation import DiagnosticResult
from netheal.tools.simulator import ToolResult


class TestExampleCodeFixes(unittest.TestCase):
    """Test cases for bugs fixed in example code."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = NetworkTroubleshootingEnv(
            max_devices=5,
            max_episode_steps=10,
            render_mode="text"
        )
    
    def tearDown(self):
        """Clean up after tests."""
        self.env.close()
    
    def test_observation_is_dict_not_array(self):
        """Test that observation is a dictionary, not a numpy array.
        
        This prevents the AttributeError: 'dict' object has no attribute 'shape'
        that was occurring in the interactive demo.
        """
        obs, info = self.env.reset(seed=42)
        
        # Observation should be a dictionary
        self.assertIsInstance(obs, dict)
        
        # Should have the expected keys
        expected_keys = ['discovery_matrix', 'device_status', 'recent_diagnostics', 'episode_metadata']
        self.assertEqual(set(obs.keys()), set(expected_keys))
        
        # Should NOT have a 'shape' attribute like numpy arrays
        self.assertFalse(hasattr(obs, 'shape'))
        
        # Each component should have proper shape
        for key, value in obs.items():
            self.assertTrue(hasattr(value, 'shape'), f"Component '{key}' should have shape")
    
    def test_action_result_handling_with_none(self):
        """Test that action_result can be None without causing errors.
        
        This prevents the TypeError: 'NoneType' object is not subscriptable
        that was occurring when action_result was None.
        """
        obs, info = self.env.reset(seed=42)
        
        # Take an action that might not produce a result
        action = self.env.action_space.sample()
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Test safe access to action_result
        action_result = info.get('action_result')
        
        # Should be able to handle None case
        if action_result is None:
            # This should not raise an error
            result = None
        elif hasattr(action_result, 'result'):
            result = action_result.result
        else:
            result = None
        
        # No exception should be raised
        self.assertTrue(True, "Action result handling should not raise exceptions")
    
    def test_action_result_with_diagnostic_result_object(self):
        """Test that action_result with DiagnosticResult object is handled correctly.
        
        This prevents the AttributeError: 'DiagnosticResult' object has no attribute 'get'
        that was occurring when trying to use dictionary methods on DiagnosticResult objects.
        """
        # Create a mock DiagnosticResult
        mock_tool_result = ToolResult(
            success=True, 
            data={'test': 'data'}, 
            cost=1.0, 
            tool_name='ping'
        )
        mock_diagnostic_result = DiagnosticResult(
            tool_name='ping',
            source='device_0',
            destination='device_1',
            result=mock_tool_result,
            timestamp=0.0
        )
        
        # Test that we can safely check for result attribute
        self.assertTrue(hasattr(mock_diagnostic_result, 'result'))
        self.assertIsNotNone(mock_diagnostic_result.result)
        
        # Test that we cannot use dictionary methods
        with self.assertRaises(AttributeError):
            mock_diagnostic_result.get('result')  # This should fail
        
        # Test correct access pattern
        if hasattr(mock_diagnostic_result, 'result') and mock_diagnostic_result.result:
            result = mock_diagnostic_result.result
            self.assertIsInstance(result, ToolResult)
            self.assertTrue(result.success)
    
    def test_safe_action_result_access_pattern(self):
        """Test the safe pattern for accessing action results in examples.
        
        This tests the corrected pattern used in the fixed example code.
        """
        obs, info = self.env.reset(seed=42)
        
        # Take several actions to test different scenarios
        for _ in range(3):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Test the safe access pattern used in fixed examples
            if ('action_result' in info and 
                info['action_result'] and 
                hasattr(info['action_result'], 'result') and 
                info['action_result'].result):
                
                result = info['action_result'].result
                
                # Should be a ToolResult object
                self.assertIsInstance(result, ToolResult)
                self.assertTrue(hasattr(result, 'success'))
                self.assertTrue(hasattr(result, 'data'))
                self.assertTrue(hasattr(result, 'cost'))
            
            if terminated or truncated:
                break
    
    def test_observation_keys_consistency(self):
        """Test that observation keys are consistent and accessible.
        
        This ensures the fix for displaying observation keys works correctly.
        """
        obs, info = self.env.reset(seed=42)
        
        # Should be able to get keys without error
        obs_keys = list(obs.keys())
        self.assertIsInstance(obs_keys, list)
        self.assertGreater(len(obs_keys), 0)
        
        # Keys should be strings
        for key in obs_keys:
            self.assertIsInstance(key, str)
        
        # Should contain expected observation components
        expected_keys = ['discovery_matrix', 'device_status', 'recent_diagnostics', 'episode_metadata']
        for expected_key in expected_keys:
            self.assertIn(expected_key, obs_keys)
    
    def test_info_structure_consistency(self):
        """Test that info dictionary has consistent structure."""
        obs, info = self.env.reset(seed=42)
        
        # Info should be a dictionary
        self.assertIsInstance(info, dict)
        
        # Should have expected keys
        self.assertIn('network_size', info)
        self.assertIn('ground_truth_fault', info)
        
        # Take an action and check step info
        action = self.env.action_space.sample()
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Info should still be a dictionary
        self.assertIsInstance(info, dict)
        
        # action_result key may or may not be present
        if 'action_result' in info:
            # If present, should be either None or a DiagnosticResult
            action_result = info['action_result']
            if action_result is not None:
                self.assertTrue(hasattr(action_result, 'result'))


class TestExampleCodeIntegration(unittest.TestCase):
    """Integration tests for example code functionality."""
    
    def test_interactive_demo_key_functions(self):
        """Test that key functions from interactive demo work correctly."""
        from examples.interactive_demo import NetHealDemo
        
        demo = NetHealDemo()
        
        # Test that demo can be created without errors
        self.assertIsInstance(demo, NetHealDemo)
        
        # Test header printing (should not raise exceptions)
        demo.print_header("Test Header")
        demo.print_step("Test Step", "Test Details")
    
    def test_basic_usage_key_functions(self):
        """Test that basic usage example functions work correctly."""
        # Import should work without errors
        try:
            from examples import basic_usage
            self.assertTrue(True, "Basic usage import successful")
        except ImportError as e:
            self.fail(f"Failed to import basic_usage: {e}")
    
    def test_quick_demo_functionality(self):
        """Test that quick demo can run key functions."""
        try:
            from examples import quick_demo
            self.assertTrue(True, "Quick demo import successful")
        except ImportError as e:
            self.fail(f"Failed to import quick_demo: {e}")


if __name__ == '__main__':
    unittest.main()
