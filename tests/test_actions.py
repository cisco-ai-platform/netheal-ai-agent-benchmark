# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the enhanced action system."""

import pytest
from netheal.environment.actions import (
    StructuredActionSpace, ActionSpec, ActionCategory,
    TopologyAction, DiagnosticAction,
    validate_action_parameters
)
from netheal.faults.injector import FaultType


class TestActionSpec:
    """Test cases for ActionSpec."""
    
    def test_action_spec_creation(self):
        """Test creating action specifications."""
        spec = ActionSpec(
            category=ActionCategory.DIAGNOSTIC,
            action_type=DiagnosticAction.PING,
            parameters={'source': 'router1', 'destination': 'switch1'},
            description='Ping from router1 to switch1'
        )
        
        assert spec.category == ActionCategory.DIAGNOSTIC
        assert spec.action_type == DiagnosticAction.PING
        assert spec.parameters['source'] == 'router1'
        assert spec.parameters['destination'] == 'switch1'
        assert spec.description == 'Ping from router1 to switch1'
        
    def test_to_dict_conversion(self):
        """Test converting action spec to dictionary."""
        spec = ActionSpec(
            category=ActionCategory.TOPOLOGY_DISCOVERY,
            action_type=TopologyAction.SCAN_NETWORK,
            parameters={'start_device': 'router1'},
            description='Scan network starting from router1'
        )
        
        spec_dict = spec.to_dict()
        
        assert isinstance(spec_dict, dict)
        assert spec_dict['category'] == 'topology_discovery'
        assert spec_dict['action_type'] == 'scan_network'
        assert spec_dict['parameters']['start_device'] == 'router1'
        assert spec_dict['description'] == 'Scan network starting from router1'


class TestStructuredActionSpace:
    """Test cases for StructuredActionSpace."""
    
    def test_action_space_creation(self):
        """Test creating structured action space."""
        action_space = StructuredActionSpace(max_devices=5)
        
        assert action_space.max_devices == 5
        assert action_space.total_actions > 0
        # Note: action_specs may be lazily populated, so we check action_map instead
        assert len(action_space.action_map) > 0
        
    def test_get_action_spec(self):
        """Test getting action specifications by ID."""
        action_space = StructuredActionSpace(max_devices=3)
        
        # Test valid action ID
        spec = action_space.get_action_spec(0)
        assert spec is not None
        assert isinstance(spec, ActionSpec)
        
        # Test invalid action ID
        invalid_spec = action_space.get_action_spec(action_space.total_actions + 10)
        assert invalid_spec is None
        
    def test_topology_discovery_actions(self):
        """Test topology discovery actions are included."""
        action_space = StructuredActionSpace(max_devices=3)
        
        topology_actions = []
        for i in range(action_space.total_actions):
            spec = action_space.get_action_spec(i)
            if spec and spec.category == ActionCategory.TOPOLOGY_DISCOVERY:
                topology_actions.append(spec)
        
        assert len(topology_actions) > 0
        
        # Check for scan network action
        scan_actions = [a for a in topology_actions if a.action_type == TopologyAction.SCAN_NETWORK]
        assert len(scan_actions) > 0
        
    def test_diagnostic_actions(self):
        """Test diagnostic actions are included."""
        action_space = StructuredActionSpace(max_devices=3)
        
        diagnostic_actions = []
        for i in range(action_space.total_actions):
            spec = action_space.get_action_spec(i)
            if spec and spec.category == ActionCategory.DIAGNOSTIC:
                diagnostic_actions.append(spec)
        
        assert len(diagnostic_actions) > 0
        
        # Check for different diagnostic types
        ping_actions = [a for a in diagnostic_actions if a.action_type == DiagnosticAction.PING]
        status_actions = [a for a in diagnostic_actions if a.action_type == DiagnosticAction.CHECK_STATUS]
        
        assert len(ping_actions) > 0
        assert len(status_actions) > 0
        
        
    def test_diagnosis_actions(self):
        """Test diagnosis actions are included."""
        action_space = StructuredActionSpace(max_devices=3)
        
        diagnosis_actions = []
        for i in range(action_space.total_actions):
            spec = action_space.get_action_spec(i)
            if spec and spec.category == ActionCategory.DIAGNOSIS:
                diagnosis_actions.append(spec)
        
        assert len(diagnosis_actions) > 0
        
        # Should have diagnosis actions for each fault type
        fault_types_found = set()
        for action in diagnosis_actions:
            fault_types_found.add(action.action_type)
        
        # Check that we have different fault types
        assert len(fault_types_found) > 1
        
    def test_get_valid_actions(self):
        """Test getting valid actions based on discovered devices."""
        action_space = StructuredActionSpace(max_devices=5)
        
        # With no discovered devices, should still have some actions available
        valid_actions = action_space.get_valid_actions([])
        assert len(valid_actions) > 0
        
        # Should include topology discovery actions
        topology_count = 0
        for action_id in valid_actions:
            spec = action_space.get_action_spec(action_id)
            if spec and spec.category == ActionCategory.TOPOLOGY_DISCOVERY:
                topology_count += 1
        assert topology_count > 0
        
        # With discovered devices, should have more actions
        discovered_devices = ['router1', 'switch1']
        valid_actions_with_devices = action_space.get_valid_actions(discovered_devices)
        assert len(valid_actions_with_devices) >= len(valid_actions)
        
    def test_get_action_descriptions(self):
        """Test getting human-readable action descriptions."""
        action_space = StructuredActionSpace(max_devices=3)
        
        descriptions = action_space.get_action_descriptions()
        
        assert isinstance(descriptions, list)
        assert len(descriptions) > 0  # Should have some descriptions
        
        # Check that descriptions contain expected keywords
        description_text = ' '.join(descriptions)
        assert 'scan_network' in description_text or 'topology_discovery' in description_text
        assert 'ping' in description_text or 'diagnostic' in description_text


class TestActionValidation:
    """Test cases for action parameter validation."""
    
    def test_validate_ping_parameters(self):
        """Test validating ping action parameters."""
        # Valid parameters
        valid_params = {'source': 'router1', 'destination': 'switch1'}
        assert validate_action_parameters(DiagnosticAction.PING, valid_params) is True
        
        # Missing destination
        invalid_params = {'source': 'router1'}
        assert validate_action_parameters(DiagnosticAction.PING, invalid_params) is False
        
        # Missing source
        invalid_params = {'destination': 'switch1'}
        assert validate_action_parameters(DiagnosticAction.PING, invalid_params) is False
        
    def test_validate_status_check_parameters(self):
        """Test validating status check parameters."""
        # Valid parameters
        valid_params = {'device': 'router1'}
        assert validate_action_parameters(DiagnosticAction.CHECK_STATUS, valid_params) is True
        
        # Missing device
        invalid_params = {}
        assert validate_action_parameters(DiagnosticAction.CHECK_STATUS, invalid_params) is False
        
        
    def test_validate_scan_network_parameters(self):
        """Test validating scan network parameters."""
        # Valid parameters (start_device is optional)
        valid_params = {'start_device': 'router1'}
        assert validate_action_parameters(TopologyAction.SCAN_NETWORK, valid_params) is True
        
        # Empty parameters should also be valid (uses default)
        empty_params = {}
        assert validate_action_parameters(TopologyAction.SCAN_NETWORK, empty_params) is True
        
    def test_validate_discover_neighbors_parameters(self):
        """Test validating discover neighbors parameters."""
        # Valid parameters
        valid_params = {'device': 'router1'}
        assert validate_action_parameters(TopologyAction.DISCOVER_NEIGHBORS, valid_params) is True
        
        # Missing device
        invalid_params = {}
        assert validate_action_parameters(TopologyAction.DISCOVER_NEIGHBORS, invalid_params) is False


class TestActionCategories:
    """Test cases for action categories and enums."""
    
    def test_action_category_enum(self):
        """Test ActionCategory enum values."""
        assert ActionCategory.TOPOLOGY_DISCOVERY.value == "topology_discovery"
        assert ActionCategory.DIAGNOSTIC.value == "diagnostic"
        assert ActionCategory.DIAGNOSIS.value == "diagnosis"
        
    def test_topology_action_enum(self):
        """Test TopologyAction enum values."""
        assert TopologyAction.SCAN_NETWORK.value == "scan_network"
        assert TopologyAction.DISCOVER_NEIGHBORS.value == "discover_neighbors"
        
    def test_diagnostic_action_enum(self):
        """Test DiagnosticAction enum values."""
        assert DiagnosticAction.PING.value == "ping"
        assert DiagnosticAction.TRACEROUTE.value == "traceroute"
        assert DiagnosticAction.CHECK_STATUS.value == "check_status"
        assert DiagnosticAction.CHECK_INTERFACES.value == "check_interfaces"
        


class TestActionSpaceIntegration:
    """Integration tests for action space functionality."""
    
    def test_action_space_completeness(self):
        """Test that action space includes all expected action types."""
        action_space = StructuredActionSpace(max_devices=4)
        
        categories_found = set()
        action_types_found = set()
        
        for i in range(action_space.total_actions):
            spec = action_space.get_action_spec(i)
            if spec:
                categories_found.add(spec.category)
                action_types_found.add(spec.action_type)
        
        # Should have all categories
        expected_categories = {
            ActionCategory.TOPOLOGY_DISCOVERY,
            ActionCategory.DIAGNOSTIC,
            ActionCategory.DIAGNOSIS
        }
        assert categories_found == expected_categories
        
        # Should have multiple action types
        assert len(action_types_found) >= 6  # Check for a reasonable number of action types
        
    def test_action_parameter_consistency(self):
        """Test that action parameters are consistent with action types."""
        action_space = StructuredActionSpace(max_devices=3)
        
        for i in range(min(100, action_space.total_actions)):  # Test first 100 actions
            spec = action_space.get_action_spec(i)
            if spec:
                # Validate parameters for this action type
                is_valid = validate_action_parameters(spec.action_type, spec.parameters)
                assert is_valid, f"Invalid parameters for action {i}: {spec.action_type} with {spec.parameters}"
                
    def test_action_space_scaling(self):
        """Test that action space scales appropriately with max_devices."""
        small_space = StructuredActionSpace(max_devices=3)
        large_space = StructuredActionSpace(max_devices=6)
        
        # Larger device count should result in more actions
        assert large_space.total_actions > small_space.total_actions
        
        # Both should have the same categories available
        small_categories = set()
        large_categories = set()
        
        for i in range(small_space.total_actions):
            spec = small_space.get_action_spec(i)
            if spec:
                small_categories.add(spec.category)
                
        for i in range(large_space.total_actions):
            spec = large_space.get_action_spec(i)
            if spec:
                large_categories.add(spec.category)
        
        assert small_categories == large_categories
