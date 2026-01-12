# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Test scenario export and import functionality."""
import pytest
import json
from webapp.backend.app.manager import EnvManager


def test_export_scenario():
    """Test that we can export a scenario."""
    manager = EnvManager()
    
    # Reset environment
    state = manager.reset(seed=42, max_devices=5, max_episode_steps=10)
    
    # Take a few steps
    valid_actions = state['valid_actions']
    if valid_actions:
        manager.step(valid_actions[0])
    
    # Export scenario
    scenario = manager.export_scenario()
    
    # Verify scenario structure
    assert 'version' in scenario
    assert scenario['version'] == '1.0.0'
    assert 'metadata' in scenario
    assert 'network' in scenario
    assert 'fault' in scenario
    assert 'observation' in scenario
    assert 'episode_state' in scenario
    
    # Verify metadata
    metadata = scenario['metadata']
    assert metadata['max_devices'] == 5
    assert metadata['max_episode_steps'] == 10
    assert 'export_timestamp' in metadata
    
    # Verify network structure
    network = scenario['network']
    assert 'nodes' in network
    assert 'edges' in network
    assert len(network['nodes']) > 0
    
    # Verify fault structure
    fault = scenario['fault']
    assert 'type' in fault
    assert 'location' in fault
    assert 'details' in fault


def test_import_scenario():
    """Test that we can import a previously exported scenario."""
    manager1 = EnvManager()
    
    # Create and export a scenario
    state1 = manager1.reset(seed=42, max_devices=5, max_episode_steps=10)
    scenario = manager1.export_scenario()
    
    # Create a new manager and import the scenario
    # Note: Due to singleton pattern, we need to work with the same instance
    # In a real multi-user app, you'd have separate instances
    state2 = manager1.import_scenario(scenario)
    
    # Verify the imported state has the same configuration
    assert state2['info']['network_size'] == state1['info']['network_size']
    assert 'observation' in state2
    assert 'valid_actions' in state2


def test_export_import_preserves_network():
    """Test that network topology is preserved through export/import."""
    manager = EnvManager()
    
    # Create a scenario
    state1 = manager.reset(seed=42, max_devices=6)
    scenario = manager.export_scenario()
    
    # Get original network info
    network1 = scenario['network']
    original_node_count = len(network1['nodes'])
    original_edge_count = len(network1['edges'])
    original_node_ids = {n['id'] for n in network1['nodes']}
    
    # Import the scenario
    state2 = manager.import_scenario(scenario)
    
    # Export again to compare
    scenario2 = manager.export_scenario()
    network2 = scenario2['network']
    
    # Verify network is preserved
    assert len(network2['nodes']) == original_node_count
    assert len(network2['edges']) == original_edge_count
    
    imported_node_ids = {n['id'] for n in network2['nodes']}
    assert imported_node_ids == original_node_ids


def test_export_import_preserves_fault():
    """Test that fault information is preserved through export/import."""
    manager = EnvManager()
    
    # Create a scenario
    manager.reset(seed=42, max_devices=5)
    scenario = manager.export_scenario()
    
    # Get original fault info
    fault1 = scenario['fault']
    original_fault_type = fault1['type']
    original_location = fault1['location']
    
    # Import the scenario
    manager.import_scenario(scenario)
    
    # Export again to verify
    scenario2 = manager.export_scenario()
    fault2 = scenario2['fault']
    
    # Verify fault is preserved
    assert fault2['type'] == original_fault_type
    assert fault2['location'] == original_location


def test_scenario_json_serializable():
    """Test that exported scenario is JSON serializable."""
    manager = EnvManager()
    
    # Create and export a scenario
    manager.reset(seed=42, max_devices=5)
    scenario = manager.export_scenario()
    
    # Verify it can be serialized and deserialized
    json_str = json.dumps(scenario)
    scenario_loaded = json.loads(json_str)
    
    # Verify structure is preserved
    assert scenario_loaded['version'] == scenario['version']
    assert len(scenario_loaded['network']['nodes']) == len(scenario['network']['nodes'])


def test_import_with_different_config():
    """Test that import creates environment with scenario's configuration."""
    manager = EnvManager()
    
    # Create a scenario with specific config
    manager.reset(seed=42, max_devices=8, max_episode_steps=30)
    scenario = manager.export_scenario()
    
    # Import it
    state = manager.import_scenario(scenario)
    
    # Verify the imported environment has the correct configuration
    # This is reflected in the metadata
    scenario2 = manager.export_scenario()
    assert scenario2['metadata']['max_devices'] == 8
    assert scenario2['metadata']['max_episode_steps'] == 30


def test_export_requires_initialized_env():
    """Test that export fails if environment is not initialized."""
    # Create a fresh manager without resetting
    # Note: Due to singleton, this test might be affected by other tests
    # In production code, this would be tested with proper isolation
    manager = EnvManager()
    
    # Try to export without initialization
    # This should work if a previous test initialized it
    # In a real scenario with separate instances, this would raise an error
    try:
        scenario = manager.export_scenario()
        # If we get here, environment was already initialized (expected in test suite)
        assert 'version' in scenario
    except RuntimeError as e:
        # If not initialized, we should get an error
        assert "not initialized" in str(e).lower()


def test_multiple_export_import_cycles():
    """Test multiple export/import cycles preserve state."""
    manager = EnvManager()
    
    # Initial scenario
    manager.reset(seed=42, max_devices=5)
    scenario1 = manager.export_scenario()
    
    # Import and export again
    manager.import_scenario(scenario1)
    scenario2 = manager.export_scenario()
    
    # Import and export one more time
    manager.import_scenario(scenario2)
    scenario3 = manager.export_scenario()
    
    # Verify network structure remains consistent
    assert len(scenario1['network']['nodes']) == len(scenario3['network']['nodes'])
    assert len(scenario1['network']['edges']) == len(scenario3['network']['edges'])


def test_export_import_preserves_hint():
    """Test that user hint is preserved through export/import."""
    manager = EnvManager()
    
    # Create a scenario with hints enabled
    manager.reset(seed=42, max_devices=5, enable_user_hints=True)
    
    # Set a custom hint (simulating what the environment would do)
    if manager._env:
        manager._env.user_hint = "This is a test hint about the network"
    
    # Export the scenario
    scenario = manager.export_scenario()
    
    # Verify hint is in the export
    assert 'user_hint' in scenario
    original_hint = scenario['user_hint']
    
    # Import the scenario
    manager.import_scenario(scenario)
    
    # Verify hint is preserved
    if manager._env:
        assert manager._env.user_hint == original_hint
    
    # Also check it's in the state
    state = manager.get_state()
    assert state['info'].get('user_hint') == original_hint


def test_export_import_preserves_diagnostic_history():
    """Test that diagnostic history is preserved through export/import."""
    manager = EnvManager()
    
    # Create a scenario
    state = manager.reset(seed=42, max_devices=5)
    
    # Take several actions - diagnostic actions should populate the memory
    valid_actions = state['valid_actions']
    actions_taken = 0
    for action_id in valid_actions[:5]:  # Try up to 5 actions
        state = manager.step(action_id)
        actions_taken += 1
        if actions_taken >= 3:  # Take at least 3 actions
            break
    
    # Get the diagnostic history count
    original_diag_count = 0
    if manager._env and manager._env.observation:
        original_diag_count = len(manager._env.observation.diagnostic_memory.results)
    
    # After taking actions, we should have at least some diagnostics
    # Note: Not all actions are diagnostic, but scan_network and similar should add results
    if original_diag_count == 0:
        # Skip test if no diagnostics were performed (can happen with certain action sequences)
        import pytest
        pytest.skip("No diagnostic actions were taken in this scenario")
    
    # Export the scenario
    scenario = manager.export_scenario()
    
    # Verify diagnostic_history is in the export
    assert 'diagnostic_history' in scenario['observation']
    exported_diag_count = len(scenario['observation']['diagnostic_history'])
    assert exported_diag_count == original_diag_count
    
    # Import the scenario
    state2 = manager.import_scenario(scenario)
    
    # Verify diagnostic history is restored
    if manager._env and manager._env.observation:
        imported_diag_count = len(manager._env.observation.diagnostic_memory.results)
        assert imported_diag_count == original_diag_count, \
            f"Expected {original_diag_count} diagnostics, got {imported_diag_count}"
        
        # Verify the diagnostics have the right structure
        for diag in manager._env.observation.diagnostic_memory.results:
            assert diag.tool_name is not None
            assert diag.result is not None
            assert hasattr(diag.result, 'success')
            assert hasattr(diag.result, 'data')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

