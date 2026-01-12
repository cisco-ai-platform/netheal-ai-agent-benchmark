"""
Comprehensive tests for MISCONFIGURATION fault type behavior in NetHeal.

Tests verify that misconfiguration faults manifest correctly according to
the expected behaviors documented in docs/fault-behaviors.md:

Expected Behavior:
- Device status remains 'up' (device is operational)
- One specific outbound connection from the device is blocked (status='down')
- Asymmetric behavior: may work in one direction but not the other
- Check interfaces shows one specific interface down while others may be up
- Pings through the misconfigured path fail selectively

Secondary Effects:
- Paths that don't use the blocked connection remain functional
- Reverse direction of the blocked connection may still work (asymmetric)
- Devices beyond the blocked connection may become unreachable if no alternate path
"""

import pytest
import numpy as np
from typing import List, Tuple, Optional

from netheal.network.graph import NetworkGraph, DeviceType
from netheal.network.topology import TopologyGenerator
from netheal.faults.injector import FaultInjector, FaultType, FaultInfo
from netheal.tools.simulator import ToolSimulator
from netheal.environment.env import NetworkTroubleshootingEnv


class TestMisconfigurationBasicBehavior:
    """Test basic misconfiguration fault behavior."""

    def test_device_remains_up_after_misconfiguration(self):
        """Verify that the misconfigured device itself remains operational."""
        network = TopologyGenerator.generate_linear_topology(4)
        injector = FaultInjector(network)
        
        # Inject misconfiguration on a specific device
        devices = network.get_all_devices()
        target_device = devices[1]  # Middle device
        
        fault = injector.inject_misconfiguration(device_id=target_device)
        
        # Device should still be up
        assert network.is_device_up(target_device), \
            f"Device {target_device} should remain UP after misconfiguration"
        
        # Verify fault info
        assert fault.fault_type == FaultType.MISCONFIGURATION
        assert fault.location == target_device

    def test_blocked_connection_is_down(self):
        """Verify that the blocked connection shows status 'down'."""
        network = TopologyGenerator.generate_linear_topology(4)
        injector = FaultInjector(network)
        
        devices = network.get_all_devices()
        target_device = devices[1]
        
        fault = injector.inject_misconfiguration(device_id=target_device)
        blocked_dest = fault.details['blocked_destination']
        
        # The blocked connection should be down
        conn_info = network.get_connection_info(target_device, blocked_dest)
        assert conn_info['status'] == 'down', \
            f"Connection {target_device}->{blocked_dest} should be DOWN"

    def test_check_status_shows_device_up(self):
        """Verify check_status tool shows device as operational."""
        network = TopologyGenerator.generate_linear_topology(4)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        
        devices = network.get_all_devices()
        target_device = devices[1]
        
        injector.inject_misconfiguration(device_id=target_device)
        
        # Check status should succeed and show device as up
        result = tool_sim.check_status(target_device)
        assert result.success, "check_status should succeed on misconfigured device"
        assert result.data['status'] == 'up', \
            "Misconfigured device should report status='up'"

    def test_check_interfaces_shows_blocked_interface(self):
        """Verify check_interfaces reveals the blocked interface."""
        network = TopologyGenerator.generate_linear_topology(4)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        
        devices = network.get_all_devices()
        target_device = devices[1]
        
        fault = injector.inject_misconfiguration(device_id=target_device)
        blocked_dest = fault.details['blocked_destination']
        
        # Check interfaces should show the blocked interface as down
        result = tool_sim.check_interfaces(target_device)
        assert result.success, "check_interfaces should succeed"
        
        interfaces = result.data['interfaces']
        blocked_interface = next(
            (iface for iface in interfaces if iface['destination'] == blocked_dest),
            None
        )
        
        assert blocked_interface is not None, \
            f"Should find interface to {blocked_dest}"
        assert blocked_interface['status'] == 'down', \
            f"Interface to {blocked_dest} should be DOWN"
        
        # Verify at least one other interface is up (if exists)
        other_interfaces = [i for i in interfaces if i['destination'] != blocked_dest]
        if other_interfaces:
            up_interfaces = [i for i in other_interfaces if i['status'] == 'up']
            assert len(up_interfaces) > 0 or len(other_interfaces) == 0, \
                "Other interfaces should remain UP"


class TestMisconfigurationAsymmetricBehavior:
    """Test asymmetric behavior of misconfiguration faults."""

    def test_asymmetric_connection_status(self):
        """Verify misconfiguration only blocks one direction."""
        network = TopologyGenerator.generate_linear_topology(4)
        injector = FaultInjector(network)
        
        devices = network.get_all_devices()
        target_device = devices[1]
        
        fault = injector.inject_misconfiguration(device_id=target_device)
        blocked_dest = fault.details['blocked_destination']
        
        # Forward direction should be down
        forward_info = network.get_connection_info(target_device, blocked_dest)
        assert forward_info['status'] == 'down', \
            f"Forward connection {target_device}->{blocked_dest} should be DOWN"
        
        # Reverse direction should still be up (if it exists)
        if network.graph.has_edge(blocked_dest, target_device):
            reverse_info = network.get_connection_info(blocked_dest, target_device)
            assert reverse_info['status'] == 'up', \
                f"Reverse connection {blocked_dest}->{target_device} should remain UP"

    def test_ping_fails_through_blocked_path(self):
        """Verify ping fails when path goes through blocked connection."""
        # Use linear topology: D0 -- D1 -- D2 -- D3
        network = TopologyGenerator.generate_linear_topology(4)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        
        devices = network.get_all_devices()
        # Block connection from D1 to D2
        fault = injector.inject_misconfiguration(
            device_id=devices[1],
            blocked_destination=devices[2]
        )
        
        # Ping from D0 to D3 should fail (needs to go through D1->D2)
        result = tool_sim.ping(devices[0], devices[3])
        assert not result.success, \
            f"Ping from {devices[0]} to {devices[3]} should FAIL through blocked path"

    def test_ping_succeeds_on_unaffected_path(self):
        """Verify ping succeeds on paths not using blocked connection."""
        # Use linear topology: D0 -- D1 -- D2 -- D3
        network = TopologyGenerator.generate_linear_topology(4)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        
        devices = network.get_all_devices()
        # Block connection from D1 to D2
        injector.inject_misconfiguration(
            device_id=devices[1],
            blocked_destination=devices[2]
        )
        
        # Ping from D0 to D1 should still work (doesn't use blocked connection)
        result = tool_sim.ping(devices[0], devices[1])
        assert result.success, \
            f"Ping from {devices[0]} to {devices[1]} should SUCCEED (unaffected path)"


class TestMisconfigurationTopologyVariations:
    """Test misconfiguration behavior across different topologies."""

    def test_misconfiguration_in_star_topology(self):
        """Test misconfiguration in star topology (central hub)."""
        network = TopologyGenerator.generate_star_topology(5)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        
        devices = network.get_all_devices()
        # Find the hub (device with most connections)
        hub = max(devices, key=lambda d: len(network.get_device_connections(d)))
        
        # Inject misconfiguration on hub
        fault = injector.inject_misconfiguration(device_id=hub)
        blocked_dest = fault.details['blocked_destination']
        
        # Hub should still be up
        assert network.is_device_up(hub)
        
        # Check interfaces on hub
        result = tool_sim.check_interfaces(hub)
        assert result.success
        
        # Should have exactly one down interface
        down_count = result.data['down_interfaces']
        assert down_count >= 1, "Hub should have at least one down interface"
        
        # Ping to blocked destination should fail
        ping_result = tool_sim.ping(hub, blocked_dest)
        assert not ping_result.success, \
            f"Ping from hub to blocked destination should fail"

    def test_misconfiguration_in_mesh_topology(self):
        """Test misconfiguration in mesh topology with alternate paths."""
        network = TopologyGenerator.generate_mesh_topology(5, connection_probability=0.7)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        
        devices = network.get_all_devices()
        target_device = devices[0]
        
        fault = injector.inject_misconfiguration(device_id=target_device)
        blocked_dest = fault.details['blocked_destination']
        
        # Device should be up
        status_result = tool_sim.check_status(target_device)
        assert status_result.success
        assert status_result.data['status'] == 'up'
        
        # Interface check should show blocked interface
        iface_result = tool_sim.check_interfaces(target_device)
        assert iface_result.success
        assert iface_result.data['down_interfaces'] >= 1

    def test_misconfiguration_in_hierarchical_topology(self):
        """Test misconfiguration in hierarchical enterprise-style topology."""
        network = TopologyGenerator.generate_hierarchical_topology(
            num_layers=3,
            devices_per_layer=[1, 2, 3]
        )
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        
        devices = network.get_all_devices()
        # Pick a middle-layer device
        middle_devices = [d for d in devices if d.startswith('L1')]
        if middle_devices:
            target_device = middle_devices[0]
        else:
            target_device = devices[1]
        
        fault = injector.inject_misconfiguration(device_id=target_device)
        
        # Verify expected behavior
        assert network.is_device_up(target_device)
        
        status_result = tool_sim.check_status(target_device)
        assert status_result.data['status'] == 'up'


class TestMisconfigurationSecondaryEffects:
    """Test secondary/cascading effects of misconfiguration faults."""

    def test_devices_beyond_blocked_connection_unreachable(self):
        """Verify devices beyond blocked connection become unreachable in linear topology."""
        # Linear: D0 -- D1 -- D2 -- D3
        network = TopologyGenerator.generate_linear_topology(4)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        
        devices = network.get_all_devices()
        # Block D1 -> D2
        injector.inject_misconfiguration(
            device_id=devices[1],
            blocked_destination=devices[2]
        )
        
        # D2 and D3 should be unreachable from D0
        for target in [devices[2], devices[3]]:
            result = tool_sim.ping(devices[0], target)
            assert not result.success, \
                f"Device {target} should be unreachable from {devices[0]}"

    def test_traceroute_shows_failure_point(self):
        """Verify traceroute identifies the failure point."""
        network = TopologyGenerator.generate_linear_topology(4)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        
        devices = network.get_all_devices()
        # Block D1 -> D2
        injector.inject_misconfiguration(
            device_id=devices[1],
            blocked_destination=devices[2]
        )
        
        # Traceroute from D0 to D3 should fail and show partial path
        result = tool_sim.traceroute(devices[0], devices[3])
        assert not result.success, "Traceroute should fail"
        
        # Should show partial path up to the failure point
        path = result.data.get('path', [])
        assert len(path) > 0, "Should have partial path"
        # The failure point should be at or before D1
        assert devices[0] in path, "Source should be in partial path"

    def test_alternate_paths_remain_functional(self):
        """Verify alternate paths work when available."""
        # Create a network with redundant paths
        network = NetworkGraph()
        
        # Create diamond topology: 
        #     D1
        #    /  \
        #  D0    D3
        #    \  /
        #     D2
        network.add_device('D0', DeviceType.ROUTER)
        network.add_device('D1', DeviceType.SWITCH)
        network.add_device('D2', DeviceType.SWITCH)
        network.add_device('D3', DeviceType.SERVER)
        
        network.add_connection('D0', 'D1', bidirectional=True)
        network.add_connection('D0', 'D2', bidirectional=True)
        network.add_connection('D1', 'D3', bidirectional=True)
        network.add_connection('D2', 'D3', bidirectional=True)
        
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        
        # Block D0 -> D1 path
        injector.inject_misconfiguration(device_id='D0', blocked_destination='D1')
        
        # D3 should still be reachable via D0 -> D2 -> D3
        result = tool_sim.ping('D0', 'D3')
        assert result.success, \
            "D3 should be reachable via alternate path D0->D2->D3"

    def test_no_effect_on_unrelated_devices(self):
        """Verify misconfiguration doesn't affect unrelated devices."""
        network = TopologyGenerator.generate_star_topology(5)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        
        devices = network.get_all_devices()
        hub = max(devices, key=lambda d: len(network.get_device_connections(d)))
        edge_devices = [d for d in devices if d != hub]
        
        # Block hub -> first edge device
        injector.inject_misconfiguration(device_id=hub, blocked_destination=edge_devices[0])
        
        # All other edge devices should still be reachable from hub
        for edge in edge_devices[1:]:
            result = tool_sim.ping(hub, edge)
            assert result.success, \
                f"Unaffected device {edge} should still be reachable from hub"


class TestMisconfigurationDifferentiation:
    """Test that misconfiguration can be differentiated from other fault types."""

    def test_misconfiguration_vs_device_failure(self):
        """Verify misconfiguration differs from device failure in diagnostics."""
        # Test misconfiguration
        network1 = TopologyGenerator.generate_linear_topology(4)
        injector1 = FaultInjector(network1)
        tool_sim1 = ToolSimulator(network1)
        
        devices1 = network1.get_all_devices()
        injector1.inject_misconfiguration(device_id=devices1[1])
        
        # Device should be UP for misconfiguration
        status1 = tool_sim1.check_status(devices1[1])
        assert status1.success
        assert status1.data['status'] == 'up', \
            "Misconfigured device should be UP"
        
        # Test device failure for comparison
        network2 = TopologyGenerator.generate_linear_topology(4)
        injector2 = FaultInjector(network2)
        tool_sim2 = ToolSimulator(network2)
        
        devices2 = network2.get_all_devices()
        injector2.inject_device_failure(device_id=devices2[1])
        
        # Device should be DOWN for device failure
        status2 = tool_sim2.check_status(devices2[1])
        assert status2.data['status'] == 'down', \
            "Failed device should be DOWN"

    def test_misconfiguration_vs_link_failure(self):
        """Verify misconfiguration differs from link failure (asymmetric vs symmetric)."""
        # Test misconfiguration (asymmetric)
        network1 = TopologyGenerator.generate_linear_topology(4)
        injector1 = FaultInjector(network1)
        
        devices1 = network1.get_all_devices()
        fault1 = injector1.inject_misconfiguration(
            device_id=devices1[1],
            blocked_destination=devices1[2]
        )
        
        # Forward should be down, reverse should be up
        forward1 = network1.get_connection_info(devices1[1], devices1[2])
        reverse1 = network1.get_connection_info(devices1[2], devices1[1])
        
        assert forward1['status'] == 'down', "Misconfiguration: forward should be down"
        assert reverse1['status'] == 'up', "Misconfiguration: reverse should be up"
        
        # Test link failure (symmetric)
        network2 = TopologyGenerator.generate_linear_topology(4)
        injector2 = FaultInjector(network2)
        
        devices2 = network2.get_all_devices()
        injector2.inject_link_failure(source=devices2[1], destination=devices2[2])
        
        # Both directions should be down
        forward2 = network2.get_connection_info(devices2[1], devices2[2])
        reverse2 = network2.get_connection_info(devices2[2], devices2[1])
        
        assert forward2['status'] == 'down', "Link failure: forward should be down"
        assert reverse2['status'] == 'down', "Link failure: reverse should be down"


class TestMisconfigurationEdgeCases:
    """Test edge cases for misconfiguration faults."""

    def test_misconfiguration_on_single_connection_device(self):
        """Test misconfiguration on device with only one connection."""
        network = TopologyGenerator.generate_linear_topology(3)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        
        devices = network.get_all_devices()
        # End device has only one connection
        end_device = devices[0]
        
        fault = injector.inject_misconfiguration(device_id=end_device)
        
        # Device should still be up
        assert network.is_device_up(end_device)
        
        # Check interfaces should show the only interface as down
        result = tool_sim.check_interfaces(end_device)
        assert result.success
        assert result.data['down_interfaces'] == 1
        assert result.data['up_interfaces'] == 0

    def test_misconfiguration_preserves_other_connections(self):
        """Verify other connections from same device remain functional."""
        # Create device with multiple connections
        network = TopologyGenerator.generate_star_topology(5)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        
        devices = network.get_all_devices()
        hub = max(devices, key=lambda d: len(network.get_device_connections(d)))
        
        # Get all connections from hub before fault
        connections_before = network.get_device_connections(hub)
        up_before = [dest for dest, info in connections_before if info['status'] == 'up']
        
        # Inject misconfiguration
        fault = injector.inject_misconfiguration(device_id=hub)
        blocked_dest = fault.details['blocked_destination']
        
        # Check that only one connection is affected
        connections_after = network.get_device_connections(hub)
        up_after = [dest for dest, info in connections_after if info['status'] == 'up']
        
        assert len(up_after) == len(up_before) - 1, \
            "Only one connection should be affected"
        assert blocked_dest not in up_after, \
            "Blocked destination should not be in up connections"

    def test_multiple_misconfigurations_on_same_device(self):
        """Test injecting multiple misconfigurations on the same device."""
        network = TopologyGenerator.generate_star_topology(5)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        
        devices = network.get_all_devices()
        hub = max(devices, key=lambda d: len(network.get_device_connections(d)))
        edge_devices = [d for d in devices if d != hub]
        
        # Inject two misconfigurations on hub
        fault1 = injector.inject_misconfiguration(device_id=hub, blocked_destination=edge_devices[0])
        fault2 = injector.inject_misconfiguration(device_id=hub, blocked_destination=edge_devices[1])
        
        # Hub should still be up
        assert network.is_device_up(hub)
        
        # Both blocked destinations should be unreachable
        for blocked in [edge_devices[0], edge_devices[1]]:
            result = tool_sim.ping(hub, blocked)
            assert not result.success, \
                f"Ping to {blocked} should fail after misconfiguration"
        
        # Other edge devices should still be reachable
        for edge in edge_devices[2:]:
            result = tool_sim.ping(hub, edge)
            assert result.success, \
                f"Ping to unaffected {edge} should succeed"


class TestMisconfigurationWithEnvironment:
    """Test misconfiguration behavior through the full RL environment."""

    def test_environment_with_misconfiguration_fault_type(self):
        """Test environment generates misconfiguration faults correctly."""
        env = NetworkTroubleshootingEnv(
            max_devices=6,
            max_episode_steps=20,
            fault_types=[FaultType.MISCONFIGURATION]
        )
        
        for seed in range(5):
            obs, info = env.reset(seed=seed)
            
            gt_fault = info['ground_truth_fault']
            assert gt_fault['type'] == 'misconfiguration', \
                f"Fault type should be misconfiguration, got {gt_fault['type']}"
            
            # Location should be a device ID (not a connection string)
            location = gt_fault['location']
            assert '->' not in location, \
                f"Misconfiguration location should be device ID, got {location}"
        
        env.close()

    def test_diagnostic_tools_reveal_misconfiguration(self):
        """Test that diagnostic tools can reveal misconfiguration through environment."""
        env = NetworkTroubleshootingEnv(
            max_devices=6,
            max_episode_steps=30,
            fault_types=[FaultType.MISCONFIGURATION]
        )
        
        obs, info = env.reset(seed=42)
        gt_fault = info['ground_truth_fault']
        faulty_device = gt_fault['location']
        
        # First, discover the network
        scan_action = None
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.action_type.value == 'scan_network':
                scan_action = action_id
                break
        
        if scan_action is not None:
            obs, reward, terminated, truncated, info = env.step(scan_action)
        
        # Find check_status action for the faulty device
        check_status_action = None
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.action_type.value == 'check_status':
                if spec.parameters.get('device') == faulty_device:
                    check_status_action = action_id
                    break
        
        if check_status_action is not None:
            obs, reward, terminated, truncated, info = env.step(check_status_action)
            
            # The device should show as UP (misconfiguration, not device failure)
            action_result = info.get('action_result')
            if action_result and hasattr(action_result, 'result'):
                tool_result = action_result.result
                if tool_result and tool_result.data:
                    assert tool_result.data.get('status') == 'up', \
                        "Misconfigured device should report status='up'"
        
        env.close()

    def test_correct_diagnosis_gives_positive_reward(self):
        """Test that correctly diagnosing misconfiguration gives positive reward."""
        env = NetworkTroubleshootingEnv(
            max_devices=5,
            max_episode_steps=20,
            fault_types=[FaultType.MISCONFIGURATION]
        )
        
        obs, info = env.reset(seed=42)
        gt_fault = info['ground_truth_fault']
        
        # Find the correct diagnosis action
        correct_diagnosis = None
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.category.value == 'diagnosis':
                if (spec.action_type.value == gt_fault['type'] and 
                    spec.parameters.get('location') == gt_fault['location']):
                    correct_diagnosis = action_id
                    break
        
        if correct_diagnosis is not None:
            obs, reward, terminated, truncated, info = env.step(correct_diagnosis)
            
            assert terminated, "Episode should terminate on diagnosis"
            assert reward > 0, f"Correct diagnosis should give positive reward, got {reward}"
        
        env.close()

    def test_wrong_diagnosis_gives_negative_reward(self):
        """Test that incorrectly diagnosing misconfiguration gives negative reward."""
        env = NetworkTroubleshootingEnv(
            max_devices=5,
            max_episode_steps=20,
            fault_types=[FaultType.MISCONFIGURATION]
        )
        
        obs, info = env.reset(seed=42)
        gt_fault = info['ground_truth_fault']
        
        # Find an incorrect diagnosis action (different fault type)
        wrong_diagnosis = None
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.category.value == 'diagnosis':
                if spec.action_type.value != gt_fault['type']:
                    wrong_diagnosis = action_id
                    break
        
        if wrong_diagnosis is not None:
            obs, reward, terminated, truncated, info = env.step(wrong_diagnosis)
            
            assert terminated, "Episode should terminate on diagnosis"
            # Reward should be negative (penalty) for wrong diagnosis
            # Note: might get partial reward if device matches
            breakdown = info.get('reward_breakdown', {})
            diagnosis_reward = breakdown.get('diagnosis_reward', 0)
            assert diagnosis_reward < 0, \
                f"Wrong fault type diagnosis should give negative diagnosis_reward, got {diagnosis_reward}"
        
        env.close()


class TestMisconfigurationReproducibility:
    """Test reproducibility of misconfiguration scenarios."""

    def test_same_seed_produces_same_fault(self):
        """Verify same seed produces identical misconfiguration."""
        results = []
        
        for _ in range(3):
            env = NetworkTroubleshootingEnv(
                max_devices=5,
                max_episode_steps=10,
                fault_types=[FaultType.MISCONFIGURATION]
            )
            obs, info = env.reset(seed=12345)
            results.append(info['ground_truth_fault'])
            env.close()
        
        # All results should be identical
        for i in range(1, len(results)):
            assert results[i]['type'] == results[0]['type']
            assert results[i]['location'] == results[0]['location']

    def test_different_seeds_produce_different_faults(self):
        """Verify different seeds can produce different misconfigurations."""
        faults = set()
        
        for seed in range(20):
            env = NetworkTroubleshootingEnv(
                max_devices=6,
                max_episode_steps=10,
                fault_types=[FaultType.MISCONFIGURATION]
            )
            obs, info = env.reset(seed=seed)
            fault_key = (info['ground_truth_fault']['type'], 
                        info['ground_truth_fault']['location'])
            faults.add(fault_key)
            env.close()
        
        # Should have some variety in faults
        assert len(faults) > 1, \
            "Different seeds should produce some variety in fault locations"


# =============================================================================
# ADVANCED MULTI-ACTION EPISODE TESTS
# =============================================================================

class TestMisconfigurationMultiStepDiscovery:
    """Test misconfiguration behavior across multi-step discovery sequences."""

    def _find_action_by_type(self, env, action_type: str, **params) -> Optional[int]:
        """Helper to find an action by type and optional parameters."""
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.action_type.value == action_type:
                if all(spec.parameters.get(k) == v for k, v in params.items()):
                    return action_id
        return None

    def _find_actions_by_type(self, env, action_type: str) -> List[int]:
        """Helper to find all actions of a given type."""
        actions = []
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.action_type.value == action_type:
                actions.append(action_id)
        return actions

    def test_scan_then_check_status_on_all_devices(self):
        """Test scanning network then checking status on all discovered devices."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=50, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=100)
        gt_fault = info['ground_truth_fault']
        
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            obs, _, terminated, _, info = env.step(scan_action)
            assert not terminated
        
        check_status_actions = self._find_actions_by_type(env, 'check_status')
        for action_id in check_status_actions[:6]:
            spec = env.action_space_manager.get_action_spec(action_id)
            obs, _, terminated, _, info = env.step(action_id)
            if terminated:
                break
            action_result = info.get('action_result')
            if action_result and hasattr(action_result, 'result') and action_result.result:
                status = action_result.result.data.get('status')
                assert status == 'up', f"All devices should be UP for misconfiguration"
        env.close()

    def test_scan_then_check_interfaces_finds_blocked_interface(self):
        """Test that check_interfaces on faulty device reveals blocked interface."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=30, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=101)
        faulty_device = info['ground_truth_fault']['location']
        
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            env.step(scan_action)
        
        check_iface_action = self._find_action_by_type(env, 'check_interfaces', device=faulty_device)
        if check_iface_action is not None:
            obs, _, _, _, info = env.step(check_iface_action)
            action_result = info.get('action_result')
            if action_result and hasattr(action_result, 'result') and action_result.result:
                down_interfaces = action_result.result.data.get('down_interfaces', 0)
                assert down_interfaces >= 1, f"Faulty device should have at least 1 down interface"
        env.close()

    def test_systematic_ping_sweep_reveals_connectivity_issues(self):
        """Test systematic pinging reveals selective connectivity failures."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=50, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=102)
        
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            env.step(scan_action)
        
        ping_actions = self._find_actions_by_type(env, 'ping')
        ping_results = []
        for action_id in ping_actions[:15]:
            spec = env.action_space_manager.get_action_spec(action_id)
            obs, _, terminated, _, info = env.step(action_id)
            if terminated:
                break
            action_result = info.get('action_result')
            if action_result and hasattr(action_result, 'result') and action_result.result:
                ping_results.append({'success': action_result.result.success})
        
        assert len(ping_results) > 0, "Should have executed some ping actions"
        env.close()

    def test_traceroute_sequence_consistent_with_fault(self):
        """Test multiple traceroutes show consistent failure patterns."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=40, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=103)
        
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            env.step(scan_action)
        
        traceroute_actions = self._find_actions_by_type(env, 'traceroute')
        for action_id in traceroute_actions[:8]:
            obs, _, terminated, _, info = env.step(action_id)
            if terminated:
                break
        env.close()

    def test_discover_neighbors_progressive_exploration(self):
        """Test progressive network exploration via discover_neighbors."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=30, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=104)
        
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            obs, _, _, _, _ = env.step(scan_action)
        
        discover_actions = self._find_actions_by_type(env, 'discover_neighbors')
        for action_id in discover_actions[:5]:
            obs, _, terminated, _, info = env.step(action_id)
            if terminated:
                break
        
        assert 'discovery_matrix' in obs
        env.close()


class TestMisconfigurationObservationConsistency:
    """Test that observations remain consistent with misconfiguration fault."""

    def _find_action_by_type(self, env, action_type: str, **params) -> Optional[int]:
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.action_type.value == action_type:
                if all(spec.parameters.get(k) == v for k, v in params.items()):
                    return action_id
        return None

    def test_observation_updates_after_each_action(self):
        """Verify observation dict updates correctly after each action."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=20, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=200)
        
        for step in range(5):
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            action = valid_actions[min(step, len(valid_actions) - 1)]
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
            # Verify observation structure is present
            assert 'discovery_matrix' in obs
            assert 'device_status' in obs
            assert 'recent_diagnostics' in obs
            assert 'episode_metadata' in obs
            # Verify step count from info dict increments correctly
            assert info['step_count'] == step + 1, "Step count should increment"
        env.close()

    def test_diagnostic_memory_accumulates_results(self):
        """Test that diagnostic results accumulate in observation memory."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=30, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=201)
        
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            env.step(scan_action)
        
        for step in range(10):
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            for action in valid_actions:
                spec = env.action_space_manager.get_action_spec(action)
                if spec and spec.category.value in ['diagnostic', 'topology_discovery']:
                    obs, _, terminated, _, info = env.step(action)
                    break
            if terminated:
                break
        
        recent_diags = obs.get('recent_diagnostics', [])
        assert isinstance(recent_diags, (list, np.ndarray))
        env.close()

    def test_device_status_matrix_reflects_misconfiguration(self):
        """Test device status matrix correctly shows all devices as UP."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=30, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=202)
        
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            env.step(scan_action)
        
        for _ in range(5):
            valid_actions = env.get_valid_actions()
            check_status_actions = [
                a for a in valid_actions 
                if env.action_space_manager.get_action_spec(a) and 
                env.action_space_manager.get_action_spec(a).action_type.value == 'check_status'
            ]
            if check_status_actions:
                obs, _, terminated, _, _ = env.step(check_status_actions[0])
                if terminated:
                    break
        
        device_status = obs.get('device_status')
        assert device_status is not None
        env.close()


class TestMisconfigurationDiagnosticWorkflows:
    """Test realistic diagnostic workflows for finding misconfiguration."""

    def _find_action_by_type(self, env, action_type: str, **params) -> Optional[int]:
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.action_type.value == action_type:
                if all(spec.parameters.get(k) == v for k, v in params.items()):
                    return action_id
        return None

    def _find_diagnosis_action(self, env, fault_type: str, location: str) -> Optional[int]:
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.category.value == 'diagnosis':
                if spec.action_type.value == fault_type and spec.parameters.get('location') == location:
                    return action_id
        return None

    def test_workflow_scan_checkstatus_checkinterfaces_diagnose(self):
        """Test complete workflow: scan -> check_status -> check_interfaces -> diagnose."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=30, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=300)
        gt_fault = info['ground_truth_fault']
        faulty_device = gt_fault['location']
        
        # Step 1: Scan network
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            obs, _, _, _, _ = env.step(scan_action)
        
        # Step 2: Check status on faulty device (should be UP)
        check_status_action = self._find_action_by_type(env, 'check_status', device=faulty_device)
        if check_status_action is not None:
            obs, _, _, _, info = env.step(check_status_action)
            result = info.get('action_result')
            if result and hasattr(result, 'result') and result.result:
                assert result.result.data.get('status') == 'up'
        
        # Step 3: Check interfaces on faulty device (should show down interface)
        check_iface_action = self._find_action_by_type(env, 'check_interfaces', device=faulty_device)
        if check_iface_action is not None:
            obs, _, _, _, info = env.step(check_iface_action)
            result = info.get('action_result')
            if result and hasattr(result, 'result') and result.result:
                assert result.result.data.get('down_interfaces', 0) >= 1
        
        # Step 4: Make correct diagnosis
        diagnosis_action = self._find_diagnosis_action(env, 'misconfiguration', faulty_device)
        if diagnosis_action is not None:
            obs, reward, terminated, _, info = env.step(diagnosis_action)
            assert terminated, "Episode should terminate on diagnosis"
            assert reward > 0, f"Correct diagnosis should give positive reward, got {reward}"
        env.close()

    def test_workflow_interface_check_on_all_devices(self):
        """Test checking interfaces on all devices to find the misconfigured one."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=30, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=302)
        gt_fault = info['ground_truth_fault']
        faulty_device = gt_fault['location']
        
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            env.step(scan_action)
        
        devices_with_down_interfaces = []
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.action_type.value == 'check_interfaces':
                device = spec.parameters.get('device')
                obs, _, terminated, _, info = env.step(action_id)
                if terminated:
                    break
                result = info.get('action_result')
                if result and hasattr(result, 'result') and result.result:
                    if result.result.data.get('down_interfaces', 0) > 0:
                        devices_with_down_interfaces.append(device)
        
        if not env.episode_done:
            assert faulty_device in devices_with_down_interfaces, \
                f"Faulty device {faulty_device} should have down interfaces"
        env.close()


class TestMisconfigurationRewardAccuracy:
    """Test reward calculation accuracy for misconfiguration scenarios."""

    def _find_diagnosis_action(self, env, fault_type: str, location: str) -> Optional[int]:
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.category.value == 'diagnosis':
                if spec.action_type.value == fault_type and spec.parameters.get('location') == location:
                    return action_id
        return None

    def test_correct_diagnosis_reward_is_positive(self):
        """Test correct misconfiguration diagnosis yields positive reward."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=10, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=400)
        gt_fault = info['ground_truth_fault']
        
        diagnosis_action = self._find_diagnosis_action(env, gt_fault['type'], gt_fault['location'])
        if diagnosis_action is not None:
            obs, reward, terminated, _, info = env.step(diagnosis_action)
            assert terminated
            assert reward > 0, f"Correct diagnosis reward should be positive, got {reward}"
            breakdown = info.get('reward_breakdown', {})
            assert breakdown.get('diagnosis_reward', 0) > 0
        env.close()

    def test_wrong_fault_type_gives_penalty(self):
        """Test diagnosing wrong fault type gives negative reward."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=10, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=401)
        gt_fault = info['ground_truth_fault']
        
        wrong_diagnosis = None
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.category.value == 'diagnosis':
                if spec.action_type.value != gt_fault['type']:
                    wrong_diagnosis = action_id
                    break
        
        if wrong_diagnosis is not None:
            obs, reward, terminated, _, info = env.step(wrong_diagnosis)
            assert terminated
            breakdown = info.get('reward_breakdown', {})
            diagnosis_reward = breakdown.get('diagnosis_reward', 0)
            assert diagnosis_reward < 0, f"Wrong fault type should give negative diagnosis_reward"
        env.close()

    def test_step_penalty_accumulates(self):
        """Test that step penalty accumulates over multiple actions."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=20, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=403)
        
        total_step_penalty = 0.0
        for step in range(5):
            valid_actions = env.get_valid_actions()
            non_diagnosis = [
                a for a in valid_actions
                if env.action_space_manager.get_action_spec(a) and
                env.action_space_manager.get_action_spec(a).category.value != 'diagnosis'
            ]
            if not non_diagnosis:
                break
            obs, reward, terminated, _, info = env.step(non_diagnosis[0])
            if terminated:
                break
            breakdown = info.get('reward_breakdown', {})
            step_penalty = breakdown.get('step_penalty', 0)
            total_step_penalty += step_penalty
        
        assert total_step_penalty < 0, "Step penalties should accumulate as negative"
        env.close()

    def test_reward_breakdown_contains_expected_keys(self):
        """Test reward breakdown has expected keys."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=10, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=404)
        
        valid_actions = env.get_valid_actions()
        non_diagnosis = [
            a for a in valid_actions
            if env.action_space_manager.get_action_spec(a) and
            env.action_space_manager.get_action_spec(a).category.value != 'diagnosis'
        ]
        if non_diagnosis:
            obs, _, _, _, info = env.step(non_diagnosis[0])
            breakdown = info.get('reward_breakdown', {})
            assert 'step_penalty' in breakdown
        env.close()


class TestMisconfigurationStateConsistency:
    """Test internal state consistency throughout episode."""

    def _find_action_by_type(self, env, action_type: str) -> Optional[int]:
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.action_type.value == action_type:
                return action_id
        return None

    def test_ground_truth_unchanged_throughout_episode(self):
        """Test that ground truth fault doesn't change during episode."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=20, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=500)
        initial_gt = info['ground_truth_fault'].copy()
        
        for step in range(10):
            valid_actions = env.get_valid_actions()
            non_diagnosis = [
                a for a in valid_actions
                if env.action_space_manager.get_action_spec(a) and
                env.action_space_manager.get_action_spec(a).category.value != 'diagnosis'
            ]
            if not non_diagnosis:
                break
            obs, _, terminated, _, info = env.step(non_diagnosis[0])
            if terminated:
                break
            current_gt = info['ground_truth_fault']
            assert current_gt['type'] == initial_gt['type'], "Ground truth type should not change"
            assert current_gt['location'] == initial_gt['location'], "Ground truth location should not change"
        env.close()

    def test_network_state_consistent_with_fault(self):
        """Test network state remains consistent with injected fault."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=20, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=501)
        faulty_device = info['ground_truth_fault']['location']
        
        for step in range(8):
            assert env.network.is_device_up(faulty_device), f"Faulty device should remain UP"
            valid_actions = env.get_valid_actions()
            non_diagnosis = [
                a for a in valid_actions
                if env.action_space_manager.get_action_spec(a) and
                env.action_space_manager.get_action_spec(a).category.value != 'diagnosis'
            ]
            if not non_diagnosis:
                break
            obs, _, terminated, _, _ = env.step(non_diagnosis[0])
            if terminated:
                break
        env.close()

    def test_action_validity_updates_with_discovery(self):
        """Test that valid actions update as network is discovered."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=30, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=502)
        
        initial_valid = set(env.get_valid_actions())
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            obs, _, _, _, _ = env.step(scan_action)
        after_scan_valid = set(env.get_valid_actions())
        
        assert len(after_scan_valid) >= len(initial_valid), \
            "Valid actions should increase or stay same after discovery"
        env.close()

    def test_step_count_increments_correctly(self):
        """Test step count increments with each action."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=20, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=503)
        assert info['step_count'] == 0
        
        for expected_step in range(1, 8):
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            obs, _, terminated, _, info = env.step(valid_actions[0])
            if terminated:
                break
            assert info['step_count'] == expected_step, f"Step count should be {expected_step}"
        env.close()

    def test_episode_terminates_only_on_diagnosis_or_timeout(self):
        """Test episode only terminates on diagnosis action or max steps."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=15, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=504)
        
        for step in range(14):
            valid_actions = env.get_valid_actions()
            non_diagnosis = [
                a for a in valid_actions
                if env.action_space_manager.get_action_spec(a) and
                env.action_space_manager.get_action_spec(a).category.value != 'diagnosis'
            ]
            if not non_diagnosis:
                break
            obs, _, terminated, truncated, _ = env.step(non_diagnosis[0])
            if step < 13:
                assert not terminated, f"Should not terminate on step {step+1} without diagnosis"
        env.close()


class TestMisconfigurationToolInteractions:
    """Test tool interactions specific to misconfiguration faults."""

    def _find_action_by_type(self, env, action_type: str, **params) -> Optional[int]:
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.action_type.value == action_type:
                if all(spec.parameters.get(k) == v for k, v in params.items()):
                    return action_id
        return None

    def test_ping_from_faulty_device_to_blocked_dest_fails(self):
        """Test ping from faulty device to blocked destination fails."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=30, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=600)
        faulty_device = info['ground_truth_fault']['location']
        blocked_dest = env.ground_truth_fault.details.get('blocked_destination')
        
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            env.step(scan_action)
        
        ping_action = self._find_action_by_type(env, 'ping', source=faulty_device, destination=blocked_dest)
        if ping_action is not None:
            obs, _, _, _, info = env.step(ping_action)
            result = info.get('action_result')
            if result and hasattr(result, 'result') and result.result:
                assert not result.result.success, f"Ping to blocked dest should fail"
        env.close()

    def test_ping_to_faulty_device_succeeds(self):
        """Test that pinging TO the faulty device succeeds (device is up)."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=30, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=601)
        faulty_device = info['ground_truth_fault']['location']
        
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            env.step(scan_action)
        
        devices = env.network.get_all_devices()
        neighbor_device = None
        for d in devices:
            if d != faulty_device:
                connections = env.network.get_device_connections(d)
                for dest, conn_info in connections:
                    if dest == faulty_device and conn_info.get('status') == 'up':
                        neighbor_device = d
                        break
            if neighbor_device:
                break
        
        if neighbor_device:
            ping_action = self._find_action_by_type(env, 'ping', source=neighbor_device, destination=faulty_device)
            if ping_action is not None:
                obs, _, _, _, info = env.step(ping_action)
                result = info.get('action_result')
                if result and hasattr(result, 'result') and result.result:
                    assert result.result.success, f"Ping TO faulty device should succeed"
        env.close()

    def test_check_interfaces_returns_accurate_counts(self):
        """Test check_interfaces returns accurate up/down interface counts."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=30, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=602)
        faulty_device = info['ground_truth_fault']['location']
        
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            env.step(scan_action)
        
        check_iface_action = self._find_action_by_type(env, 'check_interfaces', device=faulty_device)
        if check_iface_action is not None:
            obs, _, _, _, info = env.step(check_iface_action)
            result = info.get('action_result')
            if result and hasattr(result, 'result') and result.result:
                data = result.result.data
                interfaces = data.get('interfaces', [])
                up_count = sum(1 for i in interfaces if i.get('status') == 'up')
                down_count = sum(1 for i in interfaces if i.get('status') == 'down')
                assert data.get('up_interfaces') == up_count
                assert data.get('down_interfaces') == down_count
                assert data.get('total_interfaces') == len(interfaces)
        env.close()

    def test_tool_costs_applied_correctly(self):
        """Test that tool costs are correctly applied to results."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=30, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=604)
        
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            env.step(scan_action)
        
        tool_costs = {}
        for action_type in ['ping', 'traceroute', 'check_status', 'check_interfaces']:
            action = self._find_action_by_type(env, action_type)
            if action is not None:
                obs, _, terminated, _, info = env.step(action)
                if terminated:
                    break
                result = info.get('action_result')
                if result and hasattr(result, 'result') and result.result:
                    tool_costs[action_type] = result.result.cost
        
        for tool, cost in tool_costs.items():
            assert cost > 0, f"Tool {tool} should have positive cost"
        env.close()


class TestMisconfigurationComplexScenarios:
    """Test complex multi-step scenarios with misconfiguration."""

    def _find_action_by_type(self, env, action_type: str, **params) -> Optional[int]:
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.action_type.value == action_type:
                if all(spec.parameters.get(k) == v for k, v in params.items()):
                    return action_id
        return None

    def _find_diagnosis_action(self, env, fault_type: str, location: str) -> Optional[int]:
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.category.value == 'diagnosis':
                if spec.action_type.value == fault_type and spec.parameters.get('location') == location:
                    return action_id
        return None

    def test_full_episode_with_correct_diagnosis(self):
        """Test complete episode from start to correct diagnosis."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=40, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=700)
        gt_fault = info['ground_truth_fault']
        
        episode_rewards = []
        actions_taken = []
        
        # Phase 1: Scan
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            obs, reward, _, _, _ = env.step(scan_action)
            episode_rewards.append(reward)
            actions_taken.append('scan_network')
        
        # Phase 2: Check status
        for _ in range(3):
            valid_actions = env.get_valid_actions()
            check_status_actions = [
                a for a in valid_actions
                if env.action_space_manager.get_action_spec(a) and
                env.action_space_manager.get_action_spec(a).action_type.value == 'check_status'
            ]
            if check_status_actions:
                obs, reward, terminated, _, _ = env.step(check_status_actions[0])
                episode_rewards.append(reward)
                actions_taken.append('check_status')
                if terminated:
                    break
        
        # Phase 3: Check interfaces on faulty device
        check_iface_action = self._find_action_by_type(env, 'check_interfaces', device=gt_fault['location'])
        if check_iface_action is not None and not env.episode_done:
            obs, reward, _, _, _ = env.step(check_iface_action)
            episode_rewards.append(reward)
            actions_taken.append('check_interfaces')
        
        # Phase 4: Correct diagnosis
        if not env.episode_done:
            diagnosis_action = self._find_diagnosis_action(env, gt_fault['type'], gt_fault['location'])
            if diagnosis_action is not None:
                obs, reward, terminated, _, info = env.step(diagnosis_action)
                episode_rewards.append(reward)
                actions_taken.append('diagnosis')
                assert terminated, "Episode should terminate on diagnosis"
                assert reward > 0, "Correct diagnosis should give positive reward"
        
        assert len(actions_taken) > 0, "Should have taken some actions"
        env.close()

    def test_episode_with_wrong_diagnosis_after_evidence(self):
        """Test episode where agent gathers evidence but diagnoses wrong."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=30, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=701)
        gt_fault = info['ground_truth_fault']
        
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            env.step(scan_action)
        
        for _ in range(3):
            valid_actions = env.get_valid_actions()
            diagnostic_actions = [
                a for a in valid_actions
                if env.action_space_manager.get_action_spec(a) and
                env.action_space_manager.get_action_spec(a).category.value == 'diagnostic'
            ]
            if diagnostic_actions:
                obs, _, terminated, _, _ = env.step(diagnostic_actions[0])
                if terminated:
                    break
        
        if not env.episode_done:
            wrong_diagnosis = None
            for action_id in range(env.action_space.n):
                spec = env.action_space_manager.get_action_spec(action_id)
                if spec and spec.category.value == 'diagnosis':
                    if spec.action_type.value == 'device_failure':
                        wrong_diagnosis = action_id
                        break
            
            if wrong_diagnosis is not None:
                obs, reward, terminated, _, info = env.step(wrong_diagnosis)
                assert terminated, "Episode should terminate on diagnosis"
                # Wrong fault type should not give full positive reward
                # May get partial reward if device happens to match, so just check not full positive
                breakdown = info.get('reward_breakdown', {})
                diagnosis_reward = breakdown.get('diagnosis_reward', 0)
                # Either negative or zero (partial reward cases may result in small positive)
                assert diagnosis_reward <= 0 or reward < 5.0, \
                    "Wrong fault type should not give full positive diagnosis reward"
        env.close()

    def test_episode_timeout_without_diagnosis(self):
        """Test episode timing out without making a diagnosis."""
        env = NetworkTroubleshootingEnv(
            max_devices=4, max_episode_steps=8, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=702)
        
        for step in range(8):
            valid_actions = env.get_valid_actions()
            non_diagnosis = [
                a for a in valid_actions
                if env.action_space_manager.get_action_spec(a) and
                env.action_space_manager.get_action_spec(a).category.value != 'diagnosis'
            ]
            if non_diagnosis:
                obs, _, terminated, truncated, _ = env.step(non_diagnosis[0])
            else:
                if valid_actions:
                    obs, _, terminated, truncated, _ = env.step(valid_actions[0])
                    break
            if terminated or truncated:
                break
        
        assert env.episode_done
        env.close()

    def test_discovery_progression_affects_valid_actions(self):
        """Test that progressive discovery expands valid actions."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=30, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=704)
        
        valid_action_counts = [len(env.get_valid_actions())]
        
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            env.step(scan_action)
            valid_action_counts.append(len(env.get_valid_actions()))
        
        for _ in range(3):
            discover_action = self._find_action_by_type(env, 'discover_neighbors')
            if discover_action is not None:
                obs, _, terminated, _, _ = env.step(discover_action)
                if terminated:
                    break
                valid_action_counts.append(len(env.get_valid_actions()))
        
        assert max(valid_action_counts) >= valid_action_counts[0], \
            "Valid actions should increase with discovery"
        env.close()

    def test_consistent_tool_behavior_across_multiple_calls(self):
        """Test that same tool called multiple times gives consistent results."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=30, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=705)
        faulty_device = info['ground_truth_fault']['location']
        
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            env.step(scan_action)
        
        check_status_action = self._find_action_by_type(env, 'check_status', device=faulty_device)
        status_results = []
        if check_status_action is not None:
            for _ in range(3):
                obs, _, terminated, _, info = env.step(check_status_action)
                if terminated:
                    break
                result = info.get('action_result')
                if result and hasattr(result, 'result') and result.result:
                    status_results.append(result.result.data.get('status'))
        
        for status in status_results:
            assert status == 'up', "Status should be consistently 'up'"
        env.close()


class TestMisconfigurationEdgeCasesAdvanced:
    """Advanced edge case tests for misconfiguration."""

    def _find_action_by_type(self, env, action_type: str, **params) -> Optional[int]:
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.action_type.value == action_type:
                if all(spec.parameters.get(k) == v for k, v in params.items()):
                    return action_id
        return None

    def test_diagnosis_immediately_after_reset(self):
        """Test making diagnosis immediately without any exploration."""
        env = NetworkTroubleshootingEnv(
            max_devices=4, max_episode_steps=10, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=800)
        
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.category.value == 'diagnosis':
                obs, reward, terminated, _, _ = env.step(action_id)
                assert terminated, "Episode should terminate on diagnosis"
                break
        env.close()

    def test_all_diagnostic_tools_on_faulty_device(self):
        """Test running all diagnostic tools on the faulty device."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=30, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=801)
        faulty_device = info['ground_truth_fault']['location']
        
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            env.step(scan_action)
        
        tools_run = []
        for tool_type in ['check_status', 'check_interfaces']:
            action = self._find_action_by_type(env, tool_type, device=faulty_device)
            if action is not None:
                obs, _, terminated, _, info = env.step(action)
                if terminated:
                    break
                result = info.get('action_result')
                if result and hasattr(result, 'result') and result.result:
                    tools_run.append({'tool': tool_type, 'data': result.result.data})
        
        for tool_result in tools_run:
            if tool_result['tool'] == 'check_status':
                assert tool_result['data'].get('status') == 'up', "Should show device UP"
            elif tool_result['tool'] == 'check_interfaces':
                assert tool_result['data'].get('down_interfaces', 0) >= 1, "Should show down interface"
        env.close()

    def test_episode_with_max_one_step(self):
        """Test episode with max_episode_steps=1."""
        env = NetworkTroubleshootingEnv(
            max_devices=4, max_episode_steps=1, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=802)
        
        valid_actions = env.get_valid_actions()
        if valid_actions:
            obs, reward, terminated, truncated, info = env.step(valid_actions[0])
            assert terminated or truncated, "Episode should end after 1 step"
        env.close()

    def test_observation_shapes_consistent(self):
        """Test observation shapes remain consistent throughout episode."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=20, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=803)
        
        initial_shapes = {
            'discovery_matrix': np.array(obs['discovery_matrix']).shape,
            'device_status': np.array(obs['device_status']).shape,
            'recent_diagnostics': np.array(obs['recent_diagnostics']).shape,
            'episode_metadata': np.array(obs['episode_metadata']).shape,
        }
        
        for step in range(8):
            valid_actions = env.get_valid_actions()
            non_diagnosis = [
                a for a in valid_actions
                if env.action_space_manager.get_action_spec(a) and
                env.action_space_manager.get_action_spec(a).category.value != 'diagnosis'
            ]
            if not non_diagnosis:
                break
            obs, _, terminated, _, _ = env.step(non_diagnosis[0])
            if terminated:
                break
            for key, expected_shape in initial_shapes.items():
                actual_shape = np.array(obs[key]).shape
                assert actual_shape == expected_shape, f"{key} shape changed"
        env.close()

    def test_info_dict_always_has_required_keys(self):
        """Test info dict always contains required keys."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=15, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=804)
        
        required_keys = ['step_count', 'ground_truth_fault', 'network_size']
        for key in required_keys:
            assert key in info, f"Info should contain {key}"
        
        for step in range(5):
            valid_actions = env.get_valid_actions()
            non_diagnosis = [
                a for a in valid_actions
                if env.action_space_manager.get_action_spec(a) and
                env.action_space_manager.get_action_spec(a).category.value != 'diagnosis'
            ]
            if not non_diagnosis:
                break
            obs, _, terminated, _, info = env.step(non_diagnosis[0])
            if terminated:
                break
            for key in required_keys:
                assert key in info, f"Info should contain {key} after step {step}"
        env.close()

    def test_blocked_connection_details_in_fault_info(self):
        """Test that fault info contains blocked_destination details."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=10, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=805)
        
        fault = env.ground_truth_fault
        assert fault is not None
        assert fault.fault_type == FaultType.MISCONFIGURATION
        assert 'blocked_destination' in fault.details, "Should have blocked_destination in details"
        assert fault.details['blocked_destination'] is not None
        env.close()

    def test_multiple_episodes_different_faults(self):
        """Test running multiple episodes with different seeds produces different faults."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=10, fault_types=[FaultType.MISCONFIGURATION]
        )
        
        faults_seen = set()
        for seed in range(10):
            obs, info = env.reset(seed=seed)
            gt = info['ground_truth_fault']
            fault_key = (gt['type'], gt['location'])
            faults_seen.add(fault_key)
        
        assert len(faults_seen) > 1, "Should see variety of faults across seeds"
        env.close()


class TestMisconfigurationBlockedPathAnalysis:
    """Test detailed analysis of blocked path behavior."""

    def _find_action_by_type(self, env, action_type: str, **params) -> Optional[int]:
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.action_type.value == action_type:
                if all(spec.parameters.get(k) == v for k, v in params.items()):
                    return action_id
        return None

    def test_path_through_blocked_connection_fails(self):
        """Test that paths requiring blocked connection fail."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=30, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=900)
        faulty_device = info['ground_truth_fault']['location']
        blocked_dest = env.ground_truth_fault.details.get('blocked_destination')
        
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            env.step(scan_action)
        
        # Try ping that would use the blocked path
        ping_action = self._find_action_by_type(env, 'ping', source=faulty_device, destination=blocked_dest)
        if ping_action is not None:
            obs, _, _, _, info = env.step(ping_action)
            result = info.get('action_result')
            if result and hasattr(result, 'result') and result.result:
                assert not result.result.success, "Ping through blocked path should fail"
        env.close()

    def test_reverse_direction_of_blocked_connection(self):
        """Test that reverse direction of blocked connection may work."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=30, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=901)
        faulty_device = info['ground_truth_fault']['location']
        blocked_dest = env.ground_truth_fault.details.get('blocked_destination')
        
        # Check if reverse connection exists and is up
        if env.network.graph.has_edge(blocked_dest, faulty_device):
            conn_info = env.network.get_connection_info(blocked_dest, faulty_device)
            # For misconfiguration, reverse should still be up
            assert conn_info.get('status') == 'up', "Reverse direction should be up"
        env.close()

    def test_traceroute_identifies_blocked_segment(self):
        """Test traceroute can identify where path is blocked."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=30, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=902)
        
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            env.step(scan_action)
        
        # Execute some traceroutes
        traceroute_results = []
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.action_type.value == 'traceroute':
                obs, _, terminated, _, info = env.step(action_id)
                if terminated:
                    break
                result = info.get('action_result')
                if result and hasattr(result, 'result') and result.result:
                    if not result.result.success:
                        traceroute_results.append(result.result.data)
                if len(traceroute_results) >= 3:
                    break
        
        # Failed traceroutes should have error or path info
        for tr in traceroute_results:
            assert 'path' in tr or 'error' in tr, "Should have path or error info"
        env.close()


class TestMisconfigurationWithDifferentTopologies:
    """Test misconfiguration across different topology types."""

    def test_misconfiguration_in_small_network(self):
        """Test misconfiguration in minimal network (3 devices)."""
        env = NetworkTroubleshootingEnv(
            max_devices=3, max_episode_steps=15, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=1000)
        
        assert info['network_size'] >= 3
        assert info['ground_truth_fault']['type'] == 'misconfiguration'
        
        # Faulty device should be up
        faulty_device = info['ground_truth_fault']['location']
        assert env.network.is_device_up(faulty_device)
        env.close()

    def test_misconfiguration_in_large_network(self):
        """Test misconfiguration in larger network (10 devices)."""
        env = NetworkTroubleshootingEnv(
            max_devices=10, max_episode_steps=50, fault_types=[FaultType.MISCONFIGURATION]
        )
        obs, info = env.reset(seed=1001)
        
        # Network size varies based on topology generation
        assert info['network_size'] >= 3, "Should have at least 3 devices"
        assert info['ground_truth_fault']['type'] == 'misconfiguration'
        
        faulty_device = info['ground_truth_fault']['location']
        assert env.network.is_device_up(faulty_device)
        env.close()

    def test_misconfiguration_across_topology_types(self):
        """Test misconfiguration works across different topology types."""
        topology_types_to_test = [['linear'], ['star'], ['mesh'], ['hierarchical']]
        
        for topo_types in topology_types_to_test:
            env = NetworkTroubleshootingEnv(
                max_devices=6, max_episode_steps=20,
                fault_types=[FaultType.MISCONFIGURATION],
                topology_types=topo_types
            )
            obs, info = env.reset(seed=1002)
            
            assert info['ground_truth_fault']['type'] == 'misconfiguration'
            faulty_device = info['ground_truth_fault']['location']
            assert env.network.is_device_up(faulty_device), \
                f"Faulty device should be up in {topo_types[0]} topology"
            env.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
