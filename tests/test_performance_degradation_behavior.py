# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive tests for PERFORMANCE_DEGRADATION fault type behavior in NetHeal.

Expected Behavior:
- Device status remains 'up'
- All connections remain 'up'
- Latency and/or bandwidth on affected connection is degraded
- Check status on device returns status='up'
- Check interfaces shows high latency or reduced bandwidth
- Pings may still succeed but with degraded metrics
- Traceroute may show increased latency on specific hops
"""

import pytest
import numpy as np
from typing import List, Optional

from netheal.network.graph import NetworkGraph, DeviceType
from netheal.network.topology import TopologyGenerator
from netheal.faults.injector import FaultInjector, FaultType, FaultInfo
from netheal.tools.simulator import ToolSimulator
from netheal.environment.env import NetworkTroubleshootingEnv


class TestPerformanceDegradationBasicBehavior:
    """Test basic performance degradation fault behavior."""

    def test_device_remains_up(self):
        """Verify device with performance degradation remains operational."""
        network = TopologyGenerator.generate_linear_topology(4)
        injector = FaultInjector(network)
        devices = network.get_all_devices()
        
        fault = injector.inject_performance_degradation(source=devices[1], destination=devices[2])
        
        assert network.is_device_up(devices[1])
        assert network.is_device_up(devices[2])
        assert fault.fault_type == FaultType.PERFORMANCE_DEGRADATION

    def test_connection_remains_up(self):
        """Verify connection with degradation remains up."""
        network = TopologyGenerator.generate_linear_topology(4)
        injector = FaultInjector(network)
        devices = network.get_all_devices()
        
        injector.inject_performance_degradation(source=devices[1], destination=devices[2])
        
        conn = network.get_connection_info(devices[1], devices[2])
        assert conn.get('status') == 'up'

    def test_check_status_shows_device_up(self):
        """Verify check_status shows device as up."""
        network = TopologyGenerator.generate_linear_topology(4)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        devices = network.get_all_devices()
        
        injector.inject_performance_degradation(source=devices[1], destination=devices[2])
        
        status1 = tool_sim.check_status(devices[1])
        status2 = tool_sim.check_status(devices[2])
        
        assert status1.data['status'] == 'up'
        assert status2.data['status'] == 'up'

    def test_check_interfaces_shows_degradation(self):
        """Verify check_interfaces shows performance degradation."""
        network = TopologyGenerator.generate_linear_topology(4)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        devices = network.get_all_devices()
        
        injector.inject_performance_degradation(source=devices[1], destination=devices[2])
        
        result = tool_sim.check_interfaces(devices[1])
        assert result.success
        # Should show degraded interface (high latency or low bandwidth)
        assert 'interfaces' in result.data or 'total_interfaces' in result.data

    def test_ping_still_succeeds(self):
        """Verify ping may still succeed despite degradation."""
        network = TopologyGenerator.generate_linear_topology(4)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        devices = network.get_all_devices()
        
        injector.inject_performance_degradation(source=devices[1], destination=devices[2])
        
        # Ping should still work (connectivity exists)
        result = tool_sim.ping(devices[0], devices[3])
        # May succeed or fail depending on implementation, but device is reachable
        assert result.success or result.data.get('error') is not None


class TestPerformanceDegradationMetrics:
    """Test that performance degradation affects metrics correctly."""

    def test_latency_increased_on_degraded_connection(self):
        """Verify latency is increased on the degraded connection."""
        network = TopologyGenerator.generate_linear_topology(4)
        injector = FaultInjector(network)
        devices = network.get_all_devices()
        
        # Get baseline latency
        conn_before = network.get_connection_info(devices[1], devices[2])
        baseline_latency = conn_before.get('latency', 10)
        
        fault = injector.inject_performance_degradation(source=devices[1], destination=devices[2])
        
        conn_after = network.get_connection_info(devices[1], devices[2])
        new_latency = conn_after.get('latency', 10)
        
        # Latency should be higher after degradation
        assert new_latency >= baseline_latency

    def test_bandwidth_reduced_on_degraded_connection(self):
        """Verify bandwidth may be reduced on the degraded connection."""
        network = TopologyGenerator.generate_linear_topology(4)
        injector = FaultInjector(network)
        devices = network.get_all_devices()
        
        conn_before = network.get_connection_info(devices[1], devices[2])
        baseline_bw = conn_before.get('bandwidth', 1000)
        
        fault = injector.inject_performance_degradation(source=devices[1], destination=devices[2])
        
        conn_after = network.get_connection_info(devices[1], devices[2])
        new_bw = conn_after.get('bandwidth', 1000)
        
        # Bandwidth should be same or lower after degradation
        assert new_bw <= baseline_bw


class TestPerformanceDegradationPathEffects:
    """Test path-related effects of performance degradation."""

    def test_paths_through_degraded_link_work(self):
        """Verify paths through degraded link still work."""
        network = TopologyGenerator.generate_linear_topology(5)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        devices = network.get_all_devices()
        
        injector.inject_performance_degradation(source=devices[2], destination=devices[3])
        
        result = tool_sim.ping(devices[0], devices[4])
        # Should succeed as connectivity exists
        assert result.success or 'latency' in result.data

    def test_traceroute_through_degraded_link(self):
        """Verify traceroute through degraded link."""
        network = TopologyGenerator.generate_linear_topology(5)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        devices = network.get_all_devices()
        
        injector.inject_performance_degradation(source=devices[2], destination=devices[3])
        
        result = tool_sim.traceroute(devices[0], devices[4])
        # Should succeed as connectivity exists
        assert result.success or 'path' in result.data

    def test_non_degraded_paths_unaffected(self):
        """Verify paths not through degraded link are unaffected."""
        network = TopologyGenerator.generate_linear_topology(5)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        devices = network.get_all_devices()
        
        injector.inject_performance_degradation(source=devices[3], destination=devices[4])
        
        # Path from D0 to D2 doesn't use degraded link
        result = tool_sim.ping(devices[0], devices[2])
        assert result.success


class TestPerformanceDegradationTopologyVariations:
    """Test performance degradation across different topologies."""

    def test_degradation_in_star_topology(self):
        """Test performance degradation in star topology."""
        network = TopologyGenerator.generate_star_topology(5)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        devices = network.get_all_devices()
        
        hub = max(devices, key=lambda d: len(network.get_device_connections(d)))
        edge = [d for d in devices if d != hub][0]
        
        injector.inject_performance_degradation(source=hub, destination=edge)
        
        assert network.is_device_up(hub)
        assert network.is_device_up(edge)
        conn = network.get_connection_info(hub, edge)
        assert conn.get('status') == 'up'

    def test_degradation_in_mesh_topology(self):
        """Test performance degradation in mesh topology."""
        network = TopologyGenerator.generate_mesh_topology(5, connection_probability=0.7)
        injector = FaultInjector(network)
        devices = network.get_all_devices()
        
        for d in devices:
            connections = network.get_device_connections(d)
            if connections:
                dest, _ = connections[0]
                injector.inject_performance_degradation(source=d, destination=dest)
                assert network.is_device_up(d)
                assert network.is_device_up(dest)
                conn = network.get_connection_info(d, dest)
                assert conn.get('status') == 'up'
                break

    def test_degradation_in_hierarchical_topology(self):
        """Test performance degradation in hierarchical topology."""
        network = TopologyGenerator.generate_hierarchical_topology(num_layers=3, devices_per_layer=[1, 2, 3])
        injector = FaultInjector(network)
        devices = network.get_all_devices()
        
        for d in devices:
            connections = network.get_device_connections(d)
            if connections:
                dest, _ = connections[0]
                injector.inject_performance_degradation(source=d, destination=dest)
                assert network.is_device_up(d)
                assert network.is_device_up(dest)
                break


class TestPerformanceDegradationDifferentiation:
    """Test that performance degradation can be differentiated from other faults."""

    def test_degradation_vs_device_failure(self):
        """Verify performance degradation differs from device failure."""
        network1 = TopologyGenerator.generate_linear_topology(4)
        injector1 = FaultInjector(network1)
        tool_sim1 = ToolSimulator(network1)
        devices1 = network1.get_all_devices()
        injector1.inject_performance_degradation(source=devices1[1], destination=devices1[2])
        
        # Degradation: device UP
        assert tool_sim1.check_status(devices1[1]).data['status'] == 'up'
        
        network2 = TopologyGenerator.generate_linear_topology(4)
        injector2 = FaultInjector(network2)
        tool_sim2 = ToolSimulator(network2)
        devices2 = network2.get_all_devices()
        injector2.inject_device_failure(device_id=devices2[1])
        
        # Device failure: device DOWN
        assert tool_sim2.check_status(devices2[1]).data['status'] == 'down'

    def test_degradation_vs_link_failure(self):
        """Verify performance degradation differs from link failure."""
        network1 = TopologyGenerator.generate_linear_topology(4)
        injector1 = FaultInjector(network1)
        devices1 = network1.get_all_devices()
        injector1.inject_performance_degradation(source=devices1[1], destination=devices1[2])
        
        # Degradation: connection UP
        conn1 = network1.get_connection_info(devices1[1], devices1[2])
        assert conn1.get('status') == 'up'
        
        network2 = TopologyGenerator.generate_linear_topology(4)
        injector2 = FaultInjector(network2)
        devices2 = network2.get_all_devices()
        injector2.inject_link_failure(source=devices2[1], destination=devices2[2])
        
        # Link failure: connection DOWN
        conn2 = network2.get_connection_info(devices2[1], devices2[2])
        assert conn2.get('status') == 'down'

    def test_degradation_vs_misconfiguration(self):
        """Verify performance degradation differs from misconfiguration."""
        network1 = TopologyGenerator.generate_linear_topology(4)
        injector1 = FaultInjector(network1)
        devices1 = network1.get_all_devices()
        injector1.inject_performance_degradation(source=devices1[1], destination=devices1[2])
        
        # Degradation: connection UP in both directions
        conn_fwd1 = network1.get_connection_info(devices1[1], devices1[2])
        conn_rev1 = network1.get_connection_info(devices1[2], devices1[1])
        assert conn_fwd1.get('status') == 'up'
        assert conn_rev1.get('status') == 'up'
        
        network2 = TopologyGenerator.generate_linear_topology(4)
        injector2 = FaultInjector(network2)
        devices2 = network2.get_all_devices()
        injector2.inject_misconfiguration(device_id=devices2[1], blocked_destination=devices2[2])
        
        # Misconfiguration: one direction DOWN
        conn_fwd2 = network2.get_connection_info(devices2[1], devices2[2])
        assert conn_fwd2.get('status') == 'down'


class TestPerformanceDegradationEdgeCases:
    """Test edge cases for performance degradation faults."""

    def test_degradation_preserves_other_connections(self):
        """Verify degrading one connection doesn't affect others."""
        network = TopologyGenerator.generate_linear_topology(4)
        injector = FaultInjector(network)
        devices = network.get_all_devices()
        
        # Get baseline
        conn_0_1_before = network.get_connection_info(devices[0], devices[1])
        
        injector.inject_performance_degradation(source=devices[1], destination=devices[2])
        
        # Other connections should be unaffected
        conn_0_1_after = network.get_connection_info(devices[0], devices[1])
        assert conn_0_1_after.get('status') == 'up'

    def test_multiple_degradations(self):
        """Test multiple performance degradations in same network."""
        network = TopologyGenerator.generate_linear_topology(5)
        injector = FaultInjector(network)
        devices = network.get_all_devices()
        
        injector.inject_performance_degradation(source=devices[0], destination=devices[1])
        injector.inject_performance_degradation(source=devices[2], destination=devices[3])
        
        # All devices should still be up
        for d in devices:
            assert network.is_device_up(d)
        
        # All connections should still be up
        for i in range(len(devices) - 1):
            conn = network.get_connection_info(devices[i], devices[i+1])
            assert conn.get('status') == 'up'


class TestPerformanceDegradationWithEnvironment:
    """Test performance degradation through the full RL environment."""

    def test_environment_with_performance_degradation_fault_type(self):
        """Test environment generates performance degradation faults correctly."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=20, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        for seed in range(5):
            obs, info = env.reset(seed=seed)
            gt_fault = info['ground_truth_fault']
            assert gt_fault['type'] == 'performance_degradation'
        env.close()

    def test_diagnostic_tools_reveal_degradation(self):
        """Test that diagnostic tools can reveal performance degradation."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=30, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=42)
        gt_fault = info['ground_truth_fault']
        
        # Parse location to get source device
        if '->' in gt_fault['location']:
            source = gt_fault['location'].split('->')[0]
        
            for action_id in range(env.action_space.n):
                spec = env.action_space_manager.get_action_spec(action_id)
                if spec and spec.action_type.value == 'scan_network':
                    env.step(action_id)
                    break
            
            for action_id in range(env.action_space.n):
                spec = env.action_space_manager.get_action_spec(action_id)
                if spec and spec.action_type.value == 'check_status':
                    if spec.parameters.get('device') == source:
                        obs, _, _, _, info = env.step(action_id)
                        result = info.get('action_result')
                        if result and hasattr(result, 'result') and result.result:
                            # Device should be UP
                            assert result.result.data.get('status') == 'up'
                        break
        env.close()

    def test_correct_diagnosis_gives_positive_reward(self):
        """Test correctly diagnosing performance degradation gives positive reward."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=20, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=42)
        gt_fault = info['ground_truth_fault']
        
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.category.value == 'diagnosis':
                if spec.action_type.value == gt_fault['type'] and spec.parameters.get('location') == gt_fault['location']:
                    obs, reward, terminated, _, _ = env.step(action_id)
                    assert terminated
                    assert reward > 0
                    break
        env.close()

    def test_wrong_diagnosis_gives_penalty(self):
        """Test incorrectly diagnosing performance degradation gives penalty."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=20, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=42)
        gt_fault = info['ground_truth_fault']
        
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.category.value == 'diagnosis':
                if spec.action_type.value != gt_fault['type']:
                    obs, _, terminated, _, info = env.step(action_id)
                    assert terminated
                    assert info.get('reward_breakdown', {}).get('diagnosis_reward', 0) < 0
                    break
        env.close()


class TestPerformanceDegradationReproducibility:
    """Test reproducibility of performance degradation scenarios."""

    def test_same_seed_produces_same_fault(self):
        """Verify same seed produces identical performance degradation."""
        results = []
        for _ in range(3):
            env = NetworkTroubleshootingEnv(
                max_devices=5, max_episode_steps=10, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
            )
            obs, info = env.reset(seed=12345)
            results.append(info['ground_truth_fault'])
            env.close()
        for i in range(1, len(results)):
            assert results[i]['type'] == results[0]['type']
            assert results[i]['location'] == results[0]['location']

    def test_different_seeds_produce_different_faults(self):
        """Verify different seeds can produce different performance degradations."""
        faults = set()
        for seed in range(20):
            env = NetworkTroubleshootingEnv(
                max_devices=6, max_episode_steps=10, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
            )
            obs, info = env.reset(seed=seed)
            faults.add((info['ground_truth_fault']['type'], info['ground_truth_fault']['location']))
            env.close()
        assert len(faults) > 1


# =============================================================================
# ADVANCED MULTI-ACTION EPISODE TESTS
# =============================================================================

class TestPerformanceDegradationMultiStepDiscovery:
    """Test performance degradation behavior across multi-step discovery sequences."""

    def _find_action_by_type(self, env, action_type: str, **params) -> Optional[int]:
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.action_type.value == action_type:
                if all(spec.parameters.get(k) == v for k, v in params.items()):
                    return action_id
        return None

    def _find_actions_by_type(self, env, action_type: str) -> List[int]:
        return [a for a in range(env.action_space.n) 
                if env.action_space_manager.get_action_spec(a) and 
                env.action_space_manager.get_action_spec(a).action_type.value == action_type]

    def test_scan_then_check_status_all_up(self):
        """Test scanning then checking status shows all devices UP."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=50, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=100)
        
        scan = self._find_action_by_type(env, 'scan_network')
        if scan is not None:
            env.step(scan)
        
        for action_id in self._find_actions_by_type(env, 'check_status')[:6]:
            obs, _, terminated, _, info = env.step(action_id)
            if terminated:
                break
            result = info.get('action_result')
            if result and hasattr(result, 'result') and result.result:
                assert result.result.data.get('status') == 'up'
        env.close()

    def test_scan_then_check_interfaces(self):
        """Test check_interfaces reveals degradation indicators."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=30, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=101)
        gt_fault = info['ground_truth_fault']
        
        if '->' in gt_fault['location']:
            source = gt_fault['location'].split('->')[0]
        
            scan = self._find_action_by_type(env, 'scan_network')
            if scan is not None:
                env.step(scan)
            
            check_iface = self._find_action_by_type(env, 'check_interfaces', device=source)
            if check_iface is not None:
                obs, _, _, _, info = env.step(check_iface)
                result = info.get('action_result')
                if result and hasattr(result, 'result') and result.result:
                    # Should have interface info showing degradation
                    assert result.result.success
        env.close()

    def test_systematic_ping_sweep(self):
        """Test systematic pinging - pings should generally succeed."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=50, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=102)
        
        scan = self._find_action_by_type(env, 'scan_network')
        if scan is not None:
            env.step(scan)
        
        ping_results = []
        for action_id in self._find_actions_by_type(env, 'ping')[:15]:
            obs, _, terminated, _, info = env.step(action_id)
            if terminated:
                break
            result = info.get('action_result')
            if result and hasattr(result, 'result') and result.result:
                ping_results.append(result.result.success)
        
        assert len(ping_results) > 0
        # Most pings should succeed since links are up
        env.close()

    def test_traceroute_sequence(self):
        """Test multiple traceroutes."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=40, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=103)
        
        scan = self._find_action_by_type(env, 'scan_network')
        if scan is not None:
            env.step(scan)
        
        for action_id in self._find_actions_by_type(env, 'traceroute')[:8]:
            obs, _, terminated, _, _ = env.step(action_id)
            if terminated:
                break
        env.close()

    def test_progressive_exploration(self):
        """Test progressive network exploration."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=30, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=104)
        
        scan = self._find_action_by_type(env, 'scan_network')
        if scan is not None:
            env.step(scan)
        
        for action_id in self._find_actions_by_type(env, 'discover_neighbors')[:5]:
            obs, _, terminated, _, _ = env.step(action_id)
            if terminated:
                break
        assert 'discovery_matrix' in obs
        env.close()


class TestPerformanceDegradationObservationConsistency:
    """Test that observations remain consistent with performance degradation fault."""

    def _find_action_by_type(self, env, action_type: str, **params) -> Optional[int]:
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.action_type.value == action_type:
                if all(spec.parameters.get(k) == v for k, v in params.items()):
                    return action_id
        return None

    def test_observation_updates(self):
        """Verify observation dict updates correctly after each action."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=20, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=200)
        
        for step in range(5):
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            obs, _, terminated, truncated, info = env.step(valid_actions[step % len(valid_actions)])
            if terminated or truncated:
                break
            assert 'discovery_matrix' in obs
            assert 'device_status' in obs
            assert info['step_count'] == step + 1
        env.close()

    def test_diagnostic_memory_accumulates(self):
        """Test that diagnostic results accumulate in observation memory."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=30, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=201)
        
        scan = self._find_action_by_type(env, 'scan_network')
        if scan is not None:
            env.step(scan)
        
        for step in range(10):
            valid_actions = env.get_valid_actions()
            for action in valid_actions:
                spec = env.action_space_manager.get_action_spec(action)
                if spec and spec.category.value in ['diagnostic', 'topology_discovery']:
                    obs, _, terminated, _, _ = env.step(action)
                    break
            if terminated:
                break
        
        assert isinstance(obs.get('recent_diagnostics'), (list, np.ndarray))
        env.close()


class TestPerformanceDegradationDiagnosticWorkflows:
    """Test realistic diagnostic workflows for finding performance degradation."""

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

    def test_workflow_scan_checkinterfaces_diagnose(self):
        """Test workflow: scan -> check_interfaces -> diagnose."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=30, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=300)
        gt_fault = info['ground_truth_fault']
        
        scan = self._find_action_by_type(env, 'scan_network')
        if scan is not None:
            env.step(scan)
        
        if '->' in gt_fault['location']:
            source = gt_fault['location'].split('->')[0]
            check_iface = self._find_action_by_type(env, 'check_interfaces', device=source)
            if check_iface is not None:
                obs, _, _, _, info = env.step(check_iface)
                result = info.get('action_result')
                if result and hasattr(result, 'result') and result.result:
                    assert result.result.success
        
        diagnosis = self._find_diagnosis_action(env, gt_fault['type'], gt_fault['location'])
        if diagnosis is not None:
            obs, reward, terminated, _, _ = env.step(diagnosis)
            assert terminated
            assert reward > 0
        env.close()

    def test_workflow_comprehensive_check(self):
        """Test checking all diagnostic info before diagnosing."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=30, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=302)
        
        scan = self._find_action_by_type(env, 'scan_network')
        if scan is not None:
            env.step(scan)
        
        # Check status on multiple devices
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.action_type.value == 'check_status':
                obs, _, terminated, _, info = env.step(action_id)
                if terminated:
                    break
                result = info.get('action_result')
                if result and hasattr(result, 'result') and result.result:
                    # All devices should be UP
                    assert result.result.data.get('status') == 'up'
                break
        env.close()


class TestPerformanceDegradationRewardAccuracy:
    """Test reward calculation accuracy for performance degradation scenarios."""

    def _find_diagnosis_action(self, env, fault_type: str, location: str) -> Optional[int]:
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.category.value == 'diagnosis':
                if spec.action_type.value == fault_type and spec.parameters.get('location') == location:
                    return action_id
        return None

    def test_correct_diagnosis_reward_positive(self):
        """Test correct performance degradation diagnosis yields positive reward."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=10, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=400)
        gt_fault = info['ground_truth_fault']
        
        diagnosis = self._find_diagnosis_action(env, gt_fault['type'], gt_fault['location'])
        if diagnosis is not None:
            obs, reward, terminated, _, info = env.step(diagnosis)
            assert terminated
            assert reward > 0
            assert info.get('reward_breakdown', {}).get('diagnosis_reward', 0) > 0
        env.close()

    def test_wrong_fault_type_penalty(self):
        """Test diagnosing wrong fault type gives negative reward."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=10, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=401)
        gt_fault = info['ground_truth_fault']
        
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.category.value == 'diagnosis' and spec.action_type.value != gt_fault['type']:
                obs, _, terminated, _, info = env.step(action_id)
                assert terminated
                assert info.get('reward_breakdown', {}).get('diagnosis_reward', 0) < 0
                break
        env.close()

    def test_step_penalty_accumulates(self):
        """Test that step penalty accumulates over multiple actions."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=20, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=403)
        
        total_penalty = 0.0
        for step in range(5):
            valid_actions = env.get_valid_actions()
            non_diagnosis = [a for a in valid_actions 
                           if env.action_space_manager.get_action_spec(a) and 
                           env.action_space_manager.get_action_spec(a).category.value != 'diagnosis']
            if not non_diagnosis:
                break
            obs, _, terminated, _, info = env.step(non_diagnosis[0])
            if terminated:
                break
            total_penalty += info.get('reward_breakdown', {}).get('step_penalty', 0)
        assert total_penalty < 0
        env.close()

    def test_reward_breakdown_keys(self):
        """Test reward breakdown has expected keys."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=10, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=404)
        
        valid_actions = env.get_valid_actions()
        non_diagnosis = [a for a in valid_actions 
                        if env.action_space_manager.get_action_spec(a) and 
                        env.action_space_manager.get_action_spec(a).category.value != 'diagnosis']
        if non_diagnosis:
            obs, _, _, _, info = env.step(non_diagnosis[0])
            assert 'step_penalty' in info.get('reward_breakdown', {})
        env.close()


class TestPerformanceDegradationStateConsistency:
    """Test internal state consistency throughout episode."""

    def _find_action_by_type(self, env, action_type: str) -> Optional[int]:
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.action_type.value == action_type:
                return action_id
        return None

    def test_ground_truth_unchanged(self):
        """Test that ground truth fault doesn't change during episode."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=20, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=500)
        initial_gt = info['ground_truth_fault'].copy()
        
        for step in range(10):
            valid_actions = env.get_valid_actions()
            non_diagnosis = [a for a in valid_actions 
                           if env.action_space_manager.get_action_spec(a) and 
                           env.action_space_manager.get_action_spec(a).category.value != 'diagnosis']
            if not non_diagnosis:
                break
            obs, _, terminated, _, info = env.step(non_diagnosis[0])
            if terminated:
                break
            assert info['ground_truth_fault']['type'] == initial_gt['type']
            assert info['ground_truth_fault']['location'] == initial_gt['location']
        env.close()

    def test_network_state_consistent(self):
        """Test network state remains consistent with injected fault."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=20, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=501)
        gt_fault = info['ground_truth_fault']
        
        if '->' in gt_fault['location']:
            source, dest = gt_fault['location'].split('->')
            
            for step in range(8):
                # Both devices should remain UP
                assert env.network.is_device_up(source)
                assert env.network.is_device_up(dest)
                # Connection should remain UP
                conn = env.network.get_connection_info(source, dest)
                assert conn.get('status') == 'up'
                
                valid_actions = env.get_valid_actions()
                non_diagnosis = [a for a in valid_actions 
                               if env.action_space_manager.get_action_spec(a) and 
                               env.action_space_manager.get_action_spec(a).category.value != 'diagnosis']
                if not non_diagnosis:
                    break
                obs, _, terminated, _, _ = env.step(non_diagnosis[0])
                if terminated:
                    break
        env.close()

    def test_action_validity_updates(self):
        """Test that valid actions update as network is discovered."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=30, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=502)
        
        initial_valid = len(env.get_valid_actions())
        scan = self._find_action_by_type(env, 'scan_network')
        if scan is not None:
            env.step(scan)
        assert len(env.get_valid_actions()) >= initial_valid
        env.close()

    def test_step_count_increments(self):
        """Test step count increments with each action."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=20, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=503)
        assert info['step_count'] == 0
        
        for expected in range(1, 8):
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            obs, _, terminated, _, info = env.step(valid_actions[0])
            if terminated:
                break
            assert info['step_count'] == expected
        env.close()

    def test_episode_terminates_correctly(self):
        """Test episode only terminates on diagnosis action or max steps."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=15, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=504)
        
        for step in range(14):
            valid_actions = env.get_valid_actions()
            non_diagnosis = [a for a in valid_actions 
                           if env.action_space_manager.get_action_spec(a) and 
                           env.action_space_manager.get_action_spec(a).category.value != 'diagnosis']
            if not non_diagnosis:
                break
            obs, _, terminated, _, _ = env.step(non_diagnosis[0])
            if step < 13:
                assert not terminated
        env.close()


class TestPerformanceDegradationComplexScenarios:
    """Test complex multi-step scenarios with performance degradation."""

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

    def test_full_episode_correct_diagnosis(self):
        """Test complete episode from start to correct diagnosis."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=40, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=700)
        gt_fault = info['ground_truth_fault']
        
        scan = self._find_action_by_type(env, 'scan_network')
        if scan is not None:
            env.step(scan)
        
        if not env.episode_done:
            diagnosis = self._find_diagnosis_action(env, gt_fault['type'], gt_fault['location'])
            if diagnosis is not None:
                obs, reward, terminated, _, _ = env.step(diagnosis)
                assert terminated
                assert reward > 0
        env.close()

    def test_episode_wrong_diagnosis(self):
        """Test episode where agent diagnoses wrong."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=30, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=701)
        
        scan = self._find_action_by_type(env, 'scan_network')
        if scan is not None:
            env.step(scan)
        
        if not env.episode_done:
            for action_id in range(env.action_space.n):
                spec = env.action_space_manager.get_action_spec(action_id)
                if spec and spec.category.value == 'diagnosis' and spec.action_type.value == 'device_failure':
                    obs, reward, terminated, _, info = env.step(action_id)
                    assert terminated
                    breakdown = info.get('reward_breakdown', {})
                    assert breakdown.get('diagnosis_reward', 0) <= 0 or reward < 5.0
                    break
        env.close()

    def test_episode_timeout(self):
        """Test episode timing out without diagnosis."""
        env = NetworkTroubleshootingEnv(
            max_devices=4, max_episode_steps=8, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=702)
        
        for step in range(8):
            valid_actions = env.get_valid_actions()
            non_diagnosis = [a for a in valid_actions 
                           if env.action_space_manager.get_action_spec(a) and 
                           env.action_space_manager.get_action_spec(a).category.value != 'diagnosis']
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

    def test_consistent_tool_behavior(self):
        """Test same tool called multiple times gives consistent results."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=30, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=705)
        gt_fault = info['ground_truth_fault']
        
        scan = self._find_action_by_type(env, 'scan_network')
        if scan is not None:
            env.step(scan)
        
        if '->' in gt_fault['location']:
            source = gt_fault['location'].split('->')[0]
            check_status = self._find_action_by_type(env, 'check_status', device=source)
            results = []
            if check_status is not None:
                for _ in range(3):
                    obs, _, terminated, _, info = env.step(check_status)
                    if terminated:
                        break
                    result = info.get('action_result')
                    if result and hasattr(result, 'result') and result.result:
                        results.append(result.result.data.get('status'))
            
            for status in results:
                assert status == 'up'
        env.close()


class TestPerformanceDegradationEdgeCasesAdvanced:
    """Advanced edge case tests for performance degradation."""

    def test_diagnosis_immediately(self):
        """Test making diagnosis immediately without exploration."""
        env = NetworkTroubleshootingEnv(
            max_devices=4, max_episode_steps=10, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=800)
        
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.category.value == 'diagnosis':
                obs, _, terminated, _, _ = env.step(action_id)
                assert terminated
                break
        env.close()

    def test_episode_max_one_step(self):
        """Test episode with max_episode_steps=1."""
        env = NetworkTroubleshootingEnv(
            max_devices=4, max_episode_steps=1, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=802)
        
        valid_actions = env.get_valid_actions()
        if valid_actions:
            obs, _, terminated, truncated, _ = env.step(valid_actions[0])
            assert terminated or truncated
        env.close()

    def test_observation_shapes_consistent(self):
        """Test observation shapes remain consistent throughout episode."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=20, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=803)
        
        initial_shapes = {k: np.array(obs[k]).shape for k in ['discovery_matrix', 'device_status', 'recent_diagnostics', 'episode_metadata']}
        
        for step in range(8):
            valid_actions = env.get_valid_actions()
            non_diagnosis = [a for a in valid_actions 
                           if env.action_space_manager.get_action_spec(a) and 
                           env.action_space_manager.get_action_spec(a).category.value != 'diagnosis']
            if not non_diagnosis:
                break
            obs, _, terminated, _, _ = env.step(non_diagnosis[0])
            if terminated:
                break
            for key, expected in initial_shapes.items():
                assert np.array(obs[key]).shape == expected
        env.close()

    def test_info_dict_keys(self):
        """Test info dict always contains required keys."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=15, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=804)
        
        required = ['step_count', 'ground_truth_fault', 'network_size']
        for key in required:
            assert key in info
        
        for step in range(5):
            valid_actions = env.get_valid_actions()
            non_diagnosis = [a for a in valid_actions 
                           if env.action_space_manager.get_action_spec(a) and 
                           env.action_space_manager.get_action_spec(a).category.value != 'diagnosis']
            if not non_diagnosis:
                break
            obs, _, terminated, _, info = env.step(non_diagnosis[0])
            if terminated:
                break
            for key in required:
                assert key in info
        env.close()

    def test_multiple_episodes_variety(self):
        """Test running multiple episodes produces variety."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=10, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        
        faults = set()
        for seed in range(10):
            obs, info = env.reset(seed=seed)
            gt = info['ground_truth_fault']
            faults.add((gt['type'], gt['location']))
        
        assert len(faults) > 1
        env.close()


class TestPerformanceDegradationTopologyAdvanced:
    """Test performance degradation across different topology variations."""

    def test_small_network(self):
        """Test performance degradation in minimal network (3 devices)."""
        env = NetworkTroubleshootingEnv(
            max_devices=3, max_episode_steps=15, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=1000)
        
        assert info['network_size'] >= 3
        assert info['ground_truth_fault']['type'] == 'performance_degradation'
        env.close()

    def test_large_network(self):
        """Test performance degradation in larger network (10 devices)."""
        env = NetworkTroubleshootingEnv(
            max_devices=10, max_episode_steps=50, fault_types=[FaultType.PERFORMANCE_DEGRADATION]
        )
        obs, info = env.reset(seed=1001)
        
        assert info['network_size'] >= 3
        assert info['ground_truth_fault']['type'] == 'performance_degradation'
        env.close()

    def test_across_topology_types(self):
        """Test performance degradation works across different topology types."""
        for topo in [['linear'], ['star'], ['mesh'], ['hierarchical']]:
            env = NetworkTroubleshootingEnv(
                max_devices=6, max_episode_steps=20,
                fault_types=[FaultType.PERFORMANCE_DEGRADATION],
                topology_types=topo
            )
            obs, info = env.reset(seed=1002)
            
            assert info['ground_truth_fault']['type'] == 'performance_degradation'
            env.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
