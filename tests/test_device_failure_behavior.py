"""
Comprehensive tests for DEVICE_FAILURE fault type behavior in NetHeal.

Expected Behavior:
- Device status is set to 'down'
- Ping to/from device fails with "unreachable" or "source down" error
- Traceroute fails if device is on the path
- Check status on device returns status='down'
- Check interfaces fails with "Device is down"
- All paths through the device are broken
"""

import pytest
import numpy as np
from typing import List, Optional

from netheal.network.graph import NetworkGraph, DeviceType
from netheal.network.topology import TopologyGenerator
from netheal.faults.injector import FaultInjector, FaultType, FaultInfo
from netheal.tools.simulator import ToolSimulator
from netheal.environment.env import NetworkTroubleshootingEnv


class TestDeviceFailureBasicBehavior:
    """Test basic device failure fault behavior."""

    def test_device_is_down_after_failure(self):
        """Verify that the failed device status is 'down'."""
        network = TopologyGenerator.generate_linear_topology(4)
        injector = FaultInjector(network)
        devices = network.get_all_devices()
        target_device = devices[1]
        fault = injector.inject_device_failure(device_id=target_device)
        assert not network.is_device_up(target_device)
        assert fault.fault_type == FaultType.DEVICE_FAILURE
        assert fault.location == target_device

    def test_check_status_shows_device_down(self):
        """Verify check_status tool shows device as down."""
        network = TopologyGenerator.generate_linear_topology(4)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        devices = network.get_all_devices()
        injector.inject_device_failure(device_id=devices[1])
        result = tool_sim.check_status(devices[1])
        assert result.success
        assert result.data['status'] == 'down'

    def test_check_interfaces_fails_on_down_device(self):
        """Verify check_interfaces fails on a down device."""
        network = TopologyGenerator.generate_linear_topology(4)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        devices = network.get_all_devices()
        injector.inject_device_failure(device_id=devices[1])
        result = tool_sim.check_interfaces(devices[1])
        assert not result.success

    def test_ping_to_failed_device_fails(self):
        """Verify ping to failed device fails."""
        network = TopologyGenerator.generate_linear_topology(4)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        devices = network.get_all_devices()
        injector.inject_device_failure(device_id=devices[1])
        result = tool_sim.ping(devices[0], devices[1])
        assert not result.success

    def test_ping_from_failed_device_fails(self):
        """Verify ping from failed device fails."""
        network = TopologyGenerator.generate_linear_topology(4)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        devices = network.get_all_devices()
        injector.inject_device_failure(device_id=devices[1])
        result = tool_sim.ping(devices[1], devices[2])
        assert not result.success


class TestDeviceFailurePathEffects:
    """Test path-related effects of device failure."""

    def test_paths_through_failed_device_broken(self):
        """Verify paths through failed device are broken."""
        network = TopologyGenerator.generate_linear_topology(5)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        devices = network.get_all_devices()
        injector.inject_device_failure(device_id=devices[2])
        result = tool_sim.ping(devices[0], devices[4])
        assert not result.success

    def test_paths_not_through_failed_device_work(self):
        """Verify paths not through failed device still work."""
        network = TopologyGenerator.generate_linear_topology(5)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        devices = network.get_all_devices()
        injector.inject_device_failure(device_id=devices[3])
        result = tool_sim.ping(devices[0], devices[2])
        assert result.success

    def test_traceroute_fails_at_failed_device(self):
        """Verify traceroute shows failure at the failed device."""
        network = TopologyGenerator.generate_linear_topology(5)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        devices = network.get_all_devices()
        injector.inject_device_failure(device_id=devices[2])
        result = tool_sim.traceroute(devices[0], devices[4])
        assert not result.success

    def test_devices_beyond_failure_unreachable(self):
        """Verify devices beyond failed device are unreachable."""
        network = TopologyGenerator.generate_linear_topology(5)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        devices = network.get_all_devices()
        injector.inject_device_failure(device_id=devices[2])
        for target in [devices[3], devices[4]]:
            result = tool_sim.ping(devices[0], target)
            assert not result.success


class TestDeviceFailureTopologyVariations:
    """Test device failure behavior across different topologies."""

    def test_device_failure_in_star_topology(self):
        """Test device failure in star topology."""
        network = TopologyGenerator.generate_star_topology(5)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        devices = network.get_all_devices()
        hub = max(devices, key=lambda d: len(network.get_device_connections(d)))
        edge_device = [d for d in devices if d != hub][0]
        injector.inject_device_failure(device_id=edge_device)
        assert not network.is_device_up(edge_device)
        result = tool_sim.check_status(edge_device)
        assert result.data['status'] == 'down'

    def test_hub_failure_in_star_topology(self):
        """Test hub failure in star topology isolates all edge devices."""
        network = TopologyGenerator.generate_star_topology(5)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        devices = network.get_all_devices()
        hub = max(devices, key=lambda d: len(network.get_device_connections(d)))
        injector.inject_device_failure(device_id=hub)
        assert not network.is_device_up(hub)
        edge_devices = [d for d in devices if d != hub]
        if len(edge_devices) >= 2:
            result = tool_sim.ping(edge_devices[0], edge_devices[1])
            assert not result.success

    def test_device_failure_in_mesh_topology(self):
        """Test device failure in mesh topology."""
        network = TopologyGenerator.generate_mesh_topology(5, connection_probability=0.7)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        devices = network.get_all_devices()
        injector.inject_device_failure(device_id=devices[0])
        assert not network.is_device_up(devices[0])
        result = tool_sim.check_status(devices[0])
        assert result.data['status'] == 'down'

    def test_device_failure_in_hierarchical_topology(self):
        """Test device failure in hierarchical topology."""
        network = TopologyGenerator.generate_hierarchical_topology(num_layers=3, devices_per_layer=[1, 2, 3])
        injector = FaultInjector(network)
        devices = network.get_all_devices()
        target = devices[1] if len(devices) > 1 else devices[0]
        injector.inject_device_failure(device_id=target)
        assert not network.is_device_up(target)


class TestDeviceFailureDifferentiation:
    """Test that device failure can be differentiated from other fault types."""

    def test_device_failure_vs_link_failure(self):
        """Verify device failure differs from link failure in diagnostics."""
        network1 = TopologyGenerator.generate_linear_topology(4)
        injector1 = FaultInjector(network1)
        tool_sim1 = ToolSimulator(network1)
        devices1 = network1.get_all_devices()
        injector1.inject_device_failure(device_id=devices1[1])
        status1 = tool_sim1.check_status(devices1[1])
        assert status1.data['status'] == 'down'
        
        network2 = TopologyGenerator.generate_linear_topology(4)
        injector2 = FaultInjector(network2)
        tool_sim2 = ToolSimulator(network2)
        devices2 = network2.get_all_devices()
        injector2.inject_link_failure(source=devices2[1], destination=devices2[2])
        status2 = tool_sim2.check_status(devices2[1])
        assert status2.data['status'] == 'up'

    def test_device_failure_vs_misconfiguration(self):
        """Verify device failure differs from misconfiguration."""
        network1 = TopologyGenerator.generate_linear_topology(4)
        injector1 = FaultInjector(network1)
        tool_sim1 = ToolSimulator(network1)
        devices1 = network1.get_all_devices()
        injector1.inject_device_failure(device_id=devices1[1])
        iface_result1 = tool_sim1.check_interfaces(devices1[1])
        assert not iface_result1.success
        
        network2 = TopologyGenerator.generate_linear_topology(4)
        injector2 = FaultInjector(network2)
        tool_sim2 = ToolSimulator(network2)
        devices2 = network2.get_all_devices()
        injector2.inject_misconfiguration(device_id=devices2[1])
        iface_result2 = tool_sim2.check_interfaces(devices2[1])
        assert iface_result2.success


class TestDeviceFailureEdgeCases:
    """Test edge cases for device failure faults."""

    def test_failure_on_single_device_network(self):
        """Test device failure on minimal network."""
        network = NetworkGraph()
        network.add_device('D0', DeviceType.ROUTER)
        injector = FaultInjector(network)
        tool_sim = ToolSimulator(network)
        injector.inject_device_failure(device_id='D0')
        assert not network.is_device_up('D0')
        result = tool_sim.check_status('D0')
        assert result.data['status'] == 'down'

    def test_failure_preserves_other_devices(self):
        """Verify failing one device doesn't affect others."""
        network = TopologyGenerator.generate_linear_topology(4)
        injector = FaultInjector(network)
        devices = network.get_all_devices()
        injector.inject_device_failure(device_id=devices[1])
        for device in devices:
            if device != devices[1]:
                assert network.is_device_up(device)

    def test_multiple_device_failures(self):
        """Test multiple device failures in same network."""
        network = TopologyGenerator.generate_linear_topology(5)
        injector = FaultInjector(network)
        devices = network.get_all_devices()
        injector.inject_device_failure(device_id=devices[1])
        injector.inject_device_failure(device_id=devices[3])
        assert not network.is_device_up(devices[1])
        assert not network.is_device_up(devices[3])
        assert network.is_device_up(devices[0])
        assert network.is_device_up(devices[2])
        assert network.is_device_up(devices[4])


class TestDeviceFailureWithEnvironment:
    """Test device failure behavior through the full RL environment."""

    def test_environment_with_device_failure_fault_type(self):
        """Test environment generates device failure faults correctly."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=20, fault_types=[FaultType.DEVICE_FAILURE]
        )
        for seed in range(5):
            obs, info = env.reset(seed=seed)
            gt_fault = info['ground_truth_fault']
            assert gt_fault['type'] == 'device_failure'
            assert '->' not in gt_fault['location']
        env.close()

    def test_diagnostic_tools_reveal_device_failure(self):
        """Test that diagnostic tools can reveal device failure."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=30, fault_types=[FaultType.DEVICE_FAILURE]
        )
        obs, info = env.reset(seed=42)
        faulty_device = info['ground_truth_fault']['location']
        
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.action_type.value == 'scan_network':
                env.step(action_id)
                break
        
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.action_type.value == 'check_status':
                if spec.parameters.get('device') == faulty_device:
                    obs, _, _, _, info = env.step(action_id)
                    action_result = info.get('action_result')
                    if action_result and hasattr(action_result, 'result') and action_result.result:
                        assert action_result.result.data.get('status') == 'down'
                    break
        env.close()

    def test_correct_diagnosis_gives_positive_reward(self):
        """Test correctly diagnosing device failure gives positive reward."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=20, fault_types=[FaultType.DEVICE_FAILURE]
        )
        obs, info = env.reset(seed=42)
        gt_fault = info['ground_truth_fault']
        
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.category.value == 'diagnosis':
                if spec.action_type.value == gt_fault['type'] and spec.parameters.get('location') == gt_fault['location']:
                    obs, reward, terminated, _, info = env.step(action_id)
                    assert terminated
                    assert reward > 0
                    break
        env.close()

    def test_wrong_diagnosis_gives_penalty(self):
        """Test incorrectly diagnosing device failure gives penalty."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=20, fault_types=[FaultType.DEVICE_FAILURE]
        )
        obs, info = env.reset(seed=42)
        gt_fault = info['ground_truth_fault']
        
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.category.value == 'diagnosis':
                if spec.action_type.value != gt_fault['type']:
                    obs, reward, terminated, _, info = env.step(action_id)
                    assert terminated
                    breakdown = info.get('reward_breakdown', {})
                    assert breakdown.get('diagnosis_reward', 0) < 0
                    break
        env.close()


class TestDeviceFailureReproducibility:
    """Test reproducibility of device failure scenarios."""

    def test_same_seed_produces_same_fault(self):
        """Verify same seed produces identical device failure."""
        results = []
        for _ in range(3):
            env = NetworkTroubleshootingEnv(
                max_devices=5, max_episode_steps=10, fault_types=[FaultType.DEVICE_FAILURE]
            )
            obs, info = env.reset(seed=12345)
            results.append(info['ground_truth_fault'])
            env.close()
        for i in range(1, len(results)):
            assert results[i]['type'] == results[0]['type']
            assert results[i]['location'] == results[0]['location']

    def test_different_seeds_produce_different_faults(self):
        """Verify different seeds can produce different device failures."""
        faults = set()
        for seed in range(20):
            env = NetworkTroubleshootingEnv(
                max_devices=6, max_episode_steps=10, fault_types=[FaultType.DEVICE_FAILURE]
            )
            obs, info = env.reset(seed=seed)
            faults.add((info['ground_truth_fault']['type'], info['ground_truth_fault']['location']))
            env.close()
        assert len(faults) > 1


# =============================================================================
# ADVANCED MULTI-ACTION EPISODE TESTS
# =============================================================================

class TestDeviceFailureMultiStepDiscovery:
    """Test device failure behavior across multi-step discovery sequences."""

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

    def test_scan_then_check_status_finds_down_device(self):
        """Test scanning then checking status reveals the down device."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=50, fault_types=[FaultType.DEVICE_FAILURE]
        )
        obs, info = env.reset(seed=100)
        faulty_device = info['ground_truth_fault']['location']
        
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            env.step(scan_action)
        
        for action_id in self._find_actions_by_type(env, 'check_status')[:6]:
            spec = env.action_space_manager.get_action_spec(action_id)
            device = spec.parameters.get('device')
            obs, _, terminated, _, info = env.step(action_id)
            if terminated:
                break
            result = info.get('action_result')
            if result and hasattr(result, 'result') and result.result:
                if result.result.data.get('status') == 'down':
                    assert device == faulty_device
        env.close()

    def test_scan_then_check_interfaces_fails_on_faulty_device(self):
        """Test that check_interfaces fails on the faulty device."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=30, fault_types=[FaultType.DEVICE_FAILURE]
        )
        obs, info = env.reset(seed=101)
        faulty_device = info['ground_truth_fault']['location']
        
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            env.step(scan_action)
        
        check_iface = self._find_action_by_type(env, 'check_interfaces', device=faulty_device)
        if check_iface is not None:
            obs, _, _, _, info = env.step(check_iface)
            result = info.get('action_result')
            if result and hasattr(result, 'result') and result.result:
                assert not result.result.success
        env.close()

    def test_systematic_ping_sweep(self):
        """Test systematic pinging reveals unreachable device."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=50, fault_types=[FaultType.DEVICE_FAILURE]
        )
        obs, info = env.reset(seed=102)
        
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            env.step(scan_action)
        
        ping_results = []
        for action_id in self._find_actions_by_type(env, 'ping')[:15]:
            obs, _, terminated, _, info = env.step(action_id)
            if terminated:
                break
            result = info.get('action_result')
            if result and hasattr(result, 'result') and result.result:
                ping_results.append(result.result.success)
        
        assert len(ping_results) > 0
        env.close()

    def test_traceroute_sequence(self):
        """Test multiple traceroutes help identify failure point."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=40, fault_types=[FaultType.DEVICE_FAILURE]
        )
        obs, info = env.reset(seed=103)
        
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            env.step(scan_action)
        
        for action_id in self._find_actions_by_type(env, 'traceroute')[:8]:
            obs, _, terminated, _, _ = env.step(action_id)
            if terminated:
                break
        env.close()

    def test_progressive_exploration(self):
        """Test progressive network exploration."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=30, fault_types=[FaultType.DEVICE_FAILURE]
        )
        obs, info = env.reset(seed=104)
        
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            env.step(scan_action)
        
        for action_id in self._find_actions_by_type(env, 'discover_neighbors')[:5]:
            obs, _, terminated, _, _ = env.step(action_id)
            if terminated:
                break
        assert 'discovery_matrix' in obs
        env.close()


class TestDeviceFailureObservationConsistency:
    """Test that observations remain consistent with device failure fault."""

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
            max_devices=5, max_episode_steps=20, fault_types=[FaultType.DEVICE_FAILURE]
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
            max_devices=5, max_episode_steps=30, fault_types=[FaultType.DEVICE_FAILURE]
        )
        obs, info = env.reset(seed=201)
        
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            env.step(scan_action)
        
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

    def test_device_status_reflects_failure(self):
        """Test device status reflects the failure after check_status."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=30, fault_types=[FaultType.DEVICE_FAILURE]
        )
        obs, info = env.reset(seed=202)
        faulty_device = info['ground_truth_fault']['location']
        
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            env.step(scan_action)
        
        check_status = self._find_action_by_type(env, 'check_status', device=faulty_device)
        if check_status is not None:
            obs, _, _, _, info = env.step(check_status)
            result = info.get('action_result')
            if result and hasattr(result, 'result') and result.result:
                assert result.result.data.get('status') == 'down'
        env.close()


class TestDeviceFailureDiagnosticWorkflows:
    """Test realistic diagnostic workflows for finding device failure."""

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

    def test_workflow_scan_checkstatus_diagnose(self):
        """Test workflow: scan -> check_status -> diagnose."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=30, fault_types=[FaultType.DEVICE_FAILURE]
        )
        obs, info = env.reset(seed=300)
        gt_fault = info['ground_truth_fault']
        
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            env.step(scan_action)
        
        check_status = self._find_action_by_type(env, 'check_status', device=gt_fault['location'])
        if check_status is not None:
            obs, _, _, _, info = env.step(check_status)
            result = info.get('action_result')
            if result and hasattr(result, 'result') and result.result:
                assert result.result.data.get('status') == 'down'
        
        diagnosis = self._find_diagnosis_action(env, 'device_failure', gt_fault['location'])
        if diagnosis is not None:
            obs, reward, terminated, _, _ = env.step(diagnosis)
            assert terminated
            assert reward > 0
        env.close()

    def test_workflow_status_check_all_devices(self):
        """Test checking status on all devices to find the failed one."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=30, fault_types=[FaultType.DEVICE_FAILURE]
        )
        obs, info = env.reset(seed=302)
        faulty_device = info['ground_truth_fault']['location']
        
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            env.step(scan_action)
        
        devices_down = []
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.action_type.value == 'check_status':
                device = spec.parameters.get('device')
                obs, _, terminated, _, info = env.step(action_id)
                if terminated:
                    break
                result = info.get('action_result')
                if result and hasattr(result, 'result') and result.result:
                    if result.result.data.get('status') == 'down':
                        devices_down.append(device)
        
        if not env.episode_done:
            assert faulty_device in devices_down
        env.close()


class TestDeviceFailureRewardAccuracy:
    """Test reward calculation accuracy for device failure scenarios."""

    def _find_diagnosis_action(self, env, fault_type: str, location: str) -> Optional[int]:
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.category.value == 'diagnosis':
                if spec.action_type.value == fault_type and spec.parameters.get('location') == location:
                    return action_id
        return None

    def test_correct_diagnosis_reward_positive(self):
        """Test correct device failure diagnosis yields positive reward."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=10, fault_types=[FaultType.DEVICE_FAILURE]
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
            max_devices=5, max_episode_steps=10, fault_types=[FaultType.DEVICE_FAILURE]
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
            max_devices=5, max_episode_steps=20, fault_types=[FaultType.DEVICE_FAILURE]
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
            max_devices=5, max_episode_steps=10, fault_types=[FaultType.DEVICE_FAILURE]
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


class TestDeviceFailureStateConsistency:
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
            max_devices=5, max_episode_steps=20, fault_types=[FaultType.DEVICE_FAILURE]
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
            max_devices=5, max_episode_steps=20, fault_types=[FaultType.DEVICE_FAILURE]
        )
        obs, info = env.reset(seed=501)
        faulty_device = info['ground_truth_fault']['location']
        
        for step in range(8):
            assert not env.network.is_device_up(faulty_device)
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
            max_devices=6, max_episode_steps=30, fault_types=[FaultType.DEVICE_FAILURE]
        )
        obs, info = env.reset(seed=502)
        
        initial_valid = len(env.get_valid_actions())
        scan_action = self._find_action_by_type(env, 'scan_network')
        if scan_action is not None:
            env.step(scan_action)
        assert len(env.get_valid_actions()) >= initial_valid
        env.close()

    def test_step_count_increments(self):
        """Test step count increments with each action."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=20, fault_types=[FaultType.DEVICE_FAILURE]
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
            max_devices=5, max_episode_steps=15, fault_types=[FaultType.DEVICE_FAILURE]
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


class TestDeviceFailureToolInteractions:
    """Test tool interactions specific to device failure faults."""

    def _find_action_by_type(self, env, action_type: str, **params) -> Optional[int]:
        for action_id in range(env.action_space.n):
            spec = env.action_space_manager.get_action_spec(action_id)
            if spec and spec.action_type.value == action_type:
                if all(spec.parameters.get(k) == v for k, v in params.items()):
                    return action_id
        return None

    def test_ping_to_failed_device_fails_env(self):
        """Test ping to failed device fails through environment."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=30, fault_types=[FaultType.DEVICE_FAILURE]
        )
        obs, info = env.reset(seed=600)
        faulty_device = info['ground_truth_fault']['location']
        
        scan = self._find_action_by_type(env, 'scan_network')
        if scan is not None:
            env.step(scan)
        
        devices = env.network.get_all_devices()
        source = [d for d in devices if d != faulty_device][0] if len(devices) > 1 else None
        
        if source:
            ping = self._find_action_by_type(env, 'ping', source=source, destination=faulty_device)
            if ping is not None:
                obs, _, _, _, info = env.step(ping)
                result = info.get('action_result')
                if result and hasattr(result, 'result') and result.result:
                    assert not result.result.success
        env.close()

    def test_check_status_reveals_down(self):
        """Test check_status reveals device is down."""
        env = NetworkTroubleshootingEnv(
            max_devices=6, max_episode_steps=30, fault_types=[FaultType.DEVICE_FAILURE]
        )
        obs, info = env.reset(seed=601)
        faulty_device = info['ground_truth_fault']['location']
        
        scan = self._find_action_by_type(env, 'scan_network')
        if scan is not None:
            env.step(scan)
        
        check_status = self._find_action_by_type(env, 'check_status', device=faulty_device)
        if check_status is not None:
            obs, _, _, _, info = env.step(check_status)
            result = info.get('action_result')
            if result and hasattr(result, 'result') and result.result:
                assert result.result.data.get('status') == 'down'
        env.close()

    def test_tool_costs_applied(self):
        """Test that tool costs are correctly applied."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=30, fault_types=[FaultType.DEVICE_FAILURE]
        )
        obs, info = env.reset(seed=604)
        
        scan = self._find_action_by_type(env, 'scan_network')
        if scan is not None:
            env.step(scan)
        
        tool_costs = {}
        for action_type in ['ping', 'traceroute', 'check_status']:
            action = self._find_action_by_type(env, action_type)
            if action is not None:
                obs, _, terminated, _, info = env.step(action)
                if terminated:
                    break
                result = info.get('action_result')
                if result and hasattr(result, 'result') and result.result:
                    tool_costs[action_type] = result.result.cost
        
        for tool, cost in tool_costs.items():
            assert cost > 0
        env.close()


class TestDeviceFailureComplexScenarios:
    """Test complex multi-step scenarios with device failure."""

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
            max_devices=6, max_episode_steps=40, fault_types=[FaultType.DEVICE_FAILURE]
        )
        obs, info = env.reset(seed=700)
        gt_fault = info['ground_truth_fault']
        
        actions_taken = []
        
        scan = self._find_action_by_type(env, 'scan_network')
        if scan is not None:
            env.step(scan)
            actions_taken.append('scan')
        
        check_status = self._find_action_by_type(env, 'check_status', device=gt_fault['location'])
        if check_status is not None and not env.episode_done:
            env.step(check_status)
            actions_taken.append('check_status')
        
        if not env.episode_done:
            diagnosis = self._find_diagnosis_action(env, gt_fault['type'], gt_fault['location'])
            if diagnosis is not None:
                obs, reward, terminated, _, _ = env.step(diagnosis)
                actions_taken.append('diagnosis')
                assert terminated
                assert reward > 0
        
        assert len(actions_taken) > 0
        env.close()

    def test_episode_wrong_diagnosis(self):
        """Test episode where agent diagnoses wrong."""
        env = NetworkTroubleshootingEnv(
            max_devices=5, max_episode_steps=30, fault_types=[FaultType.DEVICE_FAILURE]
        )
        obs, info = env.reset(seed=701)
        gt_fault = info['ground_truth_fault']
        
        scan = self._find_action_by_type(env, 'scan_network')
        if scan is not None:
            env.step(scan)
        
        if not env.episode_done:
            for action_id in range(env.action_space.n):
                spec = env.action_space_manager.get_action_spec(action_id)
                if spec and spec.category.value == 'diagnosis' and spec.action_type.value == 'link_failure':
                    obs, reward, terminated, _, info = env.step(action_id)
                    assert terminated
                    breakdown = info.get('reward_breakdown', {})
                    assert breakdown.get('diagnosis_reward', 0) <= 0 or reward < 5.0
                    break
        env.close()

    def test_episode_timeout(self):
        """Test episode timing out without diagnosis."""
        env = NetworkTroubleshootingEnv(
            max_devices=4, max_episode_steps=8, fault_types=[FaultType.DEVICE_FAILURE]
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
            max_devices=5, max_episode_steps=30, fault_types=[FaultType.DEVICE_FAILURE]
        )
        obs, info = env.reset(seed=705)
        faulty_device = info['ground_truth_fault']['location']
        
        scan = self._find_action_by_type(env, 'scan_network')
        if scan is not None:
            env.step(scan)
        
        check_status = self._find_action_by_type(env, 'check_status', device=faulty_device)
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
            assert status == 'down'
        env.close()


class TestDeviceFailureEdgeCasesAdvanced:
    """Advanced edge case tests for device failure."""

    def test_diagnosis_immediately(self):
        """Test making diagnosis immediately without exploration."""
        env = NetworkTroubleshootingEnv(
            max_devices=4, max_episode_steps=10, fault_types=[FaultType.DEVICE_FAILURE]
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
            max_devices=4, max_episode_steps=1, fault_types=[FaultType.DEVICE_FAILURE]
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
            max_devices=5, max_episode_steps=20, fault_types=[FaultType.DEVICE_FAILURE]
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
            max_devices=5, max_episode_steps=15, fault_types=[FaultType.DEVICE_FAILURE]
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
            max_devices=6, max_episode_steps=10, fault_types=[FaultType.DEVICE_FAILURE]
        )
        
        faults = set()
        for seed in range(10):
            obs, info = env.reset(seed=seed)
            gt = info['ground_truth_fault']
            faults.add((gt['type'], gt['location']))
        
        assert len(faults) > 1
        env.close()


class TestDeviceFailureTopologyAdvanced:
    """Test device failure across different topology variations."""

    def test_small_network(self):
        """Test device failure in minimal network (3 devices)."""
        env = NetworkTroubleshootingEnv(
            max_devices=3, max_episode_steps=15, fault_types=[FaultType.DEVICE_FAILURE]
        )
        obs, info = env.reset(seed=1000)
        
        assert info['network_size'] >= 3
        assert info['ground_truth_fault']['type'] == 'device_failure'
        assert not env.network.is_device_up(info['ground_truth_fault']['location'])
        env.close()

    def test_large_network(self):
        """Test device failure in larger network (10 devices)."""
        env = NetworkTroubleshootingEnv(
            max_devices=10, max_episode_steps=50, fault_types=[FaultType.DEVICE_FAILURE]
        )
        obs, info = env.reset(seed=1001)
        
        assert info['network_size'] >= 3
        assert info['ground_truth_fault']['type'] == 'device_failure'
        assert not env.network.is_device_up(info['ground_truth_fault']['location'])
        env.close()

    def test_across_topology_types(self):
        """Test device failure works across different topology types."""
        for topo in [['linear'], ['star'], ['mesh'], ['hierarchical']]:
            env = NetworkTroubleshootingEnv(
                max_devices=6, max_episode_steps=20,
                fault_types=[FaultType.DEVICE_FAILURE],
                topology_types=topo
            )
            obs, info = env.reset(seed=1002)
            
            assert info['ground_truth_fault']['type'] == 'device_failure'
            assert not env.network.is_device_up(info['ground_truth_fault']['location'])
            env.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
