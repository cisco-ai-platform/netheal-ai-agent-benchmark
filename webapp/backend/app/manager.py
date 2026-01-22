# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Environment Manager for the NetHeal web backend.

This module wraps NetworkTroubleshootingEnv and provides JSON-serializable
state for the FastAPI layer.

Follows clean-code rules: modular, low duplication, no globals leaked,
minimal side effects.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import threading
import time
import numpy as np
from gymnasium import spaces

from netheal import NetworkTroubleshootingEnv
from netheal.environment.actions import ActionSpec
from netheal.environment.observation import DiagnosticResult
from netheal.tools.simulator import ToolResult


class _Singleton(type):
    _instances: Dict[type, Any] = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):  # type: ignore[override]
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class EnvManager(metaclass=_Singleton):
    """Singleton manager holding a single NetHeal env instance for the demo.

    For production/multi-user setups, introduce session IDs and per-session envs.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._env: Optional[NetworkTroubleshootingEnv] = None
        self._action_meanings: List[str] = []
        self._last_reward: float = 0.0
        self._last_reward_breakdown: Dict[str, float] = {}

    # --------------------------- Public API ---------------------------

    def reset(self,
              seed: Optional[int] = None,
              max_devices: int = 15,
              max_episode_steps: int = 100,
              topology_types: Optional[List[str]] = None,
              enable_user_hints: bool = True,
              hint_provider_mode: str = "auto",
              user_context: Optional[Dict[str, Any]] = None
              ) -> Dict[str, Any]:
        """Create a new env episode and return initial state."""
        with self._lock:
            self._env = NetworkTroubleshootingEnv(
                max_devices=max_devices,
                max_episode_steps=max_episode_steps,
                topology_types=topology_types or None,
                enable_user_hints=enable_user_hints,
                hint_provider_mode=hint_provider_mode,
                user_context=user_context or {},
            )
            obs, info = self._env.reset(seed=seed)
            self._action_meanings = self._env.get_action_meanings()
            self._last_reward = 0.0
            self._last_reward_breakdown = {}
            return self._build_state(obs, info)

    def get_state(self) -> Dict[str, Any]:
        """Return current state without advancing the environment."""
        with self._lock:
            self._ensure_env()
            assert self._env is not None
            # Rebuild obs/info from the current env observation
            obs = self._env.observation.to_dict()
            info = self._env._get_info()  # internal, read-only usage for web state
            return self._build_state(obs, info)

    def step(self, action_id: int) -> Dict[str, Any]:
        """Take a step in the environment using the given action ID."""
        with self._lock:
            self._ensure_env()
            assert self._env is not None

            obs, reward, terminated, truncated, info = self._env.step(action_id)
            self._last_reward = float(reward)
            self._last_reward_breakdown = self._to_jsonable(info.get('reward_breakdown', {}))

            state = self._build_state(obs, info)
            state.update({
                'last_reward': self._last_reward,
                'terminated': bool(terminated),
                'truncated': bool(truncated),
            })

            # Compute final outcome metadata
            final_outcome = {
                'final': bool(terminated or truncated),
                'by': 'diagnosis' if terminated else ('timeout' if truncated else None),
                'correct': None,
                'diagnosed_fault': None,
                'diagnosed_location': None,
                'ground_truth_fault': None,
                'ground_truth_location': None,
                'total_reward': self._last_reward,
            }
            try:
                safe_info = state.get('info', {})
                gt = safe_info.get('ground_truth_fault') or {}
                final_outcome['ground_truth_fault'] = gt.get('type')
                final_outcome['ground_truth_location'] = gt.get('location')

                spec = safe_info.get('action_spec') or {}
                if spec and spec.get('category') == 'diagnosis':
                    final_outcome['diagnosed_fault'] = spec.get('action_type')
                    params = spec.get('parameters') or {}
                    final_outcome['diagnosed_location'] = params.get('location')
                    # Determine correctness by comparing against ground truth
                    final_outcome['correct'] = (
                        final_outcome['diagnosed_fault'] == final_outcome['ground_truth_fault'] and
                        final_outcome['diagnosed_location'] == final_outcome['ground_truth_location']
                    )
            except Exception:
                pass

            state['final_outcome'] = final_outcome
            return state

    def get_actions(self) -> Dict[str, Any]:
        """Return valid actions and action descriptions."""
        with self._lock:
            self._ensure_env()
            assert self._env is not None
            valid = self._env.get_valid_actions()
            return {
                'valid_actions': valid,
                'action_meanings': self._action_meanings,
                'valid_action_specs': self._build_valid_action_specs(valid),
            }

    def export_scenario(self) -> Dict[str, Any]:
        """Export current scenario state for later import."""
        with self._lock:
            self._ensure_env()
            assert self._env is not None
            
            # Get observation and ensure numpy arrays are converted to lists
            obs_dict = self._env.observation.to_dict() if self._env.observation else {}
            obs_dict = self._ndarray_to_list(obs_dict)
            
            # Add diagnostic history and discovery state to observation
            if self._env.observation:
                # Export diagnostic history
                diagnostic_history = []
                for diag in self._env.observation.diagnostic_memory.results:
                    diag_dict = {
                        'tool_name': diag.tool_name,
                        'source': diag.source,
                        'destination': diag.destination,
                        'timestamp': float(diag.timestamp),
                        'confidence': float(diag.confidence),
                        'tool_result': {
                            'tool_name': diag.result.tool_name,
                            'success': bool(diag.result.success),
                            'data': diag.result.data or {},
                            'cost': float(diag.result.cost),
                        } if diag.result else None
                    }
                    diagnostic_history.append(diag_dict)
                obs_dict['diagnostic_history'] = diagnostic_history
                
                # Export discovery state (device_map and next_index)
                obs_dict['discovery_device_map'] = dict(self._env.observation.discovery_matrix.device_map)
                obs_dict['discovery_next_index'] = int(self._env.observation.discovery_matrix.next_index)
            
            # Capture all the necessary state to recreate the scenario
            scenario = {
                'version': '1.0.0',
                'metadata': {
                    'export_timestamp': time.time(),
                    'step_count': self._env.step_count,
                    'max_episode_steps': self._env.max_episode_steps,
                    'max_devices': self._env.max_devices,
                    'topology_types': self._env.topology_types,
                    'enable_user_hints': self._env.enable_user_hints,
                    'hint_provider_mode': self._env.hint_provider_mode,
                },
                'network': self._export_network(),
                'fault': self._export_fault(),
                'observation': obs_dict,
                'episode_state': {
                    'step_count': self._env.step_count,
                    'episode_done': self._env.episode_done,
                },
                'user_hint': self._env.user_hint,  # Export the current hint
            }
            return scenario
    
    def import_scenario(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Import and load a previously exported scenario."""
        with self._lock:
            version = scenario_data.get('version', '1.0.0')
            if version != '1.0.0':
                raise RuntimeError(f"Unsupported scenario version: {version}")
            
            metadata = scenario_data.get('metadata')
            if metadata is None:
                raise RuntimeError("Scenario is missing required 'metadata' field")
            
            # Create a new environment with the saved parameters
            self._env = NetworkTroubleshootingEnv(
                max_devices=metadata['max_devices'],
                max_episode_steps=metadata['max_episode_steps'],
                topology_types=metadata['topology_types'],
                enable_user_hints=metadata['enable_user_hints'],
                hint_provider_mode=metadata['hint_provider_mode'],
            )
            
            # Import network, fault, and state
            self._import_network(scenario_data.get('network', {}))
            self._import_fault(scenario_data.get('fault', {}))
            self._import_observation(scenario_data.get('observation', {}))

            # Rebuild action space for imported device IDs
            if self._env.network:
                try:
                    device_ids = list(self._env.network.get_all_devices())
                    self._env.action_space_manager.rebuild_for_network(device_ids)
                    self._env.action_space = spaces.Discrete(
                        self._env.action_space_manager.total_actions
                    )
                except Exception:
                    pass
            
            # Restore episode state
            episode_state = scenario_data.get('episode_state', {})
            self._env.step_count = episode_state.get('step_count', 0)
            self._env.episode_done = episode_state.get('episode_done', False)
            self._env.episode_start_time = time.time()  # Set current time as start
            
            # Restore user hint if available
            self._env.user_hint = scenario_data.get('user_hint', None)
            
            # Update action meanings (action space is managed internally by env)
            self._action_meanings = self._env.get_action_meanings()
            self._last_reward = 0.0
            self._last_reward_breakdown = {}
            
            # Return current state
            obs = self._env.observation.to_dict()
            info = self._env._get_info()
            return self._build_state(obs, info)

    # --------------------------- Internal helpers ---------------------------

    def _build_state(self, obs: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        assert self._env is not None
        # Convert numpy arrays in obs to lists
        safe_obs = {k: self._ndarray_to_list(v) for k, v in obs.items()}

        # Sanitize info (DiagnosticResult etc.)
        safe_info = self._sanitize_info(info)

        # Provide valid actions and action meanings
        valid = self._env.get_valid_actions()
        action_specs = self._build_valid_action_specs(valid)

        # Device index mapping (index -> device_id) for discovered devices
        device_index_map = []
        try:
            device_index_map = list(self._env.observation.discovery_matrix.get_discovered_devices())  # type: ignore
        except Exception:
            device_index_map = []

        return {
            'observation': safe_obs,
            'info': safe_info,
            'valid_actions': valid,
            'action_meanings': self._action_meanings,
            'valid_action_specs': action_specs,
            'device_index_map': device_index_map,
            'recent_diagnostics_detailed': self._recent_diagnostics_detailed(),
            'action_space_size': int(self._env.action_space.n),
        }

    def _build_valid_action_specs(self, valid_ids: List[int]) -> List[Dict[str, Any]]:
        assert self._env is not None
        specs: List[Dict[str, Any]] = []
        asm = self._env.action_space_manager
        for aid in valid_ids:
            spec = asm.get_action_spec(aid)
            if isinstance(spec, ActionSpec):
                spec_dict = spec.to_dict()
                spec_dict['id'] = aid
                specs.append(spec_dict)
        return specs

    def _recent_diagnostics_detailed(self) -> List[Dict[str, Any]]:
        assert self._env is not None
        detailed: List[Dict[str, Any]] = []
        try:
            results = self._env.observation.diagnostic_memory.get_recent_results(10)  # type: ignore
            for r in results:
                entry: Dict[str, Any] = {
                    'tool': r.tool_name,
                    'source': r.source,
                    'destination': r.destination,
                    'timestamp': float(r.timestamp),
                    'confidence': float(getattr(r, 'confidence', 1.0)),
                }
                tr: Optional[ToolResult] = getattr(r, 'result', None)
                if isinstance(tr, ToolResult):
                    entry.update({
                        'success': bool(tr.success),
                        'cost': float(tr.cost),
                        'data': tr.data,
                    })
                    # Friendly summary per tool
                    summary = None
                    d = tr.data or {}
                    if r.tool_name == 'ping':
                        if tr.success:
                            summary = f"Ping ok {d.get('latency_ms', '?')} ms"
                        else:
                            summary = d.get('error', 'Ping failed')
                    elif r.tool_name == 'traceroute':
                        if tr.success:
                            hops = d.get('hops')
                            total = d.get('total_latency_ms')
                            summary = f"Traceroute ok hops={hops}, total={total} ms"
                        else:
                            summary = d.get('error', 'Traceroute failed')
                    elif r.tool_name == 'check_status':
                        summary = f"Status={d.get('status','?')} IP={d.get('ip_address','?')}"
                    elif r.tool_name == 'check_interfaces':
                        summary = f"Ifaces up={d.get('up_interfaces',0)} down={d.get('down_interfaces',0)}"
                    elif r.tool_name == 'scan_network':
                        dd = d.get('discovered_devices') or []
                        summary = f"Discovered {len(dd)} devices"
                    elif r.tool_name == 'discover_neighbors':
                        nb = d.get('neighbors') or []
                        summary = f"Neighbors: {len(nb)}"
                    entry['summary'] = summary
                detailed.append(entry)
        except Exception:
            pass
        return detailed

    def _sanitize_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        info = dict(info) if info is not None else {}
        # action_spec may be an object; convert using to_dict if present
        spec = info.get('action_spec')
        if isinstance(spec, ActionSpec):
            info['action_spec'] = spec.to_dict()

        # action_result is a DiagnosticResult or None
        ar = info.get('action_result')
        if isinstance(ar, DiagnosticResult):
            info['action_result'] = self._diagnostic_result_to_dict(ar)
        else:
            info['action_result'] = None

        # reward_breakdown should be a simple dict already; ensure jsonable
        if 'reward_breakdown' in info:
            info['reward_breakdown'] = self._to_jsonable(info['reward_breakdown'])

        # user_hint is string or None; keep as is
        return info

    @staticmethod
    def _diagnostic_result_to_dict(dr: DiagnosticResult) -> Dict[str, Any]:
        tool: Optional[ToolResult] = getattr(dr, 'result', None)
        tool_dict: Optional[Dict[str, Any]] = None
        if isinstance(tool, ToolResult):
            tool_dict = {
                'tool_name': tool.tool_name,
                'success': bool(tool.success),
                'data': tool.data,
                'cost': float(tool.cost),
            }
        return {
            'tool_name': dr.tool_name,
            'source': dr.source,
            'destination': dr.destination,
            'timestamp': float(dr.timestamp),
            'confidence': float(getattr(dr, 'confidence', 1.0)),
            'tool_result': tool_dict,
        }

    @staticmethod
    def _ndarray_to_list(x: Any) -> Any:
        if isinstance(x, np.ndarray):
            return x.tolist()
        # recurse nested
        if isinstance(x, dict):
            return {k: EnvManager._ndarray_to_list(v) for k, v in x.items()}
        if isinstance(x, list):
            return [EnvManager._ndarray_to_list(v) for v in x]
        return x

    @staticmethod
    def _to_jsonable(x: Any) -> Any:
        if isinstance(x, (int, float, str, bool)) or x is None:
            return x
        if isinstance(x, dict):
            return {str(k): EnvManager._to_jsonable(v) for k, v in x.items()}
        if isinstance(x, list):
            return [EnvManager._to_jsonable(v) for v in x]
        if isinstance(x, np.ndarray):
            return x.tolist()
        return str(x)

    def _ensure_env(self) -> None:
        if self._env is None:
            raise RuntimeError("Environment not initialized. Call /api/env/reset first.")

    def _export_network(self) -> Dict[str, Any]:
        """Export network graph structure."""
        assert self._env is not None
        network = self._env.network
        if network is None:
            return {}
        
        # Export nodes
        nodes = []
        for node_id, data in network.graph.nodes(data=True):
            nodes.append({
                'id': node_id,
                'type': str(data.get('type', 'unknown')),
                'ip_address': data.get('ip_address'),
                'status': data.get('status', 'up'),
            })
        
        # Export edges
        edges = []
        for src, dst, data in network.graph.edges(data=True):
            edges.append({
                'source': src,
                'target': dst,
                'bandwidth': float(data.get('bandwidth', 100.0)),
                'latency': float(data.get('latency', 1.0)),
                'status': data.get('status', 'up'),
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
        }
    
    def _export_fault(self) -> Dict[str, Any]:
        """Export fault information."""
        assert self._env is not None
        fault = self._env.ground_truth_fault
        if fault is None:
            return {}
        
        return {
            'type': fault.fault_type.value if hasattr(fault.fault_type, 'value') else str(fault.fault_type),
            'location': fault.location,
            'details': fault.details or {},
        }
    
    def _import_network(self, network_data: Dict[str, Any]) -> None:
        """Import network graph structure."""
        assert self._env is not None
        from netheal.network.graph import NetworkGraph, DeviceType
        
        if not network_data:
            return
        
        # Create a new NetworkGraph
        network = NetworkGraph()
        
        # Add nodes
        nodes_data = network_data.get('nodes', [])
        for node in nodes_data:
            # Convert type string to DeviceType enum
            device_type_str = node.get('type', 'host')
            try:
                device_type = DeviceType(device_type_str)
            except (ValueError, KeyError):
                device_type = DeviceType.HOST  # Default fallback
            
            network.add_device(
                device_id=node['id'],
                device_type=device_type,
                ip_address=node.get('ip_address'),
                status=node.get('status', 'up')
            )
        
        # Add edges
        edges_data = network_data.get('edges', [])
        for edge in edges_data:
            network.add_connection(
                source=edge['source'],
                destination=edge['target'],
                bandwidth=edge.get('bandwidth', 100.0),
                latency=edge.get('latency', 1.0),
                status=edge.get('status', 'up'),
                bidirectional=True  # Assume bidirectional by default
            )
        
        self._env.network = network
        
        # Initialize tool simulator and fault injector with the imported network
        from netheal.tools.simulator import ToolSimulator
        from netheal.faults.injector import FaultInjector
        self._env.tool_simulator = ToolSimulator(network)
        self._env.fault_injector = FaultInjector(network)
    
    def _import_fault(self, fault_data: Dict[str, Any]) -> None:
        """Import fault information."""
        assert self._env is not None
        if not fault_data:
            return
        
        from netheal.faults.injector import FaultType, FaultInfo
        
        # Reconstruct fault info
        fault_type_str = fault_data.get('type', '')
        try:
            fault_type = FaultType(fault_type_str)
        except (ValueError, KeyError):
            # Fallback to a default fault type
            print(f"Warning: Unknown fault type '{fault_type_str}', defaulting to LINK_FAILURE")
            fault_type = FaultType.LINK_FAILURE
        
        fault_info = FaultInfo(
            fault_type=fault_type,
            location=fault_data.get('location', ''),
            details=fault_data.get('details', {}),
        )
        
        self._env.ground_truth_fault = fault_info
        
        # Reapply the fault effects to the network
        if self._env.fault_injector and fault_type == FaultType.LINK_FAILURE:
            # For link failures, extract source and destination from details
            details = fault_data.get('details', {})
            source = details.get('source')
            destination = details.get('destination')
            if source and destination and self._env.network:
                # Set the link status to down
                try:
                    self._env.network.graph[source][destination]['status'] = 'down'
                    # Also set reverse direction if it exists (bidirectional link)
                    if self._env.network.graph.has_edge(destination, source):
                        self._env.network.graph[destination][source]['status'] = 'down'
                except KeyError:
                    pass  # Link doesn't exist in graph
        elif self._env.fault_injector and fault_type == FaultType.DEVICE_FAILURE:
            # For device failures, set device status to down
            location = fault_data.get('location', '')
            if location and self._env.network and self._env.network.graph.has_node(location):
                self._env.network.graph.nodes[location]['status'] = 'down'
    
    def _import_observation(self, obs_data: Dict[str, Any]) -> None:
        """Import observation state."""
        assert self._env is not None
        if not obs_data:
            return
        
        # Recreate the observation from dict
        from netheal.environment.observation import StructuredObservation, DiagnosticResult
        from netheal.tools.simulator import ToolResult
        import time as time_module
        
        self._env.observation = StructuredObservation(
            max_devices=self._env.max_devices,
            max_history=100
        )
        
        # Set episode metadata in observation
        self._env.observation.max_episode_steps = self._env.max_episode_steps
        
        # Restore discovery matrix if available
        discovery_matrix = obs_data.get('discovery_matrix', [])
        if discovery_matrix and len(discovery_matrix) > 0:
            self._env.observation.discovery_matrix.adjacency = np.array(discovery_matrix)
            
            # Restore device_map and next_index from exported data
            discovery_device_map = obs_data.get('discovery_device_map', {})
            discovery_next_index = obs_data.get('discovery_next_index', 0)
            
            if discovery_device_map:
                # Restore the exact device_map and reverse_map
                self._env.observation.discovery_matrix.device_map = dict(discovery_device_map)
                self._env.observation.discovery_matrix.reverse_map = {
                    idx: device_id for device_id, idx in discovery_device_map.items()
                }
                self._env.observation.discovery_matrix.next_index = discovery_next_index
            elif self._env.network:
                # Fallback: rebuild from network if discovery maps not in export
                # (for backwards compatibility with old exports)
                network_devices = list(self._env.network.graph.nodes())
                for idx, device_id in enumerate(network_devices[:self._env.max_devices]):
                    self._env.observation.discovery_matrix.device_map[device_id] = idx
                    self._env.observation.discovery_matrix.reverse_map[idx] = device_id
                self._env.observation.discovery_matrix.next_index = min(
                    len(network_devices),
                    self._env.max_devices
                )
        
        # Restore diagnostic memory if available
        diagnostic_history = obs_data.get('diagnostic_history', [])
        if diagnostic_history:
            for diag_data in diagnostic_history:
                try:
                    # Reconstruct ToolResult
                    tool_result_data = diag_data.get('tool_result')
                    if tool_result_data:
                        tool_result = ToolResult(
                            tool_name=tool_result_data.get('tool_name', ''),
                            success=tool_result_data.get('success', False),
                            data=tool_result_data.get('data', {}),
                            cost=tool_result_data.get('cost', 1.0)
                        )
                        
                        # Reconstruct DiagnosticResult
                        diagnostic_result = DiagnosticResult(
                            tool_name=diag_data.get('tool_name', ''),
                            source=diag_data.get('source'),
                            destination=diag_data.get('destination'),
                            result=tool_result,
                            timestamp=diag_data.get('timestamp', time_module.time()),
                            confidence=diag_data.get('confidence', 1.0)
                        )
                        
                        # Add to memory
                        self._env.observation.diagnostic_memory.add_result(diagnostic_result)
                except Exception as e:
                    # Skip this diagnostic if we can't restore it
                    print(f"Warning: Could not restore diagnostic: {e}")
                    continue
