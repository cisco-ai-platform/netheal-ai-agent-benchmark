# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for enhanced NetworkTroubleshootingEnv class."""

import pytest
import numpy as np
from netheal.environment.env import NetworkTroubleshootingEnv, ActionType
from netheal.environment.actions import ActionCategory
from netheal.faults.injector import FaultType


class TestNetworkTroubleshootingEnv:
    """Test cases for NetworkTroubleshootingEnv class."""
    
    def test_environment_creation(self):
        """Test creating the environment."""
        env = NetworkTroubleshootingEnv(max_devices=5, max_episode_steps=10)
        
        assert env.max_devices == 5
        assert env.max_episode_steps == 10
        assert env.action_space is not None
        assert env.observation_space is not None
    
    def test_reset_environment(self):
        """Test resetting the environment."""
        env = NetworkTroubleshootingEnv(max_devices=4, max_episode_steps=10)
        
        obs, info = env.reset(seed=42)
        
        # Check observation structure
        assert isinstance(obs, dict)
        assert 'discovery_matrix' in obs
        assert 'device_status' in obs
        assert 'recent_diagnostics' in obs
        assert 'episode_metadata' in obs
        
        # Check info
        assert 'network_size' in info
        assert 'ground_truth_fault' in info
        assert 'step_count' in info
        assert info['step_count'] == 0
        
        # Check environment state
        assert env.network is not None
        assert env.ground_truth_fault is not None
        assert env.observation is not None
    
    def test_step_with_diagnostic_action(self):
        """Test taking a diagnostic action."""
        env = NetworkTroubleshootingEnv(max_devices=4, max_episode_steps=10)
        obs, info = env.reset(seed=42)
        
        # Get valid actions and take first one
        valid_actions = env.get_valid_actions()
        assert len(valid_actions) > 0
        action = valid_actions[0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check results
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert 'action_spec' in info
        assert 'reward_breakdown' in info
        
        # Should not terminate on non-diagnosis action
        if info['action_spec'] and info['action_spec']['category'] != 'diagnosis':
            assert not terminated
    
    def test_episode_termination_on_diagnosis(self):
        """Test that episode terminates on diagnosis action."""
        env = NetworkTroubleshootingEnv(max_devices=4, max_episode_steps=10)
        obs, info = env.reset(seed=42)
        
        # Find a diagnosis action from action space manager
        diagnosis_actions = []
        for action_id in range(env.action_space.n):
            action_spec = env.action_space_manager.get_action_spec(action_id)
            if action_spec and action_spec.category == ActionCategory.DIAGNOSIS:
                diagnosis_actions.append(action_id)
        
        if diagnosis_actions:
            diagnose_action = diagnosis_actions[0]
            obs, reward, terminated, truncated, info = env.step(diagnose_action)
            
            # Episode should terminate on diagnosis
            assert terminated
            
            # Should have diagnosis reward (positive or negative)
            assert 'diagnosis_reward' in info['reward_breakdown']
    
    def test_episode_truncation(self):
        """Test episode truncation at max steps."""
        env = NetworkTroubleshootingEnv(max_devices=4, max_episode_steps=3)
        obs, info = env.reset(seed=42)
        
        # Take max_episode_steps actions
        for i in range(3):
            action = i % 10  # Take some diagnostic actions
            obs, reward, terminated, truncated, info = env.step(action)
            
            if i < 2:
                assert not truncated
            else:
                assert truncated
    
    def test_action_meanings(self):
        """Test getting action meanings."""
        env = NetworkTroubleshootingEnv(max_devices=3, max_episode_steps=10)
        obs, info = env.reset(seed=42)
        
        meanings = env.get_action_meanings()
        
        assert isinstance(meanings, list)
        assert len(meanings) > 0
        
        # Check that action meanings contain expected keywords
        meaning_text = ' '.join(meanings)
        assert 'ping' in meaning_text
        assert 'diagnosis' in meaning_text
        assert 'check_status' in meaning_text
    
    def test_render_functionality(self):
        """Test rendering functionality."""
        env = NetworkTroubleshootingEnv(max_devices=3, max_episode_steps=5)
        obs, info = env.reset(seed=42)
        
        # Should not raise an error
        env.render()
        env.render("text")
    
    def test_multiple_episodes(self):
        """Test running multiple episodes."""
        env = NetworkTroubleshootingEnv(max_devices=4, max_episode_steps=5)
        
        for episode in range(3):
            obs, info = env.reset(seed=episode)
            
            # Each episode should have different network or fault
            assert env.network is not None
            assert env.ground_truth_fault is not None
            
            # Take a few actions with valid action selection
            for step in range(3):
                valid_actions = env.get_valid_actions()
                if valid_actions:
                    action = valid_actions[min(step, len(valid_actions) - 1)]
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    if terminated or truncated:
                        break
    
    def test_reward_scaling_with_network_size(self):
        """Test that the final diagnosis reward scales with network size."""
        # Helper function to run a diagnosis and get the reward
        def get_diagnosis_reward(env, correct_diagnosis, ensure_no_partial_reward=False):
            obs, info = env.reset(seed=42)
            gt_fault = info['ground_truth_fault']

            if correct_diagnosis:
                action_id = next(i for i in range(env.action_space.n) if (spec := env.action_space_manager.get_action_spec(i)) and spec.category == ActionCategory.DIAGNOSIS and spec.action_type.value == gt_fault['type'] and spec.parameters.get('location') == gt_fault['location'])
            else:
                if ensure_no_partial_reward:
                    # Find an action with completely wrong fault type and location to avoid partial rewards
                    action_id = next(i for i in range(env.action_space.n) if (spec := env.action_space_manager.get_action_spec(i)) and spec.category == ActionCategory.DIAGNOSIS and spec.action_type.value != gt_fault['type'] and spec.parameters.get('location') != gt_fault['location'] and not any(device in spec.parameters.get('location', '') for device in gt_fault['location'].replace('->', ' ').split()))
                else:
                    action_id = next(i for i in range(env.action_space.n) if (spec := env.action_space_manager.get_action_spec(i)) and spec.category == ActionCategory.DIAGNOSIS and (spec.action_type.value != gt_fault['type'] or spec.parameters.get('location') != gt_fault['location']))

            _, reward, _, _, info_step = env.step(action_id)
            return reward, info_step.get('reward_breakdown', {})

        # Test correct diagnosis reward scaling
        env_small_correct = NetworkTroubleshootingEnv(max_devices=3, max_episode_steps=10)
        env_large_correct = NetworkTroubleshootingEnv(max_devices=10, max_episode_steps=10)

        reward_small_correct, _ = get_diagnosis_reward(env_small_correct, correct_diagnosis=True)
        reward_large_correct, _ = get_diagnosis_reward(env_large_correct, correct_diagnosis=True)

        print(f"Correct diagnosis rewards -> Small: {reward_small_correct}, Large: {reward_large_correct}")
        assert reward_large_correct > reward_small_correct

        # Test that incorrect diagnosis with no device overlap gets penalty scaling
        # (avoiding partial rewards by ensuring no device overlap)
        env_small_incorrect = NetworkTroubleshootingEnv(max_devices=3, max_episode_steps=10)
        env_large_incorrect = NetworkTroubleshootingEnv(max_devices=10, max_episode_steps=10)

        try:
            penalty_small_incorrect, breakdown_small = get_diagnosis_reward(env_small_incorrect, correct_diagnosis=False, ensure_no_partial_reward=True)
            penalty_large_incorrect, breakdown_large = get_diagnosis_reward(env_large_incorrect, correct_diagnosis=False, ensure_no_partial_reward=True)

            print(f"Incorrect diagnosis penalties -> Small: {penalty_small_incorrect}, Large: {penalty_large_incorrect}")
            print(f"Breakdown small: {breakdown_small}")
            print(f"Breakdown large: {breakdown_large}")
            
            # Both should be penalties (negative) and larger networks should have larger penalties
            assert penalty_small_incorrect < 0, "Small network should get penalty for completely wrong diagnosis"
            assert penalty_large_incorrect < 0, "Large network should get penalty for completely wrong diagnosis"
            assert penalty_large_incorrect < penalty_small_incorrect, "Larger network should get larger penalty"
        except StopIteration:
            # If we can't find a completely wrong diagnosis (no device overlap), 
            # just test that partial rewards scale with network size
            reward_small_partial, breakdown_small = get_diagnosis_reward(env_small_incorrect, correct_diagnosis=False)
            reward_large_partial, breakdown_large = get_diagnosis_reward(env_large_incorrect, correct_diagnosis=False)
            
            print(f"Partial rewards -> Small: {reward_small_partial}, Large: {reward_large_partial}")
            print(f"Breakdown small: {breakdown_small}")
            print(f"Breakdown large: {breakdown_large}")
            
            # If both got partial rewards, larger network should get larger partial reward
            if breakdown_small.get('partial_device_reward', 0) > 0 and breakdown_large.get('partial_device_reward', 0) > 0:
                assert reward_large_partial > reward_small_partial, "Larger network should get larger partial reward"    
    def test_invalid_action_after_episode_end(self):
        """Test that taking action after episode end raises error."""
        env = NetworkTroubleshootingEnv(max_devices=3, max_episode_steps=2)
        obs, info = env.reset(seed=42)
        
        # End episode quickly by taking max steps
        for _ in range(2):
            valid_actions = env.get_valid_actions()
            if valid_actions:
                action = valid_actions[0]
                obs, reward, terminated, truncated, info = env.step(action)
        
        # Episode should be done
        assert terminated or truncated
        
        # Taking another action should raise error
        with pytest.raises(RuntimeError):
            env.step(0)
    
    def test_observation_space_consistency(self):
        """Test that observations are consistent with observation space."""
        env = NetworkTroubleshootingEnv(max_devices=4, max_episode_steps=10)
        
        for _ in range(3):
            obs, info = env.reset()
            
            # Check observation structure
            assert isinstance(obs, dict)
            assert all(key in obs for key in ['discovery_matrix', 'device_status', 
                                            'recent_diagnostics', 'episode_metadata'])
            
            # Take a few steps with valid actions
            for _ in range(3):
                valid_actions = env.get_valid_actions()
                if valid_actions:
                    action = valid_actions[0]
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    assert isinstance(obs, dict)
                    assert all(key in obs for key in ['discovery_matrix', 'device_status', 
                                                    'recent_diagnostics', 'episode_metadata'])
                
                if terminated or truncated:
                    break
