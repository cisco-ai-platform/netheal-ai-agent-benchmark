# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Sparse reward system for the NetHeal environment.

This module provides a sparse reward signal, focusing on the final outcome
of the troubleshooting process rather than intermediate achievements.
Enhanced with partial rewards for correct device identification.
"""

from typing import Dict, Tuple, List

from .actions import ActionSpec, ActionCategory
from ..faults.injector import FaultInfo


class SparseRewardCalculator:
    """
    A sparse reward calculator that provides rewards only for the final diagnosis.

    It encourages efficiency by applying a small, constant penalty for each action
    taken, and provides a large reward or penalty based on the accuracy of the
    final diagnosis. Enhanced with partial rewards for correct device identification.
    """

    def __init__(self, scaling_factor: float = 10.0):
        """
        Initialize the sparse reward calculator.

        Args:
            scaling_factor: A factor to scale the diagnosis reward based on
                network size. A larger factor results in a smaller reward increase.
        """
        self.correct_diagnosis_reward: float = 10.0
        self.incorrect_diagnosis_penalty: float = -10.0
        self.step_penalty: float = -0.1
        self.scaling_factor: float = scaling_factor
        self.partial_device_reward_rate: float = 0.15  # 15% of full reward per correct device
        self.single_device_reward_rate: float = 0.30   # 30% of full reward for single device scenarios

    def calculate_reward(
        self, action_spec: ActionSpec, ground_truth: FaultInfo, network_size: int
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate the total reward and a breakdown of its components.
        Enhanced with partial rewards for correct device identification.

        Args:
            action_spec: The specification of the action taken by the agent.
            ground_truth: The ground truth information about the injected fault.
            network_size: The size of the network for scaling rewards.

        Returns:
            A tuple containing the total reward and a dictionary with the
            reward breakdown.
        """
        reward = 0.0
        breakdown = {
            "step_penalty": 0.0,
            "diagnosis_reward": 0.0,
            "partial_device_reward": 0.0,
        }

        # Apply a constant step penalty to encourage efficiency
        reward += self.step_penalty
        breakdown["step_penalty"] = self.step_penalty

        # If the action is a final diagnosis, calculate the outcome reward
        if action_spec.category == ActionCategory.DIAGNOSIS:
            diagnosed_fault = action_spec.action_type
            diagnosed_location = action_spec.parameters.get("location")

            # Check if the diagnosis is completely correct
            if (
                diagnosed_fault == ground_truth.fault_type
                and diagnosed_location == ground_truth.location
            ):
                base_reward = self.correct_diagnosis_reward
                # Scale the reward based on the network size to account for difficulty
                scaling_multiplier = 1.0 + (network_size / self.scaling_factor)
                diagnosis_reward = base_reward * scaling_multiplier
                reward += diagnosis_reward
                breakdown["diagnosis_reward"] = diagnosis_reward
            else:
                # Check for partial rewards based on correct device identification
                partial_reward = self._calculate_partial_device_reward(
                    diagnosed_location, ground_truth.location, network_size
                )
                
                if partial_reward > 0:
                    reward += partial_reward
                    breakdown["partial_device_reward"] = partial_reward
                else:
                    # No partial reward, apply penalty
                    base_penalty = self.incorrect_diagnosis_penalty
                    scaling_multiplier = 1.0 + (network_size / self.scaling_factor)
                    diagnosis_penalty = base_penalty * scaling_multiplier
                    reward += diagnosis_penalty
                    breakdown["diagnosis_reward"] = diagnosis_penalty

        return reward, breakdown

    def _calculate_partial_device_reward(
        self, diagnosed_location: str, ground_truth_location: str, network_size: int
    ) -> float:
        """
        Calculate partial reward for correct device identification.
        
        Args:
            diagnosed_location: The location diagnosed by the agent
            ground_truth_location: The actual fault location
            network_size: Size of the network for scaling
            
        Returns:
            Partial reward amount (0 if no devices match)
        """
        if not diagnosed_location or not ground_truth_location:
            return 0.0
            
        # Extract devices from locations
        diagnosed_devices = self._extract_devices_from_location(diagnosed_location)
        ground_truth_devices = self._extract_devices_from_location(ground_truth_location)
        
        if not ground_truth_devices:
            return 0.0
            
        # Count correct device matches
        correct_devices = len(set(diagnosed_devices) & set(ground_truth_devices))
        
        if correct_devices == 0:
            return 0.0
            
        # Calculate partial reward
        base_reward = self.correct_diagnosis_reward
        scaling_multiplier = 1.0 + (network_size / self.scaling_factor)
        
        if len(ground_truth_devices) == 1:
            # Single device scenario: 30% reward if identified correctly
            if correct_devices == 1:
                partial_reward = base_reward * self.single_device_reward_rate * scaling_multiplier
            else:
                partial_reward = 0.0
        else:
            # Multiple device scenario: 15% reward per correct device
            partial_reward = base_reward * self.partial_device_reward_rate * correct_devices * scaling_multiplier
            
        return partial_reward
    
    def _extract_devices_from_location(self, location: str) -> List[str]:
        """
        Extract device names from a location string.
        
        Args:
            location: Location string (device name or connection like "device1->device2")
            
        Returns:
            List of device names involved in the location
        """
        if not location:
            return []
            
        # Handle connection format (e.g., "device1->device2")
        if '->' in location:
            parts = location.split('->')
            return [part.strip() for part in parts if part.strip()]
        else:
            # Single device
            return [location.strip()] if location.strip() else []
