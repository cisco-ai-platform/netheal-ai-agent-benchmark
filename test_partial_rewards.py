#!/usr/bin/env python3
"""
Test script to verify the partial reward system for device identification.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from netheal.environment.rewards import SparseRewardCalculator
from netheal.environment.actions import ActionSpec, ActionCategory
from netheal.faults.injector import FaultInfo, FaultType


def test_partial_rewards():
    """Test the partial reward system with various scenarios."""
    calculator = SparseRewardCalculator(scaling_factor=10.0)
    network_size = 5
    
    print("=== Testing Partial Reward System ===\n")
    
    # Test case 1: Complete correct diagnosis (baseline)
    print("Test 1: Complete correct diagnosis")
    ground_truth = FaultInfo(FaultType.DEVICE_FAILURE, "device_1")
    action_spec = ActionSpec(
        category=ActionCategory.DIAGNOSIS,
        action_type=FaultType.DEVICE_FAILURE,
        parameters={"location": "device_1"},
        description="Test diagnosis"
    )
    reward, breakdown = calculator.calculate_reward(action_spec, ground_truth, network_size)
    print(f"  Ground truth: {ground_truth}")
    print(f"  Diagnosis: {action_spec.action_type.value} at {action_spec.parameters['location']}")
    print(f"  Reward: {reward:.2f}")
    print(f"  Breakdown: {breakdown}")
    print()
    
    # Test case 2: Wrong fault type, correct single device (30% reward)
    print("Test 2: Wrong fault type, correct single device (should get 30% reward)")
    ground_truth = FaultInfo(FaultType.DEVICE_FAILURE, "device_1")
    action_spec = ActionSpec(
        category=ActionCategory.DIAGNOSIS,
        action_type=FaultType.MISCONFIGURATION,  # Wrong fault type
        parameters={"location": "device_1"},     # Correct device
        description="Test diagnosis"
    )
    reward, breakdown = calculator.calculate_reward(action_spec, ground_truth, network_size)
    print(f"  Ground truth: {ground_truth}")
    print(f"  Diagnosis: {action_spec.action_type.value} at {action_spec.parameters['location']}")
    print(f"  Reward: {reward:.2f}")
    print(f"  Breakdown: {breakdown}")
    print()
    
    # Test case 3: Wrong fault type, wrong device (penalty)
    print("Test 3: Wrong fault type, wrong device (should get penalty)")
    ground_truth = FaultInfo(FaultType.DEVICE_FAILURE, "device_1")
    action_spec = ActionSpec(
        category=ActionCategory.DIAGNOSIS,
        action_type=FaultType.MISCONFIGURATION,  # Wrong fault type
        parameters={"location": "device_2"},     # Wrong device
        description="Test diagnosis"
    )
    reward, breakdown = calculator.calculate_reward(action_spec, ground_truth, network_size)
    print(f"  Ground truth: {ground_truth}")
    print(f"  Diagnosis: {action_spec.action_type.value} at {action_spec.parameters['location']}")
    print(f"  Reward: {reward:.2f}")
    print(f"  Breakdown: {breakdown}")
    print()
    
    # Test case 4: Link failure with one correct device (15% reward)
    print("Test 4: Link failure with one correct device (should get 15% reward)")
    ground_truth = FaultInfo(FaultType.LINK_FAILURE, "device_1->device_2")
    action_spec = ActionSpec(
        category=ActionCategory.DIAGNOSIS,
        action_type=FaultType.MISCONFIGURATION,  # Wrong fault type
        parameters={"location": "device_1->device_3"},  # One correct device
        description="Test diagnosis"
    )
    reward, breakdown = calculator.calculate_reward(action_spec, ground_truth, network_size)
    print(f"  Ground truth: {ground_truth}")
    print(f"  Diagnosis: {action_spec.action_type.value} at {action_spec.parameters['location']}")
    print(f"  Reward: {reward:.2f}")
    print(f"  Breakdown: {breakdown}")
    print()
    
    # Test case 5: Link failure with both correct devices (30% reward)
    print("Test 5: Link failure with both correct devices (should get 30% reward)")
    ground_truth = FaultInfo(FaultType.LINK_FAILURE, "device_1->device_2")
    action_spec = ActionSpec(
        category=ActionCategory.DIAGNOSIS,
        action_type=FaultType.MISCONFIGURATION,  # Wrong fault type
        parameters={"location": "device_2->device_1"},  # Both correct devices (reversed order)
        description="Test diagnosis"
    )
    reward, breakdown = calculator.calculate_reward(action_spec, ground_truth, network_size)
    print(f"  Ground truth: {ground_truth}")
    print(f"  Diagnosis: {action_spec.action_type.value} at {action_spec.parameters['location']}")
    print(f"  Reward: {reward:.2f}")
    print(f"  Breakdown: {breakdown}")
    print()
    
    # Test case 6: Non-diagnosis action (should only get step penalty)
    print("Test 6: Non-diagnosis action (should only get step penalty)")
    from netheal.environment.actions import DiagnosticAction
    action_spec = ActionSpec(
        category=ActionCategory.DIAGNOSTIC,
        action_type=DiagnosticAction.PING,
        parameters={"source": "device_1", "destination": "device_2"},
        description="Test ping"
    )
    reward, breakdown = calculator.calculate_reward(action_spec, ground_truth, network_size)
    print(f"  Action: {action_spec.action_type.value}")
    print(f"  Reward: {reward:.2f}")
    print(f"  Breakdown: {breakdown}")
    print()


def test_device_extraction():
    """Test the device extraction helper function."""
    calculator = SparseRewardCalculator()
    
    print("=== Testing Device Extraction ===\n")
    
    test_cases = [
        ("device_1", ["device_1"]),
        ("device_1->device_2", ["device_1", "device_2"]),
        ("router_A->switch_B", ["router_A", "switch_B"]),
        ("", []),
        ("single_device", ["single_device"]),
    ]
    
    for location, expected in test_cases:
        result = calculator._extract_devices_from_location(location)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{location}' -> {result} (expected: {expected})")
    
    print()


if __name__ == "__main__":
    test_device_extraction()
    test_partial_rewards()
    print("=== All tests completed ===")
