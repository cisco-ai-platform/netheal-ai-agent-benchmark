# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Basic example for scenario export/import functionality."""
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_basic_export_import():
    print("Testing scenario export/import functionality...")

    try:
        from webapp.backend.app.manager import EnvManager

        # Create manager and reset environment
        print("\n1. Creating environment...")
        manager = EnvManager()
        state1 = manager.reset(seed=42, max_devices=5, max_episode_steps=10)
        print(f"   [OK] Environment created with {state1['info']['network_size']} devices")

        # Take a step
        valid_actions = state1["valid_actions"]
        if valid_actions:
            print(f"\n2. Taking action {valid_actions[0]}...")
            manager.step(valid_actions[0])
            print("   [OK] Action completed")

        # Export scenario
        print("\n3. Exporting scenario...")
        scenario = manager.export_scenario()
        print(f"   [OK] Scenario exported (version {scenario['version']})")
        print(f"   - Network has {len(scenario['network']['nodes'])} nodes")
        print(f"   - Network has {len(scenario['network']['edges'])} edges")
        print(f"   - Fault type: {scenario['fault'].get('type', 'unknown')}")
        print(f"   - Fault location: {scenario['fault'].get('location', 'unknown')}")
        print(f"   - Fault details: {scenario['fault'].get('details', {})}")

        # Verify JSON serialization
        print("\n4. Testing JSON serialization...")
        json_str = json.dumps(scenario, indent=2)
        json.loads(json_str)
        print(f"   [OK] Scenario is JSON serializable ({len(json_str)} bytes)")

        # Import scenario
        print("\n5. Importing scenario...")
        state2 = manager.import_scenario(scenario)
        print("   [OK] Scenario imported successfully")
        print(f"   - Network size: {state2['info']['network_size']}")
        print(f"   - Valid actions: {len(state2['valid_actions'])}")

        # Export again to verify preservation
        print("\n6. Verifying preservation...")
        scenario2 = manager.export_scenario()
        nodes_match = len(scenario["network"]["nodes"]) == len(scenario2["network"]["nodes"])
        edges_match = len(scenario["network"]["edges"]) == len(scenario2["network"]["edges"])
        fault_type_match = scenario["fault"]["type"] == scenario2["fault"]["type"]

        print(f"   - Nodes preserved: {nodes_match}")
        print(f"   - Edges preserved: {edges_match}")
        print(f"   - Fault type preserved: {fault_type_match}")

        if nodes_match and edges_match and fault_type_match:
            print("\nAll checks passed!")
            return True

        print("\nSome preservation checks failed")
        return False
    except Exception as exc:
        print(f"\nTest failed with error: {exc}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_basic_export_import()
    sys.exit(0 if success else 1)
