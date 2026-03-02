#!/usr/bin/env python3
"""
Test the modular VLA policy integration.

This script tests the VLA policy abstraction layer by:
1. Listing available policies
2. Creating a GR00T N1.6 policy via the factory
3. Running inference with dummy observations
4. Verifying the output format

Prerequisites:
    Start the GR00T policy server first:
    cd Isaac-GR00T && .venv/bin/python gr00t/eval/run_gr00t_server.py \
        --model-path nvidia/GR00T-N1.6-G1-PnPAppleToPlate \
        --embodiment-tag UNITREE_G1 \
        --port 5556

Usage:
    python scripts/test_vla_integration.py --port 5556
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vla_models import (
    BaseVLAPolicy,
    create_policy,
    get_policy_info,
    list_available_policies,
)


def create_dummy_observation() -> dict:
    """Create a dummy observation for testing."""
    B, T = 1, 1
    H, W = 480, 640

    return {
        "video": {
            "ego_view": np.random.randint(0, 255, (B, T, H, W, 3), dtype=np.uint8)
        },
        "state": {
            "left_leg": np.zeros((B, T, 6), dtype=np.float32),
            "right_leg": np.zeros((B, T, 6), dtype=np.float32),
            "waist": np.zeros((B, T, 3), dtype=np.float32),
            "left_arm": np.zeros((B, T, 7), dtype=np.float32),
            "right_arm": np.zeros((B, T, 7), dtype=np.float32),
            "left_hand": np.zeros((B, T, 7), dtype=np.float32),
            "right_hand": np.zeros((B, T, 7), dtype=np.float32),
        },
        "language": {
            "annotation.human.task_description": [
                ["pick up the apple and place it on the plate"]
            ]
        },
    }


def test_policy_interface(policy: BaseVLAPolicy) -> bool:
    """Test that the policy implements the required interface."""
    print("Testing policy interface...")

    # Check required properties
    assert hasattr(policy, "action_horizon"), "Missing action_horizon property"
    assert hasattr(policy, "is_connected"), "Missing is_connected property"
    print(f"  action_horizon: {policy.action_horizon}")
    print(f"  is_connected: {policy.is_connected}")

    # Check required methods
    assert hasattr(policy, "get_action"), "Missing get_action method"
    assert hasattr(policy, "get_modality_config"), "Missing get_modality_config method"
    assert hasattr(policy, "reset"), "Missing reset method"
    print("  All required methods present")

    return True


def test_modality_config(policy: BaseVLAPolicy) -> bool:
    """Test the modality configuration."""
    print("\nTesting modality configuration...")

    config = policy.get_modality_config()
    assert isinstance(config, dict), "get_modality_config should return dict"

    for modality in ["video", "state", "action", "language"]:
        if modality in config:
            print(f"  {modality}:")
            print(f"    keys: {config[modality].keys}")

    return True


def test_inference(policy: BaseVLAPolicy, num_runs: int = 3) -> bool:
    """Test inference with dummy observations."""
    print(f"\nTesting inference ({num_runs} runs)...")

    observation = create_dummy_observation()
    times = []

    for i in range(num_runs):
        start = time.time()
        action, info = policy.get_action(observation)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i + 1}: {elapsed * 1000:.1f} ms")

    avg_time = np.mean(times)
    print(f"  Average: {avg_time * 1000:.1f} ms ({1 / avg_time:.1f} Hz)")

    # Verify action format
    print("\nAction output:")
    for key, value in action.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, range=[{value.min():.3f}, {value.max():.3f}]")

    return True


def main():
    parser = argparse.ArgumentParser(description="Test VLA policy integration")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--model", type=str, default="groot-n1.6")
    parser.add_argument("--num-runs", type=int, default=3)
    args = parser.parse_args()

    print("=" * 60)
    print("VLA Policy Integration Test")
    print("=" * 60)

    # List available policies
    print("\nAvailable policies:")
    for name in list_available_policies():
        info = get_policy_info(name)
        print(f"  - {name}: {info['policy_class']}")

    # Create policy via factory
    print(f"\nCreating '{args.model}' policy...")
    try:
        policy = create_policy(
            args.model,
            server_host=args.host,
            server_port=args.port,
        )
        print(f"  {policy}")
    except Exception as e:
        print(f"ERROR: Failed to create policy: {e}")
        print("\nMake sure the GR00T server is running:")
        print("  cd Isaac-GR00T && .venv/bin/python gr00t/eval/run_gr00t_server.py \\")
        print("      --model-path nvidia/GR00T-N1.6-G1-PnPAppleToPlate \\")
        print("      --embodiment-tag UNITREE_G1 \\")
        print(f"      --port {args.port}")
        sys.exit(1)

    # Run tests
    all_passed = True
    try:
        all_passed &= test_policy_interface(policy)
        all_passed &= test_modality_config(policy)
        all_passed &= test_inference(policy, args.num_runs)
    finally:
        policy.close()

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
