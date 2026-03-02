#!/usr/bin/env python3
"""
Test GR00T N1.6 inference with the Unitree G1 checkpoint.

This script connects to a running GR00T policy server and tests inference
with dummy observations to verify the setup is working correctly.

Usage:
    # First, start the policy server (in Isaac-GR00T directory):
    # .venv/bin/python gr00t/eval/run_gr00t_server.py \
    #     --model-path nvidia/GR00T-N1.6-G1-PnPAppleToPlate \
    #     --embodiment-tag UNITREE_G1 \
    #     --port 5556

    # Then run this test:
    python scripts/test_groot_inference.py --port 5556
"""

import argparse
import sys
import time

import numpy as np

# Add Isaac-GR00T to path if needed
GROOT_PATH = "/home/jalil/Developer/Perceptron/Isaac-GR00T"
sys.path.insert(0, GROOT_PATH)

from gr00t.policy.server_client import PolicyClient


def create_dummy_observation(config: dict) -> dict:
    """Create a dummy observation matching the modality config."""
    B = 1  # batch size
    T = 1  # temporal horizon

    observation = {
        "video": {
            "ego_view": np.random.randint(0, 255, size=(B, T, 480, 640, 3), dtype=np.uint8)
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
    return observation


def main():
    parser = argparse.ArgumentParser(description="Test GR00T N1.6 inference")
    parser.add_argument("--host", type=str, default="localhost", help="Policy server host")
    parser.add_argument("--port", type=int, default=5556, help="Policy server port")
    parser.add_argument(
        "--num-runs", type=int, default=3, help="Number of inference runs for timing"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GR00T N1.6 Inference Test")
    print("=" * 60)
    print(f"Connecting to server at {args.host}:{args.port}...")

    # Connect to server
    client = PolicyClient(host=args.host, port=args.port, strict=False)

    # Ping server
    if not client.ping():
        print("ERROR: Cannot connect to policy server!")
        print("Make sure the server is running with:")
        print(
            "  cd Isaac-GR00T && .venv/bin/python gr00t/eval/run_gr00t_server.py \\"
        )
        print("      --model-path nvidia/GR00T-N1.6-G1-PnPAppleToPlate \\")
        print("      --embodiment-tag UNITREE_G1 \\")
        print(f"      --port {args.port}")
        sys.exit(1)

    print("Connected!")
    print()

    # Get modality config
    print("Modality Configuration:")
    print("-" * 40)
    config = client.get_modality_config()

    video_keys = config["video"].modality_keys
    state_keys = config["state"].modality_keys
    action_keys = config["action"].modality_keys
    action_horizon = len(config["action"].delta_indices)

    print(f"  Video inputs: {video_keys}")
    print(f"  State inputs: {state_keys}")
    print(f"  Action outputs: {action_keys}")
    print(f"  Action horizon: {action_horizon} steps")
    print()

    # Create dummy observation
    observation = create_dummy_observation(config)

    # Run inference
    print(f"Running {args.num_runs} inference calls...")
    print("-" * 40)

    times = []
    for i in range(args.num_runs):
        start = time.time()
        action, info = client.get_action(observation)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i + 1}: {elapsed * 1000:.1f} ms")

    avg_time = np.mean(times)
    freq = 1.0 / avg_time

    print()
    print("Results:")
    print("-" * 40)
    print(f"  Average inference time: {avg_time * 1000:.1f} ms")
    print(f"  Inference frequency: {freq:.1f} Hz")
    print()

    # Print action shapes
    print("Action Output Shapes:")
    print("-" * 40)
    for key, value in action.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}")
            print(f"    range: [{value.min():.4f}, {value.max():.4f}]")

    print()
    print("=" * 60)
    print("Inference test PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
