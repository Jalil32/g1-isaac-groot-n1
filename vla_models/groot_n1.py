"""
GR00T N1.6 VLA Policy implementation.

This module provides a wrapper around NVIDIA's Isaac-GR00T policy server,
implementing the BaseVLAPolicy interface for use with the g1-vla deployment system.

The GR00T N1.6 model uses a server-client architecture:
- Server: Runs on a GPU machine, loads the model, handles inference
- Client: Connects via ZMQ, sends observations, receives actions

Usage:
    # Start the server first (in Isaac-GR00T directory):
    # uv run python gr00t/eval/run_gr00t_server.py \
    #     --model-path nvidia/GR00T-N1.6-G1-PnPAppleToPlate \
    #     --embodiment-tag UNITREE_G1 \
    #     --port 5556

    # Then use the policy:
    config = GR00TN1Config(
        server_host="localhost",
        server_port=5556,
        embodiment_tag="UNITREE_G1",
    )
    policy = GR00TN1Policy(config)
    action, info = policy.get_action(observation)
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseVLAPolicy, ModalitySpec, VLAPolicyConfig

# Path to Isaac-GR00T installation
ISAAC_GROOT_PATH = Path("/home/jalil/Developer/Perceptron/Isaac-GR00T")


@dataclass
class GR00TN1Config(VLAPolicyConfig):
    """Configuration for GR00T N1.6 policy."""

    model_name: str = "groot-n1.6"
    server_host: str = "localhost"
    server_port: int = 5556
    timeout_ms: int = 15000
    embodiment_tag: str = "UNITREE_G1"
    checkpoint_path: str = "nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
    action_horizon: int = 30  # GR00T N1.6 uses 30-step horizon
    device: str = "cuda"


class GR00TN1Policy(BaseVLAPolicy):
    """
    GR00T N1.6 VLA Policy implementation.

    This policy connects to a running GR00T policy server and forwards
    inference requests. The server handles model loading and GPU inference.

    The observation format expected by GR00T N1.6 (UNITREE_G1 embodiment):
        {
            "video": {
                "ego_view": np.ndarray (B, T, H, W, 3) uint8
            },
            "state": {
                "left_leg": np.ndarray (B, T, 6) float32
                "right_leg": np.ndarray (B, T, 6) float32
                "waist": np.ndarray (B, T, 3) float32
                "left_arm": np.ndarray (B, T, 7) float32
                "right_arm": np.ndarray (B, T, 7) float32
                "left_hand": np.ndarray (B, T, 7) float32
                "right_hand": np.ndarray (B, T, 7) float32
            },
            "language": {
                "annotation.human.task_description": [["task"]]
            }
        }

    The action output:
        {
            "left_arm": np.ndarray (B, 30, 7) - relative
            "right_arm": np.ndarray (B, 30, 7) - relative
            "left_hand": np.ndarray (B, 30, 7) - absolute
            "right_hand": np.ndarray (B, 30, 7) - absolute
            "waist": np.ndarray (B, 30, 3) - absolute
            "base_height_command": np.ndarray (B, 30, 1)
            "navigate_command": np.ndarray (B, 30, 3)
        }
    """

    def __init__(self, config: GR00TN1Config):
        self.config = config
        self._client = None
        self._modality_config = None
        self._action_horizon = config.action_horizon
        self._connect()

    def _connect(self) -> None:
        """Connect to the GR00T policy server."""
        # Add Isaac-GR00T to path for imports
        if str(ISAAC_GROOT_PATH) not in sys.path:
            sys.path.insert(0, str(ISAAC_GROOT_PATH))

        try:
            from gr00t.policy.server_client import PolicyClient

            self._client = PolicyClient(
                host=self.config.server_host,
                port=self.config.server_port,
                timeout_ms=self.config.timeout_ms,
                strict=False,
            )

            # Verify connection
            if not self._client.ping():
                raise ConnectionError(
                    f"Cannot connect to GR00T server at "
                    f"{self.config.server_host}:{self.config.server_port}"
                )

            # Cache modality config
            self._modality_config = self._client.get_modality_config()

        except ImportError as e:
            raise ImportError(
                f"Failed to import gr00t. Ensure Isaac-GR00T is installed at "
                f"{ISAAC_GROOT_PATH}. Error: {e}"
            )

    def get_action(
        self, observation: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Get action predictions from the GR00T server.

        Args:
            observation: Observation dictionary (see class docstring for format)

        Returns:
            Tuple of (action_dict, info_dict)
        """
        if self._client is None:
            raise RuntimeError("Policy not connected. Call connect() first.")

        # Ensure observation is in the correct format
        formatted_obs = self._format_observation(observation)

        # Time the inference
        start_time = time.time()
        action, info = self._client.get_action(formatted_obs)
        inference_time = time.time() - start_time

        info["inference_time_ms"] = inference_time * 1000
        info["model"] = self.config.model_name

        return action, info

    def _format_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """
        Format observation to match GR00T's expected input structure.

        Handles both batched (B, T, ...) and unbatched (...) inputs.
        """
        formatted = {"video": {}, "state": {}, "language": {}}

        # Process video
        if "video" in observation:
            for key, value in observation["video"].items():
                formatted["video"][key] = self._ensure_batch_time_dims(value, is_video=True)
        elif "video.ego_view" in observation:
            # Flat key format
            formatted["video"]["ego_view"] = self._ensure_batch_time_dims(
                observation["video.ego_view"], is_video=True
            )

        # Process state
        if "state" in observation:
            for key, value in observation["state"].items():
                formatted["state"][key] = self._ensure_batch_time_dims(value)
        else:
            # Check for flat key format (state.left_arm, etc.)
            state_keys = [
                "left_leg", "right_leg", "waist",
                "left_arm", "right_arm", "left_hand", "right_hand"
            ]
            for key in state_keys:
                flat_key = f"state.{key}"
                if flat_key in observation:
                    formatted["state"][key] = self._ensure_batch_time_dims(
                        observation[flat_key]
                    )

        # Process language
        if "language" in observation:
            formatted["language"] = observation["language"]
        elif "annotation.human.task_description" in observation:
            task = observation["annotation.human.task_description"]
            if isinstance(task, str):
                formatted["language"]["annotation.human.task_description"] = [[task]]
            else:
                formatted["language"]["annotation.human.task_description"] = task

        return formatted

    def _ensure_batch_time_dims(
        self, arr: np.ndarray, is_video: bool = False
    ) -> np.ndarray:
        """Ensure array has batch and time dimensions."""
        arr = np.asarray(arr)

        if is_video:
            # Video expected: (B, T, H, W, C)
            if arr.ndim == 3:  # (H, W, C)
                arr = arr[np.newaxis, np.newaxis, ...]  # Add B and T
            elif arr.ndim == 4:  # (T, H, W, C) or (B, H, W, C)
                arr = arr[np.newaxis, ...]  # Assume missing B
        else:
            # State expected: (B, T, D)
            if arr.ndim == 1:  # (D,)
                arr = arr[np.newaxis, np.newaxis, ...]  # Add B and T
            elif arr.ndim == 2:  # (T, D) or (B, D)
                arr = arr[np.newaxis, ...]  # Assume missing B

        return arr

    def get_modality_config(self) -> dict[str, ModalitySpec]:
        """Get the modality configuration from the server."""
        if self._modality_config is None:
            self._modality_config = self._client.get_modality_config()

        # Convert to our ModalitySpec format
        config = {}
        for modality, mc in self._modality_config.items():
            config[modality] = ModalitySpec(
                keys=mc.modality_keys,
                delta_indices=mc.delta_indices,
            )
        return config

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the policy state."""
        if self._client is None:
            return {"status": "not_connected"}

        return self._client.reset(options)

    @property
    def action_horizon(self) -> int:
        """Number of future action steps (30 for GR00T N1.6)."""
        return self._action_horizon

    @property
    def is_connected(self) -> bool:
        """Check if connected to the server."""
        if self._client is None:
            return False
        try:
            return self._client.ping()
        except Exception:
            return False

    def close(self) -> None:
        """Close the connection to the server."""
        self._client = None
        self._modality_config = None

    @classmethod
    def from_config(cls, config: GR00TN1Config) -> "GR00TN1Policy":
        """Create a policy instance from configuration."""
        return cls(config)

    def __repr__(self) -> str:
        return (
            f"GR00TN1Policy("
            f"server={self.config.server_host}:{self.config.server_port}, "
            f"connected={self.is_connected})"
        )
