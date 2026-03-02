"""
Base interface for Vision-Language-Action (VLA) policies.

This module defines the abstract interface that all VLA models must implement
to be used with the g1-vla deployment system. This ensures the codebase remains
model-agnostic and can easily switch between different VLA implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ModalitySpec:
    """Specification for a single modality (video, state, action, language)."""

    keys: list[str]
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    dtypes: dict[str, np.dtype] = field(default_factory=dict)
    delta_indices: list[int] = field(default_factory=lambda: [0])


@dataclass
class VLAPolicyConfig:
    """Base configuration for VLA policies."""

    model_name: str
    action_horizon: int = 16
    device: str = "cuda"


class BaseVLAPolicy(ABC):
    """
    Abstract base class for Vision-Language-Action policies.

    All VLA model implementations must inherit from this class and implement
    the required abstract methods. This ensures a consistent interface across
    different models (GR00T, OpenVLA, RT-2, etc.).

    The policy interface follows a simple pattern:
    1. Initialize with configuration
    2. Call get_action() with observations to get actions
    3. Optionally reset() between episodes

    Example:
        policy = SomeVLAPolicy.from_config(config)
        action = policy.get_action(observation)
    """

    @abstractmethod
    def get_action(
        self, observation: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Get action predictions from observations.

        Args:
            observation: Dictionary containing:
                - video: Dict[str, np.ndarray] - Camera images
                    Shape: (B, T, H, W, C) or (H, W, C)
                - state: Dict[str, np.ndarray] - Robot state
                    Shape: (B, T, D) or (D,)
                - language: Dict[str, list] - Task instructions
                    Format: [["instruction"]] or "instruction"

        Returns:
            Tuple of:
                - action: Dict[str, np.ndarray] - Predicted actions
                    Shape: (B, action_horizon, D) or (action_horizon, D)
                - info: Dict[str, Any] - Additional information (timing, etc.)
        """
        pass

    @abstractmethod
    def get_modality_config(self) -> dict[str, ModalitySpec]:
        """
        Get the modality configuration for this policy.

        Returns:
            Dictionary with keys: "video", "state", "action", "language"
            Each containing a ModalitySpec with expected keys and shapes.
        """
        pass

    @abstractmethod
    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Reset the policy state between episodes.

        Args:
            options: Optional configuration for reset

        Returns:
            Info dictionary with reset status
        """
        pass

    @property
    @abstractmethod
    def action_horizon(self) -> int:
        """Number of future action steps predicted by the policy."""
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the policy is connected and ready for inference."""
        pass

    def close(self) -> None:
        """Clean up resources. Override if needed."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
