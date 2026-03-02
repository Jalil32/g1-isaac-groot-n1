"""
VLA Models - Modular Vision-Language-Action Policy Interface

This package provides a model-agnostic interface for VLA policies used in
robotic manipulation tasks. It allows easy switching between different VLA
implementations (GR00T, OpenVLA, RT-2, etc.) without changing deployment code.

Quick Start:
    from vla_models import create_policy, list_available_policies

    # See available policies
    print(list_available_policies())  # ['groot-n1.6']

    # Create a policy
    policy = create_policy("groot-n1.6", server_port=5556)

    # Use the policy
    action, info = policy.get_action(observation)

Architecture:
    BaseVLAPolicy (ABC)
        │
        ├── GR00TN1Policy  - NVIDIA GR00T N1.6 (via server-client)
        │
        └── [Future models can be added here]

Adding New Models:
    1. Create a new module (e.g., vla_models/my_model.py)
    2. Implement BaseVLAPolicy and VLAPolicyConfig
    3. Register with: register_policy("my-model", MyPolicy, MyConfig)
"""

from .base import BaseVLAPolicy, ModalitySpec, VLAPolicyConfig
from .factory import (
    create_policy,
    create_policy_from_dict,
    get_policy_info,
    list_available_policies,
    register_policy,
    unregister_policy,
)

# Conditionally import GR00T if available
try:
    from .groot_n1 import GR00TN1Config, GR00TN1Policy
except ImportError:
    GR00TN1Policy = None
    GR00TN1Config = None

__all__ = [
    # Base classes
    "BaseVLAPolicy",
    "VLAPolicyConfig",
    "ModalitySpec",
    # Factory functions
    "create_policy",
    "create_policy_from_dict",
    "list_available_policies",
    "get_policy_info",
    "register_policy",
    "unregister_policy",
    # Implementations
    "GR00TN1Policy",
    "GR00TN1Config",
]
