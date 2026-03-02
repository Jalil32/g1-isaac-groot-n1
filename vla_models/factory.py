"""
VLA Policy Factory for creating and managing VLA model instances.

This module provides a registry-based factory pattern for creating VLA policies.
New models can be registered and instantiated by name, making it easy to switch
between different VLA implementations without changing deployment code.

Usage:
    from vla_models import create_policy, list_available_policies

    # List available policies
    print(list_available_policies())

    # Create a policy by name
    policy = create_policy("groot-n1.6", server_port=5556)

    # Or with full config
    from vla_models.groot_n1 import GR00TN1Config
    config = GR00TN1Config(server_port=5556)
    policy = create_policy("groot-n1.6", config=config)
"""

from typing import Any, Callable, Type

from .base import BaseVLAPolicy, VLAPolicyConfig

# Registry of available policy types
_POLICY_REGISTRY: dict[str, tuple[Type[BaseVLAPolicy], Type[VLAPolicyConfig]]] = {}


def register_policy(
    name: str,
    policy_class: Type[BaseVLAPolicy],
    config_class: Type[VLAPolicyConfig],
) -> None:
    """
    Register a new VLA policy type.

    Args:
        name: Unique identifier for the policy (e.g., "groot-n1.6", "openvla")
        policy_class: The policy class to instantiate
        config_class: The configuration class for the policy
    """
    if name in _POLICY_REGISTRY:
        raise ValueError(f"Policy '{name}' is already registered")
    _POLICY_REGISTRY[name] = (policy_class, config_class)


def unregister_policy(name: str) -> None:
    """Remove a policy from the registry."""
    if name in _POLICY_REGISTRY:
        del _POLICY_REGISTRY[name]


def list_available_policies() -> list[str]:
    """List all registered policy names."""
    return list(_POLICY_REGISTRY.keys())


def get_policy_info(name: str) -> dict[str, Any]:
    """
    Get information about a registered policy.

    Args:
        name: Policy name

    Returns:
        Dictionary with policy_class, config_class, and docstring
    """
    if name not in _POLICY_REGISTRY:
        raise ValueError(
            f"Unknown policy: '{name}'. Available: {list_available_policies()}"
        )

    policy_class, config_class = _POLICY_REGISTRY[name]
    return {
        "name": name,
        "policy_class": policy_class.__name__,
        "config_class": config_class.__name__,
        "description": policy_class.__doc__,
    }


def create_policy(
    name: str,
    config: VLAPolicyConfig | None = None,
    **kwargs: Any,
) -> BaseVLAPolicy:
    """
    Create a VLA policy instance by name.

    Args:
        name: Registered policy name (e.g., "groot-n1.6")
        config: Optional pre-configured config object
        **kwargs: Configuration parameters passed to config constructor

    Returns:
        Initialized BaseVLAPolicy instance

    Example:
        # Using kwargs
        policy = create_policy("groot-n1.6", server_port=5556)

        # Using config object
        config = GR00TN1Config(server_port=5556)
        policy = create_policy("groot-n1.6", config=config)
    """
    if name not in _POLICY_REGISTRY:
        raise ValueError(
            f"Unknown policy: '{name}'. Available: {list_available_policies()}"
        )

    policy_class, config_class = _POLICY_REGISTRY[name]

    if config is None:
        config = config_class(**kwargs)
    elif kwargs:
        raise ValueError("Cannot specify both config and kwargs")

    return policy_class(config)


def create_policy_from_dict(config_dict: dict[str, Any]) -> BaseVLAPolicy:
    """
    Create a policy from a configuration dictionary.

    The dictionary must contain a "model_name" key that matches a registered policy.

    Args:
        config_dict: Dictionary with model_name and configuration parameters

    Returns:
        Initialized BaseVLAPolicy instance
    """
    config_dict = config_dict.copy()
    model_name = config_dict.pop("model_name", None)

    if model_name is None:
        raise ValueError("config_dict must contain 'model_name' key")

    return create_policy(model_name, **config_dict)


# =============================================================================
# Register built-in policies
# =============================================================================

def _register_builtin_policies() -> None:
    """Register all built-in VLA policies."""
    # GR00T N1.6
    try:
        from .groot_n1 import GR00TN1Config, GR00TN1Policy
        register_policy("groot-n1.6", GR00TN1Policy, GR00TN1Config)
    except ImportError:
        pass  # GR00T not available


# Register on module import
_register_builtin_policies()
