# VLA Model Architecture

The g1-vla project uses a modular VLA (Vision-Language-Action) interface that allows easy switching between different models.

## Quick Start

```python
from vla_models import create_policy, list_available_policies

# List available policies
print(list_available_policies())  # ['groot-n1.6']

# Create and use a policy
policy = create_policy("groot-n1.6", server_port=5556)
action, info = policy.get_action(observation)
```

## Architecture

```
Deployment Layer (client_node.py)
        │
        ▼
┌───────────────────────────────┐
│     BaseVLAPolicy (ABC)       │
│  - get_action(obs) -> action  │
│  - get_modality_config()      │
│  - reset()                    │
└───────────────┬───────────────┘
                │
    ┌───────────┴───────────┐
    ▼                       ▼
GR00TN1Policy         [Future Models]
(server-client)        OpenVLA, RT-2, etc.
```

## Adding a New Model

1. Create `vla_models/my_model.py`:

```python
from vla_models.base import BaseVLAPolicy, VLAPolicyConfig

@dataclass
class MyConfig(VLAPolicyConfig):
    model_name: str = "my-model"

class MyPolicy(BaseVLAPolicy):
    def get_action(self, observation):
        # Your inference logic
        return action, info

    def get_modality_config(self):
        return {"video": ..., "state": ..., "action": ...}

    def reset(self, options=None):
        return {}

    @property
    def action_horizon(self) -> int:
        return 16

    @property
    def is_connected(self) -> bool:
        return True
```

2. Register in `vla_models/factory.py`:

```python
from .my_model import MyConfig, MyPolicy
register_policy("my-model", MyPolicy, MyConfig)
```

3. Use it:

```python
policy = create_policy("my-model")
```

## Testing

```bash
# Start GR00T server (see docs/GROOT_SETUP.md)
cd Isaac-GR00T
.venv/bin/python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-G1-PnPAppleToPlate \
    --embodiment-tag UNITREE_G1 --port 5556

# Run integration test
cd g1-isaac-groot-n1
python scripts/test_vla_integration.py --port 5556
```

## Directory Structure

```
vla_models/
├── __init__.py    # Public API
├── base.py        # BaseVLAPolicy interface
├── factory.py     # Policy registry and factory
└── groot_n1.py    # GR00T N1.6 implementation
```
