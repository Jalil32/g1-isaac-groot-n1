# Setting Up NVIDIA Isaac GR00T N1.6

This guide documents how to set up the NVIDIA Isaac GR00T N1.6 framework for use with the Unitree G1 robot.

## Prerequisites

- **Python 3.10** (required - GR00T does not support other versions)
- **CUDA 12.4+** (recommended, 11.8 also works)
- **uv v0.8.4+** for dependency management
- **GPU**: H100, RTX 5090, RTX 4090, or Jetson AGX Thor recommended

## Installation

### 1. Clone the Isaac-GR00T Repository

```bash
cd /path/to/your/workspace
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T
```

If you already cloned without submodules:
```bash
git submodule update --init --recursive
```

### 2. Set Up the Environment

GR00T uses `uv` for dependency management. Install uv if you haven't:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the environment and install dependencies:
```bash
uv sync --python 3.10
uv pip install -e .
```

### 3. Verify Installation

```bash
cd Isaac-GR00T
.venv/bin/python -c "import gr00t; print('GR00T installed successfully')"
```

## Available Checkpoints

### Base Model
| Model | Description | HuggingFace Path |
|-------|-------------|------------------|
| GR00T N1.6 | Base 3B parameter model | `nvidia/GR00T-N1.6-3B` |

### Unitree G1 Checkpoint
| Model | Description | HuggingFace Path |
|-------|-------------|------------------|
| GR00T-N1.6-G1-PnPAppleToPlate | Fine-tuned for G1 pick-and-place | `nvidia/GR00T-N1.6-G1-PnPAppleToPlate` |

## Quick Test

### Test Model Loading

```bash
cd Isaac-GR00T
.venv/bin/python -c "
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag
print('Embodiment tags:', [t.name for t in EmbodimentTag])
"
```

### Start the Policy Server

```bash
cd Isaac-GR00T
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-G1-PnPAppleToPlate \
    --embodiment-tag UNITREE_G1 \
    --device cuda:0 \
    --port 5555
```

This will download the model from HuggingFace on first run (~6GB).

### Test Inference

Once the server is running, test inference from the g1-vla project:

```bash
cd g1-isaac-groot-n1
python scripts/test_groot_inference.py --port 5555
```

## Observation and Action Format

### Observation Structure

The policy expects observations as a nested dictionary:

```python
observation = {
    "video": {
        "ego_view": np.ndarray,  # Shape: (B, T, H, W, 3), dtype: uint8
    },
    "state": {
        "left_leg": np.ndarray,   # Shape: (B, T, 6), dtype: float32
        "right_leg": np.ndarray,  # Shape: (B, T, 6), dtype: float32
        "waist": np.ndarray,      # Shape: (B, T, 3), dtype: float32
        "left_arm": np.ndarray,   # Shape: (B, T, 7), dtype: float32
        "right_arm": np.ndarray,  # Shape: (B, T, 7), dtype: float32
        "left_hand": np.ndarray,  # Shape: (B, T, 7), dtype: float32
        "right_hand": np.ndarray, # Shape: (B, T, 7), dtype: float32
    },
    "language": {
        "annotation.human.task_description": [["task instruction"]]
    }
}
```

Where:
- `B` = batch size (typically 1)
- `T` = temporal horizon (typically 1)
- `H, W` = image height/width (e.g., 480, 640)

### Action Output

The model returns a 30-step action horizon with:

| Action Key | Shape | Description |
|------------|-------|-------------|
| `left_arm` | (B, 30, 7) | Left arm joint positions (relative) |
| `right_arm` | (B, 30, 7) | Right arm joint positions (relative) |
| `left_hand` | (B, 30, 7) | Left hand joint positions (absolute) |
| `right_hand` | (B, 30, 7) | Right hand joint positions (absolute) |
| `waist` | (B, 30, 3) | Waist joint positions (absolute) |
| `base_height_command` | (B, 30, 1) | Base height command |
| `navigate_command` | (B, 30, 3) | Navigation command (x, y, yaw) |

**Note**: Arm actions are **relative** (deltas from current position), while hand/waist actions are **absolute**.

## Embodiment Tags

GR00T supports multiple robot embodiments:

| Tag | Robot |
|-----|-------|
| `UNITREE_G1` | Unitree G1 humanoid |
| `GR1` | Fourier GR-1 |
| `LIBERO_PANDA` | Franka Panda (LIBERO) |
| `OXE_GOOGLE` | Google Robot |
| `OXE_WIDOWX` | WidowX |
| `BEHAVIOR_R1_PRO` | Galaxea R1 Pro |
| `NEW_EMBODIMENT` | Custom robots |

## Architecture Overview

GR00T N1.6 uses a vision-language-action (VLA) architecture:

1. **Vision Encoder**: Cosmos-Reason-2B VLM (handles flexible image resolutions)
2. **Language Encoder**: T5 transformer for text instructions
3. **Proprioception**: MLP for robot state encoding
4. **Action Decoder**: Flow-matching diffusion transformer (32 layers)

Key features in N1.6:
- Predicts **state-relative action chunks** (not absolute positions like N1.5)
- 2x larger DiT compared to N1.5
- Trained on G1 loco-manipulation data

## Server-Client Architecture

GR00T uses a server-client architecture for inference:

```
┌─────────────────┐     ZMQ      ┌──────────────────┐
│  Policy Server  │ ◄──────────► │  Robot Client    │
│  (GPU Machine)  │    REQ/REP   │  (Robot/Sim)     │
└─────────────────┘              └──────────────────┘
```

**Server**: Loads the model and handles inference
**Client**: Reads observations, sends to server, executes returned actions

## Troubleshooting

### Flash Attention Issues
For CUDA 11.8, install compatible flash-attn:
```bash
uv pip install flash-attn==2.8.2
```

### Import Errors
Ensure you're in the Isaac-GR00T directory and using the correct venv:
```bash
cd Isaac-GR00T
.venv/bin/python -c "import gr00t"
```

### Model Download Fails
Ensure you have enough disk space (~6GB per checkpoint) and a stable internet connection. Models are cached in `~/.cache/huggingface/`.

## References

- [Isaac-GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T)
- [GR00T N1.6 Research Blog](https://research.nvidia.com/labs/gear/gr00t-n1_6/)
- [HuggingFace Model](https://huggingface.co/nvidia/GR00T-N1.6-3B)
- [G1 Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1)
