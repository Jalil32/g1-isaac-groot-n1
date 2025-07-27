# Deploying Isaac Gr00T N1 to Unitree G1 Humanoid Robot

## Contact
For questions or collaboration opportunities, please contact:
📧 22751096@student.uwa.edu.au (Jalil)

## Overview
This repository contains the implementation for a Master of Software Engineering research project focused on robotic manipulation using the Unitree G1 robot. The project integrates Isaac Sim simulation with GR00T models for pick-and-place tasks.

**Current Status**: Work in progress. The model successfully controls arm and hand movements but requires further tuning for reliable block pickup functionality.

## Project Structure
The repository includes experimental files and testing code as development progresses. The main focus is on implementing a pick-and-place task for red blocks using dexterous manipulation.

## Environment Setup

### Prerequisites
Two conda environments are required:

1. **Simulation Environment** (from unitree_sim_isaaclab)
   ```bash
   conda activate unitree_sim_env
   ```

2. **Inference Environment** (from Isaac-GR00T)
   ```bash
   conda activate gr00t
   ```

## Usage

### Running the Simulation
Ensure you're using the `unitree_sim_env` conda environment:

```bash
python sim_main.py --device cpu --enable_cameras --task Isaac-PickPlace-RedBlock-G129-Dex3-Joint --action_source dds --enable_dex3_dds --robot_type g129
```

### Running the Inference System
Switch to the `gr00t` conda environment for the following commands:

#### 1. Start the Inference Service
The inference service loads the model and processes joint/hand position data to generate predictions:

```bash
python deployments/inference_service.py --model_path models --server
```

#### 2. Run the Client
The client reads joint positions from the robot/simulation, sends them to the inference service, receives predictions, and executes the next set of positions:

```bash
python deployment/simulate_python/run.py
```

#### 3. Monitor Camera Feed (Optional)
To view the head camera feed during execution:

```bash
python head_camera_view.py
```

## Known Issues
- The model currently moves arms and hands correctly but fails to reliably pick up blocks
- Further model tuning and calibration required

## Contact
For questions or collaboration opportunities, please contact:
📧 22751096@student.uwa.edu.au

## Acknowledgments
This project builds upon several open-source repositories:

- **[Unitree Sim IsaacLab](https://github.com/unitreerobotics/unitree_sim_isaaclab)** - Simulation framework
- **[NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)** - Base model (Version 1.0, fine-tuned for Unitree G1)
- **[Unitree SDK2 Python](https://github.com/unitreerobotics/unitree_sdk2_python)** - Robot communication interface
- **[Unitree XR Teleoperate](https://github.com/unitreerobotics/xr_teleoperate)** - Robot teleoperation interface

Please refer to the respective repositories for licensing information.

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses
This project incorporates code from several open-source repositories. Please see the [NOTICES](NOTICES) file for complete license information and attributions for all third-party components.
