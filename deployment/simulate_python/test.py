import mujoco
import numpy as np

import config

# Load the model
mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)

# Create visualization (optional)
import mujoco.viewer
viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
