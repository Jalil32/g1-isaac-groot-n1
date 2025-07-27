import mujoco
import numpy as np
import config

class G1MuJoCoReader:
    def __init__(self, model_path):
        # Load the model
        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        # Create visualization (optional)
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

        # Define joint mappings based on your real robot code
        # Adjust these indices based on your actual MuJoCo model
        self.left_arm_joint_indices = list(range(15, 22))   # 7 joints (indices 15-21)
        self.right_arm_joint_indices = list(range(22, 29))  # 7 joints (indices 22-28)

        # Hand joint indices - you'll need to find these in your model
        self.left_hand_joint_indices = list(range(29, 36))   # 7 joints (adjust as needed)
        self.right_hand_joint_indices = list(range(36, 43))  # 7 joints (adjust as needed)

        # Print model info to help identify correct joints
        self.inspect_model()

    def inspect_model(self):
        """Inspect the model to find available joints and bodies"""
        print("=== MUJOCO MODEL INSPECTION ===")
        print(f"Total joints: {self.mj_model.njnt}")
        print(f"Total bodies: {self.mj_model.nbody}")

        print("\nJoint names and indices:")
        for i in range(min(50, self.mj_model.njnt)):  # Show first 50 joints
            joint_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                print(f"  {i:2d}: {joint_name}")

        print("\nBody names (first 30):")
        for i in range(min(30, self.mj_model.nbody)):
            body_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name:
                print(f"  {i:2d}: {body_name}")

    def find_joints_by_name(self, joint_patterns):
        """Find joint indices by name patterns"""
        joint_indices = []
        for pattern in joint_patterns:
            for i in range(self.mj_model.njnt):
                joint_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if joint_name and pattern.lower() in joint_name.lower():
                    joint_indices.append(i)
                    print(f"Found joint: {joint_name} at index {i}")
        return joint_indices

    def update_joint_mappings(self):
        """Update joint mappings based on actual model joint names"""
        # Common G1 arm joint patterns - adjust based on your model
        left_arm_patterns = [
            "left_shoulder", "left_elbow", "left_wrist",
            "l_shoulder", "l_elbow", "l_wrist",
            "left_arm"
        ]
        right_arm_patterns = [
            "right_shoulder", "right_elbow", "right_wrist",
            "r_shoulder", "r_elbow", "r_wrist",
            "right_arm"
        ]
        left_hand_patterns = [
            "left_hand", "left_finger", "l_hand", "l_finger",
            "left_thumb", "left_index", "left_middle"
        ]
        right_hand_patterns = [
            "right_hand", "right_finger", "r_hand", "r_finger",
            "right_thumb", "right_index", "right_middle"
        ]

        print("\n=== SEARCHING FOR ARM/HAND JOINTS ===")
        self.left_arm_joint_indices = self.find_joints_by_name(left_arm_patterns)
        self.right_arm_joint_indices = self.find_joints_by_name(right_arm_patterns)
        self.left_hand_joint_indices = self.find_joints_by_name(left_hand_patterns)
        self.right_hand_joint_indices = self.find_joints_by_name(right_hand_patterns)

    def read_robot_state(self):
        """Read current robot state from MuJoCo simulation"""
        # Step simulation to ensure data is current
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Read arm joint positions
        try:
            left_arm_q = np.array([self.mj_data.qpos[i] for i in self.left_arm_joint_indices])
            right_arm_q = np.array([self.mj_data.qpos[i] for i in self.right_arm_joint_indices])

            # Ensure we have 7 joints for each arm (pad or truncate if needed)
            left_arm_q = self._ensure_length(left_arm_q, 7, "left arm")
            right_arm_q = self._ensure_length(right_arm_q, 7, "right arm")

        except (IndexError, ValueError) as e:
            print(f"Error reading arm joints: {e}")
            left_arm_q = np.zeros(7)
            right_arm_q = np.zeros(7)

        # Read hand joint positions
        try:
            left_hand_q = np.array([self.mj_data.qpos[i] for i in self.left_hand_joint_indices])
            right_hand_q = np.array([self.mj_data.qpos[i] for i in self.right_hand_joint_indices])

            # Ensure we have 7 joints for each hand (pad or truncate if needed)
            left_hand_q = self._ensure_length(left_hand_q, 7, "left hand")
            right_hand_q = self._ensure_length(right_hand_q, 7, "right hand")

        except (IndexError, ValueError) as e:
            print(f"Error reading hand joints: {e}")
            left_hand_q = np.zeros(7)
            right_hand_q = np.zeros(7)

        return left_arm_q, right_arm_q, left_hand_q, right_hand_q

    def _ensure_length(self, array, target_length, name):
        """Ensure array has target length by padding or truncating"""
        if len(array) == target_length:
            return array
        elif len(array) < target_length:
            print(f"Warning: {name} has {len(array)} joints, padding to {target_length}")
            return np.pad(array, (0, target_length - len(array)), 'constant')
        else:
            print(f"Warning: {name} has {len(array)} joints, truncating to {target_length}")
            return array[:target_length]

    def get_observation_dict(self):
        """Get observation dictionary in the same format as real robot"""
        # Read robot state
        left_arm_q, right_arm_q, left_hand_q, right_hand_q = self.read_robot_state()

        # Create fake camera image (replace with actual camera if available)
        camera_image = np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8)

        # Add batch dimension for observation format
        left_arm_state = np.expand_dims(left_arm_q, axis=0)
        right_arm_state = np.expand_dims(right_arm_q, axis=0)
        left_hand_state = np.expand_dims(left_hand_q, axis=0)
        right_hand_state = np.expand_dims(right_hand_q, axis=0)

        # Create observation dictionary matching real robot format
        obs = {
            # Video modalities
            "video.cam_right_high": camera_image,
            # State modalities
            "state.left_arm": left_arm_state,
            "state.right_arm": right_arm_state,
            "state.left_hand": left_hand_state,
            "state.right_hand": right_hand_state,
            # Language modality
            "annotation.human.task_description": ["pick up the object and place it on the table"],
        }

        return obs

    def print_current_state(self):
        """Print current robot state for debugging"""
        left_arm_q, right_arm_q, left_hand_q, right_hand_q = self.read_robot_state()

        print("\n=== CURRENT ROBOT STATE ===")
        print(f"Left arm joint angles: {left_arm_q}")
        print(f"Right arm joint angles: {right_arm_q}")
        print(f"Left hand joint angles: {left_hand_q}")
        print(f"Right hand joint angles: {right_hand_q}")

    def get_end_effector_positions(self):
        """Get hand/end-effector positions in world coordinates"""
        # Common hand body names - adjust based on your model
        hand_body_names = ["left_hand", "right_hand", "l_hand", "r_hand"]

        hand_positions = {}

        for body_name in hand_body_names:
            body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                pos = self.mj_data.xpos[body_id].copy()
                quat = self.mj_data.xquat[body_id].copy()
                hand_positions[body_name] = {
                    "position": pos,
                    "quaternion": quat
                }
                print(f"{body_name} position: {pos}")

        return hand_positions

    def run_simulation_loop(self, steps=1000):
        """Run simulation loop and read states"""
        for step in range(steps):
            # Step the simulation
            mujoco.mj_step(self.mj_model, self.mj_data)

            # Print state every 100 steps
            if step % 100 == 0:
                print(f"\n=== STEP {step} ===")
                self.print_current_state()

                # Get observation dict
                obs = self.get_observation_dict()
                print(f"Observation shapes:")
                for key, value in obs.items():
                    if isinstance(value, np.ndarray):
                        print(f"  {key}: {value.shape}")
                    else:
                        print(f"  {key}: {type(value)}")

            # Update viewer
            if self.viewer.is_running():
                self.viewer.sync()
            else:
                break

# Usage example
if __name__ == "__main__":
    # Initialize the reader
    g1_reader = G1MuJoCoReader(config.ROBOT_SCENE)

    # Update joint mappings based on actual model
    g1_reader.update_joint_mappings()

    # Print current state
    g1_reader.print_current_state()

    # Get end effector positions
    g1_reader.get_end_effector_positions()

    # Get observation dictionary
    obs = g1_reader.get_observation_dict()
    print("\n=== OBSERVATION SUMMARY ===")
    print(f"Camera image shape: {obs['video.cam_right_high'].shape}")
    print(f"Left arm state shape: {obs['state.left_arm'].shape}")
    print(f"Right arm state shape: {obs['state.right_arm'].shape}")
    print(f"Left hand state shape: {obs['state.left_hand'].shape}")
    print(f"Right hand state shape: {obs['state.right_hand'].shape}")

    # Run simulation loop
    # g1_reader.run_simulation_loop(500)
