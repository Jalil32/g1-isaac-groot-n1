import cv2
import numpy as np
import pyrealsense2 as rs
from typing import Dict, Any

# for policy
from gr00t.eval.service import ExternalRobotInferenceClient

# for dex3-1 hands
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_

# for arms
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as hg_LowState

# Initialize DDS communication
ChannelFactoryInitialize(0)  # 0 for real robot, 1 for simulation

# Topics
kTopicDex3LeftCommand = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"
kTopicLowState = "rt/lowstate"

# Create subscribers for robot state
lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
lowstate_subscriber.Init()

# Create subscribers for hand state
lefthand_subscriber = ChannelSubscriber(kTopicDex3LeftState, HandState_)
lefthand_subscriber.Init()
righthand_subscriber = ChannelSubscriber(kTopicDex3RightState, HandState_)
righthand_subscriber.Init()

# Initialize RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    # === CAPTURE CAMERA IMAGE ===
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if color_frame:
        # Convert to numpy array and RGB format
        frame = np.asanyarray(color_frame.get_data())
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        camera_image = np.expand_dims(frame_rgb, axis=0)  # Add batch dimension
        print("Successfully captured camera image!")
    else:
        # Fallback to random image
        camera_image = np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8)
        print("Camera capture failed, using random image")

    # === READ ARM STATE ===
    robot_data = lowstate_subscriber.Read()
    if robot_data is not None:
        left_arm_q = np.array([robot_data.motor_state[i].q for i in range(15, 22)])   # 7 joints
        right_arm_q = np.array([robot_data.motor_state[i].q for i in range(22, 29)])  # 7 joints

        # Add batch dimension for observation format
        left_arm_state = np.expand_dims(left_arm_q, axis=0)
        right_arm_state = np.expand_dims(right_arm_q, axis=0)

        print("Left arm joint angles:", left_arm_q)
        print("Right arm joint angles:", right_arm_q)
    else:
        # Fallback to random values
        left_arm_state = np.random.rand(1, 7)
        right_arm_state = np.random.rand(1, 7)
        print("Arm state read failed, using random values")

    # === READ HAND STATE ===
    left_hand_data = lefthand_subscriber.Read()
    right_hand_data = righthand_subscriber.Read()

    if left_hand_data is not None and right_hand_data is not None:
        left_hand_q = np.array([left_hand_data.motor_state[i].q for i in range(0, 7)])   # 7 joints
        right_hand_q = np.array([right_hand_data.motor_state[i].q for i in range(0, 7)])  # 7 joints

        # Add batch dimension for observation format
        left_hand_state = np.expand_dims(left_hand_q, axis=0)
        right_hand_state = np.expand_dims(right_hand_q, axis=0)

        print("Left hand joint angles:", left_hand_q)
        print("Right hand joint angles:", right_hand_q)
    else:
        # Fallback to random values
        left_hand_state = np.random.rand(1, 7)
        right_hand_state = np.random.rand(1, 7)
        print("Hand state read failed, using random values")

    # === CREATE OBSERVATION DICTIONARY ===
    obs = {
        # Video modalities - real camera image
        "video.cam_right_high": camera_image,
        # State modalities - real robot state
        "state.left_arm": left_arm_state,
        "state.right_arm": right_arm_state,
        "state.left_hand": left_hand_state,
        "state.right_hand": right_hand_state,
        # Language modality
        "annotation.human.task_description": ["pick up the object and place it on the table"],
    }

    # Print observation shapes for verification
    print("\n=== OBSERVATION SUMMARY ===")
    print(f"Camera image shape: {obs['video.cam_right_high'].shape}")
    print(f"Left arm state shape: {obs['state.left_arm'].shape}")
    print(f"Right arm state shape: {obs['state.right_arm'].shape}")
    print(f"Left hand state shape: {obs['state.left_hand'].shape}")
    print(f"Right hand state shape: {obs['state.right_hand'].shape}")
    print(f"Task description: {obs['annotation.human.task_description']}")

    # Get action chunk from inference server
    policy = ExternalRobotInferenceClient(host="localhost", port=5555)
    raw_action_chunk: Dict[str, Any] = policy.get_action(raw_obs_dict)
    print(raw_action_chunk)


finally:
    # Clean up camera
    pipeline.stop()

# NOTE: Next step will need to test this on the robot
# - Will probably need to ssh in and transfer all this stuff to it
#   and set up the environment
# - Need to add emergency code from the sim2real repo
# - Need to read up a bit more on setting constraints for the joints and joint velocity



# 1) Initialise robot
# - use sim2real to look at how they do this
# - download python sdk

# 2) Get arm join and dedx3-1 information
# -

# 3) Put into correct format

# 4) Send to policy and get raw action chunk

# 5) Execute action chunk
