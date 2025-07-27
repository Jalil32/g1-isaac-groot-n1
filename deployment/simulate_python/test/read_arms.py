import cv2
import numpy as np
import pyrealsense2 as rs
from typing import Dict, Any


# for policy
# from gr00t.eval.service import ExternalRobotInferenceClient

# for dex3-1 hands
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

# for arms
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as hg_LowState

# Initialize DDS communication
print("Initialising channel factory")
ChannelFactoryInitialize(1, "lo")  # 0 for real robot, 1 for simulation

# Topics
kTopicLowState = "rt/lowstate"

# Create subscribers for robot state
print("Initialising subscribers for robot state")
# lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
lowstate_subscriber = ChannelSubscriber(kTopicLowState, LowState_)
lowstate_subscriber.Init()

try:
    # === READ ARM STATE ===
    print("Reading robot state")
    while True:
        robot_data = lowstate_subscriber.Read()
        if robot_data is not None:
            left_arm_q = np.array([robot_data.motor_state[i].q for i in range(15, 22)])   # 7 joints
            right_arm_q = np.array([robot_data.motor_state[i].q for i in range(22, 29)])  # 7 joints

            # Add batch dimension for observation format
            left_arm_state = np.expand_dims(left_arm_q, axis=0)
            right_arm_state = np.expand_dims(right_arm_q, axis=0)

            print("Left arm joint angles:", left_arm_q)
            print("Right arm joint angles:", right_arm_q)
        import time
        time.sleep(1)

finally:
    print("done")
