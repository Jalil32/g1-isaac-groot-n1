import cv2
import zmq
import numpy as np
import time
from typing import Dict, Any
import numpy as np
from typing import Union
import time
from gr00t.eval.service import ExternalRobotInferenceClient
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as hg_LowState
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_

from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandState_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import  HandCmd_ as HandCmd
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC
from robotconfig import RobotConfig

# Initialize DDS communication
print("Initialising channel factory")
ChannelFactoryInitialize(1)  # 0 for real robot, 1 for simulation

# Create subscribers for robot state
print("Initialising subscribers for robot state")

# mode_pr_ = MotorMode.PR
# mode_machine_ = 0


# lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
# lowstate_subscriber.Init(LowStateHgHandler, 10)

# lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)

# Topics
kTopicLowState = "rt/lowstate"
kTopicLowCmd = "rt/lowcmd"
# config_path = "./configs/g1.yaml"
config_path = "deployment/simulate_python/configs/g1.yaml"
# Topics
kTopicDex3LeftCommand = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"


class HeadCameraClient:
    def __init__(self, server_address="tcp://localhost:5555", timeout=1000):
        """
        Initialize ZMQ client to receive head camera images
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(server_address)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.RCVTIMEO, timeout)  # timeout in milliseconds

    def get_latest_frame(self):
        """
        Get the latest frame from head camera
        Returns: numpy array of shape (480, 640, 3) in RGB format, or None if failed
        """
        try:
            # Try to get the most recent frame (non-blocking approach)
            # We'll read multiple messages to get the latest one
            latest_message = None
            for _ in range(5):  # Try to get up to 5 messages to get the latest
                try:
                    message = self.socket.recv(zmq.NOBLOCK)
                    latest_message = message
                except zmq.Again:
                    break

            if latest_message is None:
                # If no non-blocking message, try one blocking call with timeout
                latest_message = self.socket.recv()

            # Decode the image
            nparr = np.frombuffer(latest_message, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img_bgr is not None:
                # Convert BGR to RGB and ensure it's 640x480
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                if img_rgb.shape[:2] != (480, 640):
                    img_rgb = cv2.resize(img_rgb, (640, 480))
                return img_rgb
            else:
                return None

        except zmq.Again:
            # Timeout occurred
            return None
        except Exception as e:
            print(f"Error receiving camera frame: {e}")
            return None

    def close(self):
        """Clean up ZMQ connection"""
        self.socket.close()
        self.context.term()

def move_to_default_pos(low_cmd, lowcmd_publisher, low_state, config):
    print("Moving to default pos.")
    # move time 2s
    total_time = 2
    num_step = int(total_time / config.control_dt)

    dof_idx = config.leg_joint2motor_idx + config.arm_waist_joint2motor_idx
    # dof_idx = config.arm_waist_joint2motor_idx
    kps = config.kps + config.arm_waist_kps
    kds = config.kds + config.arm_waist_kds
    default_pos = np.concatenate((config.default_angles, config.arm_waist_target), axis=0)
    dof_size = len(dof_idx)

    # record the current pos
    init_dof_pos = np.zeros(dof_size, dtype=np.float32)
    for i in range(dof_size):
        init_dof_pos[i] = low_state.motor_state[dof_idx[i]].q

    # move to default pos
    for i in range(num_step):
        alpha = i / num_step
        for j in range(dof_size):
            motor_idx = dof_idx[j]
            target_pos = default_pos[j]
            low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
            low_cmd.motor_cmd[motor_idx].qd = 0
            low_cmd.motor_cmd[motor_idx].kp = kps[j]
            low_cmd.motor_cmd[motor_idx].kd = kds[j]
            low_cmd.motor_cmd[motor_idx].tau = 0
        send_cmd(low_cmd, lowcmd_publisher)
        time.sleep(config.control_dt)


def default_pos_state(low_cmd, lowcmd_publisher, config):
    while True:
        for i in range(len(config.leg_joint2motor_idx)):
            motor_idx = config.leg_joint2motor_idx[i]
            low_cmd.motor_cmd[motor_idx].q = config.default_angles[i]
            low_cmd.motor_cmd[motor_idx].qd = 0
            low_cmd.motor_cmd[motor_idx].kp = config.kps[i]
            low_cmd.motor_cmd[motor_idx].kd = config.kds[i]
            low_cmd.motor_cmd[motor_idx].tau = 0
        for i in range(len(config.arm_waist_joint2motor_idx)):
            motor_idx = config.arm_waist_joint2motor_idx[i]
            low_cmd.motor_cmd[motor_idx].q = config.arm_waist_target[i]
            low_cmd.motor_cmd[motor_idx].qd = 0
            low_cmd.motor_cmd[motor_idx].kp = config.arm_waist_kps[i]
            low_cmd.motor_cmd[motor_idx].kd = config.arm_waist_kds[i]
            low_cmd.motor_cmd[motor_idx].tau = 0
        send_cmd(low_cmd, lowcmd_publisher)
        time.sleep(config.control_dt)

def send_cmd(cmd: Union[LowCmdGo, LowCmdHG], lowcmd_publisher):
    cmd.crc = CRC().Crc(cmd)
    lowcmd_publisher.Write(cmd)

def create_zero_cmd(cmd: Union[LowCmdGo, LowCmdHG, HandCmd]):
    size = len(cmd.motor_cmd)
    for i in range(size):
        cmd.motor_cmd[i].q = 0
        cmd.motor_cmd[i].qd = 0
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 0
        cmd.motor_cmd[i].tau = 0


def main():
    config = RobotConfig(config_path)

    low_cmd = unitree_hg_msg_dds__LowCmd_()
    low_state = unitree_hg_msg_dds__LowState_()

    left_hand_state = unitree_hg_msg_dds__HandState_()
    right_hand_state = unitree_hg_msg_dds__HandState_()
    left_hand_cmd = unitree_hg_msg_dds__HandCmd_()
    right_hand_cmd = unitree_hg_msg_dds__HandCmd_()

    lowcmd_publisher = ChannelPublisher(kTopicLowCmd, LowCmdHG)
    lowcmd_publisher.Init()

    right_hand_publisher = ChannelPublisher(kTopicDex3RightCommand, HandCmd)
    right_hand_publisher.Init()

    left_hand_publisher = ChannelPublisher(kTopicDex3LeftCommand, HandCmd)
    left_hand_publisher.Init()

    lowstate_subscriber = ChannelSubscriber(kTopicLowState, LowStateHG)
    lowstate_subscriber.Init()

    # Create subscribers for hand state
    lefthand_subscriber = ChannelSubscriber(kTopicDex3LeftState, HandState_)
    lefthand_subscriber.Init()

    righthand_subscriber = ChannelSubscriber(kTopicDex3RightState, HandState_)
    righthand_subscriber.Init()
    # Initialize head camera client
    head_camera = HeadCameraClient(server_address="tcp://localhost:5555")

    print("")
    create_zero_cmd(low_cmd)
    create_zero_cmd(left_hand_cmd)
    create_zero_cmd(right_hand_cmd)

    # Move to default position
    move_to_default_pos(low_cmd, lowcmd_publisher, low_state, config)

    # print("Holding default position")
    # default_pos_state(low_cmd, lowcmd_publisher, config)


    print("Reading robot state")
    iteration_count = 0
    while True:

        # === CAPTURE CAMERA IMAGE ===
        frame_rgb = head_camera.get_latest_frame()
        if frame_rgb is not None:
            # Add batch dimension for observation format
            camera_image = np.expand_dims(frame_rgb, axis=0)
            # Save image every 10 iterations (adjust as needed)
            iteration_count += 1

            if iteration_count % 10 == 0:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'./images/policy_input_{iteration_count}.jpg', frame_bgr)
                print(f"Saved policy input image: policy_input_{iteration_count}.jpg")

            # print("Successfully captured head camera image!")
            # print(f"Head camera image shape: {camera_image.shape}")
        else:
            # Fallback to random image
            camera_image = np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8)
            print("Head camera capture failed, using random image")

        # === READ ARM STATE ===
        robot_data = lowstate_subscriber.Read()
        if robot_data is not None:
            left_arm_q = np.array([robot_data.motor_state[i].q for i in range(15, 22)])   # 7 joints
            right_arm_q = np.array([robot_data.motor_state[i].q for i in range(22, 29)])  # 7 joints

            # Add batch dimension for observation format
            left_arm_state = np.expand_dims(left_arm_q, axis=0)
            right_arm_state = np.expand_dims(right_arm_q, axis=0)

            # print("Left arm joint angles:", left_arm_q)
            # print("Right arm joint angles:", right_arm_q)
        else:
            # Fallback to random values
            left_arm_state = np.random.rand(1, 7)
            right_arm_state = np.random.rand(1, 7)
            print("Arm state read failed, using random values")
        import time
        time.sleep(1)

        # === READ HAND STATE ===
        left_hand_data = lefthand_subscriber.Read()
        right_hand_data = righthand_subscriber.Read()

        if left_hand_data is not None and right_hand_data is not None:
            left_hand_q = np.array([left_hand_data.motor_state[i].q for i in range(0, 7)])   # 7 joints
            right_hand_q = np.array([right_hand_data.motor_state[i].q for i in range(0, 7)])  # 7 joints

            # Add batch dimension for observation format
            left_hand_state = np.expand_dims(left_hand_q, axis=0)
            right_hand_state = np.expand_dims(right_hand_q, axis=0)

            # print("Left hand joint angles:", left_hand_q)
            # print("Right hand joint angles:", right_hand_q)
        else:
            # Fallback to random values
            left_hand_state = np.random.rand(1, 7)
            right_hand_state = np.random.rand(1, 7)
            print("Hand state read failed, using random values")

        # # Print observation shapes for verification
        # print("\n=== OBSERVATION SUMMARY ===")
        # print(f"Head camera image shape: {obs['video.cam_right_high'].shape}")
        # print(f"Left arm state shape: {obs['state.left_arm'].shape}")
        # print(f"Right arm state shape: {obs['state.right_arm'].shape}")
        # print(f"Left hand state shape: {obs['state.left_hand'].shape}")
        # print(f"Right hand state shape: {obs['state.right_hand'].shape}")
        # print(f"Task description: {obs['annotation.human.task_description']}")

        # === CREATE OBSERVATION DICTIONARY ===
        raw_obs_dict = {
            # Video modalities - real camera image
            "video.cam_right_high": camera_image,
            # State modalities - real robot state
            "state.left_arm": left_arm_state,
            "state.right_arm": right_arm_state,
            "state.left_hand": left_hand_state,
            "state.right_hand": right_hand_state,
            # Language modality
            "annotation.human.task_description": ["Move the red cube"],
        }

        # Get action chunk from inference server
        policy = ExternalRobotInferenceClient(host="localhost", port=5000)

        raw_action_chunk: Dict[str, Any] = policy.get_action(raw_obs_dict)

        # print("\n=== PREDICTION SUMMARY ===")
        # print(raw_action_chunk)
        # print(raw_action_chunk['action.right_hand'][0])
        # print(raw_action_chunk['action.right_hand'][0].shape)

        # 1) Get actions
        # Left Arm
        left_arm = raw_action_chunk['action.left_arm']
        # Right Arm
        right_arm = raw_action_chunk['action.right_arm']
        # Left Hand
        left_hand = raw_action_chunk['action.left_hand']
        # Right Hand
        right_hand = raw_action_chunk['action.right_hand']

        print("Left Hand Pred:")
        print(left_hand[1])
        print("Right Hand Pred:")
        print(right_hand[1])
        # 2) Build low cmd
        # Build left arm

        for j in range(len(left_hand)):
        # for j in range(1):

            for i in range(len(config.left_arm)):
                motor_idx = config.left_arm[i]
                low_cmd.motor_cmd[motor_idx].q = left_arm[j][i]
                low_cmd.motor_cmd[motor_idx].qd = 0
                low_cmd.motor_cmd[motor_idx].kp = config.kps[i]* 0.3
                low_cmd.motor_cmd[motor_idx].kd = config.kds[i]* 1.2
                low_cmd.motor_cmd[motor_idx].tau = 0

            # Build right arm
            for i in range(len(config.right_arm)):
                motor_idx = config.right_arm[i]
                low_cmd.motor_cmd[motor_idx].q = right_arm[j][i]
                low_cmd.motor_cmd[motor_idx].qd = 0
                low_cmd.motor_cmd[motor_idx].kp = config.kps[i] * 0.3
                low_cmd.motor_cmd[motor_idx].kd = config.kds[i] * 1.2
                low_cmd.motor_cmd[motor_idx].tau = 0

            # Build left hand
            for i in range(len(config.left_hand)):
                motor_idx = config.left_hand[i]
                left_hand_cmd.motor_cmd[motor_idx].q = left_hand[j][i]
                left_hand_cmd.motor_cmd[motor_idx].qd = 0
                left_hand_cmd.motor_cmd[motor_idx].kp = 1.5
                left_hand_cmd.motor_cmd[motor_idx].kd = 0.2
                left_hand_cmd.motor_cmd[motor_idx].tau = 0

            # Build right hand
            for i in range(len(config.right_hand)):
                motor_idx = config.right_hand[i]
                right_hand_cmd.motor_cmd[motor_idx].q = right_hand[j][i]
                right_hand_cmd.motor_cmd[motor_idx].qd = 0
                right_hand_cmd.motor_cmd[motor_idx].kp = 1.5
                right_hand_cmd.motor_cmd[motor_idx].kd = 0.2
                right_hand_cmd.motor_cmd[motor_idx].tau = 0

            # 2) Send low command
            send_cmd(low_cmd, lowcmd_publisher)
            # send_cmd(left_hand_cmd, left_hand_publisher)
            # send_cmd(right_hand_cmd, right_hand_publisher)
            left_hand_publisher.Write(left_hand_cmd)
            right_hand_publisher.Write(right_hand_cmd)

            time.sleep(0.01)
        time.sleep(1)





if __name__ == "__main__":
    main()
