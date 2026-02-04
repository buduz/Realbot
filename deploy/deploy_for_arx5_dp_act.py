import time
import numpy as np
import torch

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.robots import (Robot)
from lerobot.teleoperators import Teleoperator
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.control_utils import init_keyboard_listener, predict_action
from lerobot.utils.utils import (
    get_safe_torch_device,
    log_say,
    init_logging,
)
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

from lerobot.robots.arx5_follower import ARX5Follower
from lerobot.robots.arx5_follower import ARX5FollowerConfig
from lerobot.teleoperators.arx5_leader import ARX5Leader
from lerobot.teleoperators.arx5_leader import ARX5LeaderConfig

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.act.modeling_act import ACTPolicy

from deploy.utils.deploy_utils import process_actions
from deploy.utils.client import RobotInferenceClient

# end with 'joint' means joint_control, 'endpose' means cartesian_control
ACTION_TYPE = 'delta_endpose'
NUM_EPISODES = 100
EPISODE_TIME_SEC = 3600
HOST = "0.0.0.0"

PORT = 5555

# Number of seconds for resetting the environment after each episode.
RESET_TIME_SEC = 3600
# Pick up the cube. / Push the cube to the target position. / Stack the cube on top of the other cube.
# Pull the cube to the target position. / Pick up the ball and place it in the target position. / Pick up the peg and place it upright.
# TASK_DESCRIPTION = "Pick up the cube and place into plate"
# TASK_DESCRIPTION = "Stack the red cube on the blue cube"

# TASK_DESCRIPTION = "Pick up the peg and insert it into the container next to the peg" 
TASK_DESCRIPTION = "Stack the red cube on the blue cube"  
# TASK_DESCRIPTION = "Pick up the ball and place into the box" 
# TASK_DESCRIPTION = "Push the green cube into the pink area" 


# --------- Configuration for camera ---------
FPS = 30
NUM_IMAGE_WRITER_PROCESSES = 0
NUM_IMAGE_WRITER_THREADS_PER_CAMERA = 4

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

CTRL_FPS = 15
rollout_step = 8        # 和diffusion policy的n_action_steps保持一致

CAMERA_NAMES = ["image", "wrist_image"]
CAMERA_NAME_TO_SERIAL = {
    "image": "420222071008",
    "wrist_image": "348122070975",
}

init_logging()
_init_rerun(session_name="arx5_inference_session")

camera_configs = {}
for camera_name in CAMERA_NAMES:
    camera_configs[camera_name] = RealSenseCameraConfig(
            serial_number_or_name=CAMERA_NAME_TO_SERIAL[camera_name],
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=FPS,
        )

# Create the robot and teleoperator configurations
robot_config = ARX5FollowerConfig(
    control_mode='joint_control' if ACTION_TYPE.endswith('joint') else 'cartesian_control',
    cameras=camera_configs
)
robot = ARX5Follower(robot_config)
robot.connect(go_to_start = True)

listener, events = init_keyboard_listener()

policy = RobotInferenceClient(host=HOST, port=PORT)

@safe_stop_image_writer
def inference_loop(
    robot: Robot,
    events: dict,
    ctrl_fps: int,
    policy: PreTrainedPolicy | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
):
    policy.reset()
    timestamp = 0
    start_episode_t = time.perf_counter()
    cnt = rollout_step
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["stop_reinferring"]:
            events["stop_reinferring"] = True
            break

        observation = robot.get_observation()
        state = {k: v for k,v in observation.items() if k.endswith('.pos')}
        if ACTION_TYPE.endswith('joint'):
            state = np.array([state[k] for k in state if k.startswith('joint')] + [state['gripper.pos']])
        else:
            state = np.array([state[k] for k in state if not k.startswith('joint')])
            
        image = torch.from_numpy(observation['image'] / 255.0).permute(2,0,1).to('cuda').unsqueeze(0).to(torch.float32)
        wrist_image = torch.from_numpy(observation['wrist_image'] / 255.0).permute(2,0,1).to('cuda').unsqueeze(0).to(torch.float32)
        state = torch.from_numpy(state).to('cuda').unsqueeze(0).to(torch.float32)

        # mp_test
        state = torch.nn.functional.pad(state, (0, 6), mode='constant', value=0.0)

        element = {
            "observation.images.image": image,
            "observation.images.wrist_image": wrist_image,
            "observation.state": state,
        }
        
        if cnt == rollout_step:
            state = state.cpu().numpy()
        elif cnt == 0:
            cnt = rollout_step
            state = state.cpu().numpy()
        else:
            state = pred_action
        cnt -= 1

        if policy is not None:
            action_chunk = policy.get_action(element).cpu().numpy()
            print(f"#### Action: {action_chunk}")
        else:
            raise ValueError("No exist policy")
        
        state = state[:,:7]
        action_chunk = action_chunk[:,:7]
        pred_action = process_actions(state=state, action=action_chunk, action_type=ACTION_TYPE)

        if ACTION_TYPE.endswith('joint'):
            action = {
                'joint_1.pos': pred_action[0][0].item(), 
                'joint_2.pos': pred_action[0][1].item(), 
                'joint_3.pos': pred_action[0][2].item(), 
                'joint_4.pos': pred_action[0][3].item(), 
                'joint_5.pos': pred_action[0][4].item(), 
                'joint_6.pos': pred_action[0][5].item(), 
                'gripper.pos': pred_action[0][6].item()  
            }
        else:
            action = {
                'X_axis.pos': pred_action[0][0].item(), 
                'Y_axis.pos': pred_action[0][1].item(), 
                'Z_axis.pos': pred_action[0][2].item(), 
                'RX_axis.pos': pred_action[0][3].item(), 
                'RY_axis.pos': pred_action[0][4].item(), 
                'RZ_axis.pos': pred_action[0][5].item(), 
                'gripper.pos': pred_action[0][6].item()
            }

        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset.
        sent_action = robot.send_action(action)
        print(sent_action)

        if display_data:
            log_rerun_data(observation, {k: float(v) for k, v in action.items()})

        dt_s = time.perf_counter() - start_loop_t
        # print(1 / dt_s)
        busy_wait(1 / ctrl_fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t


recorded_episodes = 0
try:
    while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
        log_say("Reset the environment")
        robot.smooth_go_start(duration=2.0)
        time.sleep(2)
        # Logic for reset env
        if not events["stop_reinferring"] and (
            (recorded_episodes < NUM_EPISODES - 1) or events["rerecord_episode"]
        ):
            log_say(f"Infer episode {recorded_episodes}")
            inference_loop(
                robot=robot,
                events=events,
                ctrl_fps=CTRL_FPS,
                policy = policy,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
            )
            recorded_episodes += 1

except KeyboardInterrupt:
    log_say("\nCaptured Ctrl+C! Stopping the robot safely...")
    events["stop_recording"] = True
    events["stop_reinferring"] = True

except Exception as e:
    log_say(f"\nAn error occurred: {e}")

finally:
    log_say("Disconnecting robot...")
    robot.disconnect()
    listener.stop()
    log_say("Exited safely.")
