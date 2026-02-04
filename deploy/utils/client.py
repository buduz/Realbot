import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import random
from typing import Any, Callable, Dict

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from deploy.base import BaseInferenceClient


class RobotInferenceClient(BaseInferenceClient):
    """
    Client for communicating with the RealRobotServer
    """

    def get_action(self, observations: dict) -> dict:
        """
        Get the action from the server.
        The exact definition of the observations is defined
        by the policy, which contains the modalities configuration.
        """
        return self.call_endpoint("get_action", observations)
    
    def set_robot_uid(self, tags: str):
        return self.call_endpoint('set_robot_uid', tags)
    
    def reset(self) -> Dict[str, Any]:
        return self.call_endpoint("reset", requires_input=False)
    
    def select_embodiment(self, tags: str):
        return self.call_endpoint('select_embodiment', tags)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        type=str,
        help="Host address for the server.",
        default="100.84.199.92"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port number for the server.",
        default=5558
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to run the model on.",
        default="cuda"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for the model.",
        default=7
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset.",
        default="/home/pc/.cache/huggingface/lerobot/ARX5/arx5-red-cube-new-joint"
    )

    # client mode
    args = parser.parse_args()

    device = args.device
    np.random.seed(args.seed)
    policy_client = RobotInferenceClient(host=args.host, port=args.port)

    # delta_timestamps = {
    #     # Load the previous image and state at -0.1 seconds before current frame,
    #     # then load current image and state corresponding to 0.0 second.
    #     "observation.images.image": [-0.1, 0.0],
    #     "observation.images.wristimage": [-0.1, 0.0],
    #     "observation.state": [-0.1, 0.0],
    #     # Load the previous action (-0.1), the next action to be executed (0.0),
    #     # and 14 future actions with a 0.1 seconds spacing. All these actions will be
    #     # used to supervise the policy.
    #     "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    # }
    delta_timestamps = {
        'action': [-0.06666666666666667, 0.0, 0.06666666666666667, 0.13333333333333333, 0.2, 0.26666666666666666, 0.3333333333333333, 0.4, 0.4666666666666667, 0.5333333333333333, 0.6, 0.6666666666666666, 0.7333333333333333, 0.8, 0.8666666666666667, 0.9333333333333333],
        'observation.state': [-0.06666666666666667, 0.0],
        'observation.images.image': [-0.06666666666666667, 0.0],
        'observation.images.wrist_image': [-0.06666666666666667, 0.0]
     }

    # We can then instantiate the dataset with these delta_timestamps configuration.
    dataset = LeRobotDataset(args.dataset_path, delta_timestamps=delta_timestamps)
    item_idx = random.randint(0, len(dataset) - 1)
    data = dataset[item_idx]

    state = data["observation.state"][1]       # tensor, /255
    image = data['observation.images.image'][1] 
    wrist_image = data['observation.images.wrist_image'][1] 
    state = state.to(device, non_blocking=True)
    image = image.to(device, non_blocking=True)
    wrist_image = wrist_image.to(device, non_blocking=True)
    state = state.unsqueeze(0)
    image = image.unsqueeze(0)
    wrist_image = wrist_image.unsqueeze(0)

    element = {
            "observation.images.image": image.detach().cpu().numpy(),
            "observation.images.wrist_image": wrist_image.detach().cpu().numpy(),
            "observation.state": state.detach().cpu().numpy(),
    }
    element = {k: (torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v) for k, v in element.items()}
    action_chunk = policy_client.get_action(element)
    # plot_prediction_vs_groundtruth(data['action.actions'], action_chunk, save_path="/home/zhiheng/project/Isaac-GR00T/compare.png")
    print(action_chunk)
