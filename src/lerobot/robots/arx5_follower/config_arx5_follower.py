#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("ARX5_follower")
@dataclass
class ARX5FollowerConfig(RobotConfig):
    """Configuration for single ARX5 arm follower robot."""
    arm_model: str = "X5" 
    arm_port: str = "can0"
    log_level: str = "INFO"
    use_multithreading: bool = True

    # Control parameters
    controller_dt: float = 0.005  # 200Hz control loop
    interpolation_controller_dt: float = 0.01  # 50Hz high-level interpolation control frequency

    # default to joint control mode
    control_mode: str = "joint_control"  # "joint_control" or "cartesian_control"

    # Preview time in seconds for action interpolation during inference
    # Higher values (0.03-0.05) provide smoother motion but more delay
    # Lower values (0.01-0.02) are more responsive but may cause jittering
    preview_time: float = 0.05  # Default 30ms for joint control

    # home状态，默认为全0，关节值
    home_position: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    # 初始状态，设置为采集数据的初始状态，关节值
    start_position: list[float] = field(
        # default_factory=lambda: [0, 1.52, 1.39, -1.35, 0, 0, 0.0008]
        default_factory=lambda: [0, 1.52, 1.39, -1.35, 0, 0, 0.0008]
        # default_factory=lambda: [0, 1.54, 1.28, -0.90, 0, 0, 0.0] # place
        # default_factory=lambda: [0.0, 1.48, 1.02, -1.06, 0, 0, 0.0]

    )
    # Camera configuration
    cameras: dict[str, CameraConfig] = field(default_factory=lambda: {})

    def __post_init__(self):
        # Default camera configuration if not provided
        if not self.cameras:
            self.cameras = {
                "image": RealSenseCameraConfig(
                    serial_number_or_name="420222071008",
                    fps=30,
                    width=640,
                    height=480,
                ),
                "wrist_image": RealSenseCameraConfig(
                    serial_number_or_name="348122070975",
                    fps=30,
                    width=640,
                    height=480,
                ),
            }

