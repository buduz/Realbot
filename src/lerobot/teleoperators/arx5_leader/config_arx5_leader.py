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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("ARX5_leader")
@dataclass
class ARX5LeaderConfig(TeleoperatorConfig):
    """Configuration for single ARX5 arm leader robot."""
    arm_model: str = "X5" 
    arm_port: str = "can1"
    log_level: str = "DEBUG"
    use_multithreading: bool = True
    rpc_timeout: float = 10.0

    # Control parameters
    controller_dt: float = 0.005  # 200Hz control loop
    interpolation_controller_dt: float = 0.01

    # Mode settings
    inference_mode: bool = False    # 主臂是重力补偿模式, 从臂是推理模式

    # Preview time in seconds for action interpolation during inference
    # Higher values (0.03-0.05) provide smoother motion but more delay
    # Lower values (0.01-0.02) are more responsive but may cause jittering
    preview_time: float = 0.0  # Default 0ms, can be set to 30ms for smooth inference

    # Gripper calibration
    # gripper_open_readout: float = -3.4
