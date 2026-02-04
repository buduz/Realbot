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

import logging
import time
import numpy as np

import os
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parents[4].as_posix()
arx5_sdk_path = os.path.join(project_root, "arx5-sdk", "python")
if arx5_sdk_path not in sys.path:
    sys.path.insert(0, arx5_sdk_path)

try:
    import arx5_interface as arx5
except ImportError as e:
    if "LogLevel" in str(e) and "already registered" in str(e):
        # LogLevel already registered, try to get the existing module
        if "arx5_interface" in sys.modules:
            arx5 = sys.modules["arx5_interface"]
        else:
            raise e
    else:
        raise e

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.arx5_leader.config_arx5_leader import ARX5LeaderConfig

logger = logging.getLogger(__name__)


class ARX5Leader(Teleoperator):
    """
    ARX5 Leader.
    """

    config_class = ARX5LeaderConfig
    name = "ARX5_follower"

    def __init__(self, config: ARX5LeaderConfig):
        super().__init__(config)
        self.config = config

        # 机械臂控制器实例
        self.arm = None
        self._is_connected = False

        # 控制模式状态标志
        self._is_gravity_compensation_mode = False
        self._is_position_control_mode = False

        # 设置推理时的动作预览时间（前瞻时间）
        # 如果是推理模式，使用配置的时间；否则为 0（实时控制）
        self.default_preview_time = (
            self.config.preview_time if self.config.inference_mode else 0.0
        )

        # RPC 通信超时设置
        self.rpc_timeout: float = getattr(config, "rpc_timeout", 5.0)

        # 性能优化：预先计算好关节名称字符串列表，避免在循环中重复生成
        self._joint_keys = [f"joint_{i+1}.pos" for i in range(6)]


        # 获取机械臂模型配置
        self.robot_config = arx5.RobotConfigFactory.get_instance().get_config(
            config.arm_model
        )

        # 设置夹爪张开的读数范围
        # self.robot_config.gripper_open_readout = config.gripper_open_readout

        # 获取关节控制器配置
        self.controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
            "joint_controller", self.robot_config.joint_dof
        )

        # 设置控制周期 (dt) 和 前瞻时间
        self.controller_config.controller_dt = config.controller_dt
        self.controller_config.default_preview_time = self.default_preview_time

        # 根据配置决定是否使用后台线程收发数据
        self.controller_config.background_send_recv = config.use_multithreading

        # 设置 numpy 打印格式
        np.set_printoptions(precision=3, suppress=True)

    @property
    def _motors_ft(self) -> dict[str, type]:
        # ARX5 has 6 joints + 1 gripper
        # joint_names = [f"joint_{i}" for i in range(1, 7)] + ["gripper"]
        # return {f"{joint}.pos": float for joint in joint_names}
        joint_names = [f"joint_{i}" for i in range(1, 7)]
        eef_names = ["X_axis", "Y_axis", "Z_axis", "RX_axis", "RY_axis", "RZ_axis"]
        gripper_names = ["gripper"]
        names = joint_names + eef_names + gripper_names
        return {f"{n}.pos": float for n in names}

    @property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return (
            self._is_connected
            and self.arm is not None
        )

    def is_gravity_compensation_mode(self) -> bool:
        """Check if robot is currently in gravity compensation mode"""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self._is_gravity_compensation_mode

    def is_position_control_mode(self) -> bool:
        """Check if robot is currently in position control mode"""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self._is_position_control_mode

    def connect(self, calibrate: bool = False, go_to_start: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                f"{self} already connected, do not run `robot.connect()` twice."
            )

        try:
            logger.info("Creating arm controller...")
            self.arm = arx5.Arx5JointController(
                self.robot_config,
                self.controller_config,
                self.config.arm_port,
            )
            time.sleep(0.5)
            logger.info("✓ Arm controller created successfully")
            logger.info(
                f"preview_time: {self.controller_config.default_preview_time}"
            )
        except Exception as e:
            logger.error(f"Failed to create robot controller: {e}")
            self.arm = None
            raise e

        # set log lever
        self.set_log_level(self.config.log_level)

        # 配置“重力补偿”模式的增益 (Gain)
        # 重力补偿模式下，Kp (比例增益) 设为 0，意味着没有位置刚性，人可以轻松拖动机械臂
        # Kd (微分增益/阻尼) 设为较小值，提供一点阻力防止抖动
        zero_grav_gain = self.arm.get_gain()
        zero_grav_gain.kp()[:] = 0.0
        zero_grav_gain.kd()[:] = self.controller_config.default_kd * 0.15
        
        # 夹爪的增益设置
        zero_grav_gain.gripper_kp = 0.0
        zero_grav_gain.gripper_kd = self.controller_config.default_gripper_kd * 0.25

        # 应用增益设置
        self.arm.set_gain(zero_grav_gain)

        self._is_connected = True

        # 默认为重力补偿模式
        self._is_gravity_compensation_mode = True
        self._is_position_control_mode = False

        # 移动到初始位置
        logger.info("ARX5 Follower Robot connected.")

        gain = self.arm.get_gain()
        logger.info(
            f"Current arm gain: {gain.kp()}, {gain.kd()}, {gain.gripper_kp}, {gain.gripper_kd}"
        )

        # 如果是推理模式 (Inference Mode)，切换到位置控制
        # 推理时机械臂由模型控制，需要刚性 (High Gain)，不能是重力补偿模式
        if self.config.inference_mode:
            self.set_to_normal_position_control()
            logger.info("✓ Robot is now in normal position control mode for inference or MASTER/VR teleoperation")

    @property
    def is_calibrated(self) -> bool:
        """
        ARX5 does not need to calibrate in runtime
        """
        return self.is_connected
    
    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def calibrate(self) -> None:
        """ARX5 does not need to calibrate in runtime"""
        logger.info("ARX5 does not need to calibrate in runtime, skip...")
        return

    def configure(self) -> None:
        """
        Configure the robot
        """
        pass

    def setup_motors(self) -> None:
        """ARX5 motors are pre-configured, no runtime setup needed"""
        logger.info(
            f"{self} ARX5 motors are pre-configured, no runtime setup needed"
        )
        logger.info("Motor IDs are defined in the robot configuration:")
        logger.info("  - Joint motors: [1, 2, 4, 5, 6, 7]")
        logger.info("  - Gripper motor: 8")
        logger.info("Make sure your hardware matches these ID configurations")
        return

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        joints_state = self.arm.get_joint_state()
        eef_state = self.arm.get_eef_state()
        joints = joints_state.pos().copy().tolist()
        eef = eef_state.pose_6d().copy().tolist()
        gripper_state = max(0.0, min(joints_state.gripper_pos * 5, 0.088))
        gripper_torque = joints_state.gripper_torque
        # print(f'gripper_torque:{gripper_torque}')
        state = joints + eef + [gripper_state]
        action = {
            n: a for n, a in zip(self.action_features, state)
        }
        return action

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Destroy arm object - this triggers SDK cleanup
        self.arm = None

        self._is_connected = False

        logger.info(f"{self} disconnected.")

    def set_log_level(self, level: str):
        """Set robot log level

        Args:
            level: Log level string, supports: TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL, OFF
        """
        # Convert string to LogLevel enum
        log_level_map = {
            "TRACE": arx5.LogLevel.TRACE,
            "DEBUG": arx5.LogLevel.DEBUG,
            "INFO": arx5.LogLevel.INFO,
            "WARNING": arx5.LogLevel.WARNING,
            "ERROR": arx5.LogLevel.ERROR,
            "CRITICAL": arx5.LogLevel.CRITICAL,
            "OFF": arx5.LogLevel.OFF,
        }

        if level.upper() not in log_level_map:
            raise ValueError(
                f"Invalid log level: {level}. Supported levels: {list(log_level_map.keys())}"
            )

        log_level = log_level_map[level.upper()]

        # Set log level for arm if connected
        if self.arm is not None:
            self.arm.set_log_level(log_level)

    def set_to_gravity_compensation_mode(self):
        """切换到重力补偿模式 (用于示教/手动拖动,主臂使用)"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        logger.info("Switching to gravity compensation mode...")

        # Reset to zero PD with 0.15 * default kd
        zero_grav_gain = arx5.Gain(self.robot_config.joint_dof)
        zero_grav_gain.kp()[:] = 0.0
        zero_grav_gain.kd()[:] = self.controller_config.default_kd * 0.15
        zero_grav_gain.gripper_kp = 0.0
        zero_grav_gain.gripper_kd = self.controller_config.default_gripper_kd * 0.25

        self.arm.set_gain(zero_grav_gain)

        # Update control mode state
        self._is_gravity_compensation_mode = True
        self._is_position_control_mode = False

        logger.info("✓ Arm is now in gravity compensation mode")

    def set_to_normal_position_control(self):
        """切换到位置控制模式 (用于回放/推理,从臂使用)"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        logger.info("Switching to normal position control mode...")

        # Reset to default gain
        default_gain = self.arm.get_gain()
        default_gain.kp()[:] = self.controller_config.default_kp * 0.5
        default_gain.kd()[:] = self.controller_config.default_kd * 1.5
        default_gain.gripper_kp = self.controller_config.default_gripper_kp
        default_gain.gripper_kd = self.controller_config.default_gripper_kd

        self.arm.set_gain(default_gain)

        # Update control mode state
        self._is_gravity_compensation_mode = False
        self._is_position_control_mode = True

        logger.info("✓ Arm is now in normal position control mode")