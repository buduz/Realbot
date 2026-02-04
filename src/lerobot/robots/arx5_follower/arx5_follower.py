#!/usr/bin/env python

# Copyright 2025 The XenseRobotics Inc. team. All rights reserved.
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
import math
import time
from functools import cached_property
from typing import Any, Sequence
import numpy as np

import os
import sys
from pathlib import Path


from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot.robots.robot import Robot
from lerobot.robots.arx5_follower.config_arx5_follower import ARX5FollowerConfig

current_file = Path(__file__).resolve()
project_root = current_file.parents[4].as_posix()
arx5_sdk_path = os.path.join(project_root, "arx5-sdk", "python")
if arx5_sdk_path not in sys.path:
    sys.path.append(arx5_sdk_path)

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

logger = logging.getLogger(__name__)


class ARX5Follower(Robot):
    """
    [Single ARX5 Arm Follower Robot]

    A simplified version of BiARX5 for single-arm operation.
    Suitable for teleoperation with one follower arm.
    """

    config_class = ARX5FollowerConfig
    name = "ARX5_follower"

    def __init__(self, config: ARX5FollowerConfig):
        super().__init__(config)
        self.config = config

        # 机械臂控制器实例
        self.arm = None
        self._is_connected = False

        self._is_gravity_compensation_mode = True

        # 真正的动作控制模式, 关节控制/末段位姿（笛卡尔）控制
        self._is_joint_control_mode = False
        self._is_cartesian_control_mode = False

        # 设置推理时的动作预览时间（前瞻时间）
        if self.config.control_mode == 'cartesian_control':
            # 使用sdk默认的针对笛卡尔控制前瞻时间(0.1s)
            self.default_preview_time = 0.1
            logger.info("Cartesian control mode: using SDK default preview_time (0.1s)")
        else:
            self.default_preview_time = self.config.preview_time
            logger.info(f"Joint control mode: using preview_time {self.default_preview_time}s")

        # 数据采集和推理的键名保持一致
        self._joint_keys = [f"joint_{i+1}.pos" for i in range(6)]
        self._eef_keys = ["X_axis.pos", "Y_axis.pos", "Z_axis.pos", "RX_axis.pos", "RY_axis.pos", "RZ_axis.pos"]
        self._gripper_key = "gripper.pos"

        # 预分配命令缓冲区，避免重复内存分配
        self._cmd_buffer = None  # JointState buffer for joint control
        self._eef_cmd_buffer = None  # EEFState buffer for cartesian control

        # 定义归位位置和起始位置
        self._home_position = self.config.home_position
        self._start_position = self.config.start_position

        # 获取机械臂模型配置
        self.robot_config = arx5.RobotConfigFactory.get_instance().get_config(
            config.arm_model
        )

        # 初始化运动学求解器
        urdf_path = os.path.join(
            arx5_sdk_path, "..", "models", f"{config.arm_model}.urdf"
        )
        self._solver = arx5.Arx5Solver(
            urdf_path,
            self.robot_config.joint_dof,
            self.robot_config.joint_pos_min,
            self.robot_config.joint_pos_max,
        )
        if config.control_mode == 'cartesian_control':
            # 计算home状态和初始状态的末端位姿
            home_joint_pos = np.array(self._home_position[:6], dtype=np.float64)
            start_joint_pos = np.array(self._start_position[:6], dtype=np.float64)
            self._home_position_eef = np.concatenate([
                self._solver.forward_kinematics(home_joint_pos),
                [self._home_position[6]]  # gripper
            ])
            # 对于初始状态，令姿态（rpy）为0，便于插值
            start_eef_pose = self._solver.forward_kinematics(start_joint_pos)
            # start_eef_pose[3:6] = 0.0  
            self._start_position_eef = np.concatenate([
                start_eef_pose,
                [self._start_position[6]]  # gripper
            ])
            logger.info(f"EEF home position (FK): {self._home_position}")
            logger.info(f"EEF start position (FK with rpy=0): {self._start_position}")

        if config.control_mode == 'cartesian_control':
            # Cartesian controller requires background_send_recv = True
            controller_type = "cartesian_controller"
            use_background = True
        else:
            controller_type = "joint_controller"
            use_background = config.use_multithreading

        # 获取关节控制器配置
        self.controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
            controller_type, self.robot_config.joint_dof
        )
        logger.info(f"Using {controller_type} for control mode: {config.control_mode}")

        # 设置控制周期 (dt) 和 前瞻时间
        self.controller_config.controller_dt = config.controller_dt
        self.controller_config.default_preview_time = self.default_preview_time

        # 根据配置决定是否使用后台线程收发数据
        self.controller_config.background_send_recv = use_background

        # 初始化相机配置
        self.cameras = make_cameras_from_configs(config.cameras)
        # 设置 numpy 打印格式
        np.set_printoptions(precision=3, suppress=True)

    @property
    def _motors_ft(self) -> dict[str, type]:
        # 仅用于数据采集，保证采集到所有的数据（关节角+末段位姿+夹爪宽度）
        joint_names = [f"joint_{i}" for i in range(1, 7)]
        eef_names = ["X_axis", "Y_axis", "Z_axis", "RX_axis", "RY_axis", "RZ_axis"]
        gripper_names = ["gripper"]
        names = joint_names + eef_names + gripper_names
        return {f"{n}.pos": float for n in names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        print(f"camera_features: {self._cameras_ft}")
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self._is_connected
            and self.arm is not None
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def is_gravity_compensation_mode(self) -> bool:
        """Check if robot is currently in gravity compensation mode"""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self._is_gravity_compensation_mode


    def connect(self, calibrate: bool = False, go_to_start: bool = False) -> None:
        if self._is_connected:
            raise DeviceAlreadyConnectedError(
                f"{self} already connected, do not run `robot.connect()` twice."
            )

        try:
            logger.info(f"Creating arm controller (mode: {self.config.control_mode})...")
            if self.config.control_mode == 'cartesian_control':
                self.arm = arx5.Arx5CartesianController(
                    self.robot_config,
                    self.controller_config,
                    self.config.arm_port,
                )
            else:
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

        self._is_connected = True
        # 连接相机
        for cam in self.cameras.values():
            cam.connect()

        # set log lever
        self.set_log_level(self.config.log_level)

        # 机械臂归位 using sdk method
        self.reset_to_home()

        self.set_to_gravity_compensation_mode()

        # 初始化命令缓冲区，用于优化 send_action
        if self.config.control_mode == 'cartesian_control':
            self._eef_cmd_buffer = arx5.EEFState()
            self._cmd_buffer = None  # Not used in cartesian mode
        else:
            self._cmd_buffer = arx5.JointState(self.robot_config.joint_dof)
            self._eef_cmd_buffer = None  # Not used in joint mode

        # 移动到初始位置
        logger.info("ARX5 Follower Robot connected (Inference Mode).")
        if go_to_start:
            # 平滑移动到 start 位置，方便推理准备
            self.smooth_go_start(duration=2.0)
            logger.info(
                "✓ Robot go to start position, arm is ready in Position Control Mode"
            )
        else:
            logger.info(
                "Robot go to home position, arm is ready in Position Control Mode"
            )

        gain = self.arm.get_gain()
        logger.info(
            f"Current arm gain: {gain.kp()}, {gain.kd()}, {gain.gripper_kp}, {gain.gripper_kd}"
        )

        if self.config.control_mode == 'cartesian_control':
            self.set_to_normal_cartesian_control()
            logger.info("✅ Robot is now in cartesian control mode for inference")
        elif self.config.control_mode == 'joint_control':
            self.set_to_normal_position_control()
            logger.info("✅ Robot is now in joint position control mode for inference")
        else:
            logger.error(f"Invalid inference time control mode: {self.config.control_mode}")
            raise ValueError(f"Invalid inference time control mode: {self.config.control_mode}")
        logger.info(f"✅ Robot is now connected and ready for inference in {self.config.control_mode} mode.")

    @property
    def is_calibrated(self) -> bool:
        """
        ARX5 does not need to calibrate in runtime
        """
        return self.is_connected

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

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        joints_state = self.arm.get_joint_state()
        eef_state = self.arm.get_eef_state()
        joints = joints_state.pos().copy().tolist()
        eef = eef_state.pose_6d().copy().tolist()
        gripper_state = joints_state.gripper_pos
        gripper_torque = joints_state.gripper_torque
        state = joints + eef + [gripper_state]

        # 创建观测字典
        obs_dict.update(zip(self.action_features, state))

        # 异步读取所有相机图像
        camera_times = {}

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            image = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            obs_dict[cam_key] = image
            camera_times[cam_key] = dt_ms

        # 存储相机时间信息用来debug
        self.last_camera_times = camera_times

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        if self.config.control_mode == 'cartesian_control':
            # Cartesian mode: use EEF command
            cmd = self._eef_cmd_buffer
            pose_6d = cmd.pose_6d()
            for i, key in enumerate(self._eef_keys):
                pose_6d[i] = action.get(key, pose_6d[i])
            cmd.gripper_pos = action.get(self._gripper_key, cmd.gripper_pos)

            # Use set_eef_cmd - SDK will use default_preview_time for interpolation
            # Debug: Print commands before sending
            # print(
            #     f"Arm command - pose_6d: {cmd.pose_6d()}, gripper: {cmd.gripper_pos}"
            # )
            self.arm.set_eef_cmd(cmd)
        else:
            # Joint mode (including teach mode): use joint command
            cmd = self._cmd_buffer
            pos = cmd.pos()
            for i, key in enumerate(self._joint_keys):
                pos[i] = action.get(key, pos[i])
            cmd.gripper_pos = action.get(self._gripper_key, cmd.gripper_pos)
            # Debug: Print commands before sending
            # print(
            #     f"Arm command - pos: {cmd.pos()}, gripper: {cmd.gripper_pos}"
            # )
            self.arm.set_joint_cmd(cmd)
        # Return the input action
        return action

    @staticmethod
    def _ease_in_out_quad(t: float) -> float:
        """Smooth easing function used for joint interpolation."""
        tt = t * 2.0
        if tt < 1.0:
            return (tt * tt) / 2.0
        tt -= 1.0
        return -(tt * (tt - 2.0) - 1.0) / 2.0

    def move_joint_trajectory(
        self,
        target_joint_poses: Sequence[float] | Sequence[Sequence[float]],
        durations: float | Sequence[float],
        *,
        easing: str = "ease_in_out_quad",
        steps_per_segment: int | None = None,
    ) -> None:
        """Move the arm smoothly towards the provided joint targets.

        Uses send_action to send interpolated commands step by step.

        Args:
            target_joint_poses: A sequence of 6 or 7 joint values (including gripper)
                or a sequence of such sequences to execute multiple segments.
            durations: Duration in seconds for the corresponding target poses.
            easing: Easing profile to apply ("ease_in_out_quad" or "linear").
            steps_per_segment: Optional fixed number of interpolation steps per
                segment. When omitted the controller's ``controller_dt`` is used
                to compute the number of steps from the duration.

        Raises:
            DeviceNotConnectedError: If the robot is not connected.
            ValueError: If inputs are malformed.
        """

        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Normalize input to list of targets
        if isinstance(target_joint_poses[0], (int, float)):
            trajectory = [target_joint_poses]
        else:
            trajectory = list(target_joint_poses)

        if isinstance(durations, (int, float)):
            segment_durations = [float(durations)]
        else:
            segment_durations = [float(d) for d in durations]

        if len(trajectory) != len(segment_durations):
            raise ValueError(
                "target_joint_poses and durations must have the same length"
            )

        # Determine controller timestep (fallback to 10 ms if unavailable)
        controller_dt = getattr(self.config, "interpolation_controller_dt", 0.01)

        # Fetch the current joint position as starting state
        def _get_current_state() -> np.ndarray:
            state = self.arm.get_joint_state()
            return np.concatenate([state.pos().copy(), [state.gripper_pos]])

        current = _get_current_state()

        def _parse_target(values: Sequence[float], default: np.ndarray) -> np.ndarray:
            arr = np.asarray(values, dtype=np.float64)
            if arr.shape[0] not in (6, 7):
                raise ValueError(
                    "Target must provide 6 or 7 joint values"
                )
            if arr.shape[0] == 6:
                arr = np.concatenate([arr, [default[-1]]])
            return arr

        def _apply_easing(alpha: float) -> float:
            alpha = np.clip(alpha, 0.0, 1.0)
            if easing == "ease_in_out_quad":
                return self._ease_in_out_quad(alpha)
            if easing == "linear":
                return alpha
            raise ValueError(f"Unsupported easing profile: {easing}")

        try:
            for segment, duration in zip(trajectory, segment_durations, strict=True):
                target = _parse_target(segment, current)

                if duration <= 0:
                    action = dict(zip(self._joint_keys, target[:6].tolist()))
                    action[self._gripper_key] = float(target[6])
                    self.send_action(action)
                    current = target
                    continue

                steps = (
                    steps_per_segment
                    if steps_per_segment is not None
                    else max(1, int(math.ceil(duration / controller_dt)))
                )

                for step in range(1, steps + 1):
                    progress = step / steps
                    ratio = _apply_easing(progress)
                    interp = current + (target - current) * ratio

                    action = dict(zip(self._joint_keys, interp[:6].tolist()))
                    action[self._gripper_key] = float(interp[6])

                    self.send_action(action)
                    time.sleep(duration / steps if steps_per_segment else controller_dt)

                current = target
        except KeyboardInterrupt:
            logger.warn(
                "Joint trajectory interrupted by user. Holding current pose."
            )

    def move_eef_trajectory(
        self,
        target_eef_poses: Sequence[float] | Sequence[Sequence[float]],
        durations: float | Sequence[float],
        *,
        easing: str = "linear",
        steps_per_segment: int | None = None,
    ) -> None:
        """Move the arm smoothly towards the provided EEF targets (Cartesian mode).

        Uses send_action to send interpolated commands step by step.

        Args:
            target_eef_poses: A sequence of 6 or 7 values (x,y,z,roll,pitch,yaw + optional gripper)
                or a sequence of such sequences to execute multiple segments.
            durations: Duration in seconds for the corresponding target poses.
            easing: Easing profile to apply ("ease_in_out_quad" or "linear").
            steps_per_segment: Optional fixed number of interpolation steps per
                segment. When omitted the controller's ``controller_dt`` is used
                to compute the number of steps from the duration.

        Raises:
            DeviceNotConnectedError: If the robot is not connected.
            ValueError: If inputs are malformed or not in Cartesian mode.
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.control_mode != 'cartesian_control':
            raise ValueError("move_eef_trajectory requires CARTESIAN_CONTROL mode")

        # Normalize input to list of targets
        if isinstance(target_eef_poses[0], (int, float)):
            trajectory = [target_eef_poses]
        else:
            trajectory = list(target_eef_poses)

        if isinstance(durations, (int, float)):
            segment_durations = [float(durations)]
        else:
            segment_durations = [float(d) for d in durations]

        if len(trajectory) != len(segment_durations):
            raise ValueError(
                "target_eef_poses and durations must have the same length"
            )

        # Determine controller timestep (fallback to 10 ms if unavailable)
        controller_dt = getattr(self.config, "interpolation_controller_dt", 0.01)

        # Fetch the current EEF position as starting state
        def _get_current_state() -> np.ndarray:
            state = self.arm.get_eef_state()
            return np.concatenate([state.pose_6d().copy(), [state.gripper_pos]])

        current = _get_current_state()

        def _parse_target(values: Sequence[float], default: np.ndarray) -> np.ndarray:
            arr = np.asarray(values, dtype=np.float64)
            if arr.shape[0] not in (6, 7):
                raise ValueError(
                    "Target must provide 6 EEF values (+ optional gripper)"
                )
            if arr.shape[0] == 6:
                arr = np.concatenate([arr, [default[-1]]])
            return arr

        def _apply_easing(alpha: float) -> float:
            alpha = np.clip(alpha, 0.0, 1.0)
            if easing == "ease_in_out_quad":
                return self._ease_in_out_quad(alpha)
            if easing == "linear":
                return alpha
            raise ValueError(f"Unsupported easing profile: {easing}")

        try:
            for segment, duration in zip(trajectory, segment_durations, strict=True):
                target = _parse_target(segment, current)

                if duration <= 0:
                    action = dict(zip(self._eef_keys, target[:6].tolist()))
                    action[self._gripper_key] = float(target[6])
                    self.send_action(action)
                    current = target
                    continue

                steps = (
                    steps_per_segment
                    if steps_per_segment is not None
                    else max(1, int(math.ceil(duration / controller_dt)))
                )

                for step in range(1, steps + 1):
                    progress = step / steps
                    ratio = _apply_easing(progress)
                    interp = current + (target - current) * ratio

                    action = dict(zip(self._eef_keys, interp[:6].tolist()))
                    action[self._gripper_key] = float(interp[6])

                    self.send_action(action)
                    time.sleep(duration / steps if steps_per_segment else controller_dt)

                current = target

        except KeyboardInterrupt:
            logger.warn(
                "EEF trajectory interrupted by user. Holding current pose."
            )

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Reset to home and set to damping mode for safety
        try:
            logger.info("Disconnecting arm...")
            self.arm.reset_to_home()
            self.arm.set_to_damping()
            logger.info("✓ Arm disconnected successfully")
        except Exception as e:
            logger.warning(f"Arm disconnect failed: {e}")

        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()

        # Destroy arm object - this triggers SDK cleanup
        self.arm = None

        self._is_connected = False

        logger.info(f"{self} disconnected.")

    def set_log_level(self, level: str):
        """Set robot log level"""
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

    def reset_to_home(self):
        """Reset arm to home position"""
        if self.arm is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self.arm.reset_to_home()
        logger.info("Arm reset to home position.")

    def set_to_gravity_compensation_mode(self):
        """Switch from normal position control or cartesian control to gravity compensation mode.

        Uses SDK's set_to_gravity_compensation() which:
        1. Sets kp=0, kd=default (damping only, no position control)
        2. Resets interpolator to current position (important for Cartesian mode)
        3. Gravity compensation is handled by SDK if controller_config.gravity_compensation=True
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self._is_gravity_compensation_mode:
            logger.info("Arm is already in gravity compensation mode")
            return

        logger.info("Switching to gravity compensation mode...")

        # Use SDK's set_to_gravity_compensation() which properly resets the interpolator
        if self._is_joint_control_mode:
            logger.info("Switching to gravity compensation mode from joint control mode...")
        elif self._is_cartesian_control_mode:
            logger.info("Switching to gravity compensation mode from cartesian control mode...")

        self.arm.set_to_gravity_compensation()
        # Update control mode state
        self._is_gravity_compensation_mode = True
        self._is_joint_control_mode = False
        self._is_cartesian_control_mode = False

        logger.info("✅ Arm is now in gravity compensation mode.")

    def set_to_normal_position_control(self):
        """Switch from gravity compensation to normal position control or cartesian control mode"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        logger.info("Switching to normal position control mode...")

        is_joint_mode = self.config.control_mode == 'joint_control'

        if self._is_gravity_compensation_mode and is_joint_mode:
            # Reset to default gain
            default_gain = self.arm.get_gain()
            default_gain.kp()[:] = self.controller_config.default_kp * 0.5
            default_gain.kd()[:] = self.controller_config.default_kd * 1.5
            default_gain.gripper_kp = self.controller_config.default_gripper_kp
            default_gain.gripper_kd = self.controller_config.default_gripper_kd

            self.arm.set_gain(default_gain)

            # Update control mode state
            self._is_joint_control_mode = True
            self._is_cartesian_control_mode = False
            self._is_gravity_compensation_mode = False
            logger.info("✅ Arm is now in normal position control mode")
        elif not self._is_gravity_compensation_mode and is_joint_mode:
            logger.info("Arm is already in normal position control mode")
            return
        else:
            logger.warn(f"Can't switch to normal position control mode from current mode: {self.config.control_mode}")
            return

    def set_to_normal_cartesian_control(self):
        """Switch from gravity compensation to normal cartesian control mode"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        logger.info("Switching to normal cartesian control mode...")

        is_cartesian_mode = self.config.control_mode == 'cartesian_control'

        if self._is_gravity_compensation_mode and is_cartesian_mode:
            # Reset to default gain
            default_gain = self.arm.get_gain()
            default_gain.kp()[:] = self.controller_config.default_kp
            default_gain.kd()[:] = self.controller_config.default_kd
            default_gain.gripper_kp = self.controller_config.default_gripper_kp
            default_gain.gripper_kd = self.controller_config.default_gripper_kd

            self.arm.set_gain(default_gain)

            # Update control mode state
            self._is_joint_control_mode = False
            self._is_cartesian_control_mode = True
            self._is_gravity_compensation_mode = False

            logger.info("✅ Arm is now switch from gravity compensation to normal cartesian control mode")
        elif not self._is_gravity_compensation_mode and is_cartesian_mode:
            logger.info("Arm is already in normal cartesian control mode")
            return
        else:
            logger.warn(
                f"Can't switch to normal cartesian control mode from current mode: {self.config.control_mode}"
            )
            return

    def _calculate_motion_duration(
        self,
        target: np.ndarray,
        min_duration: float = 0.5,
        speed_factor: float = 2.0,
    ) -> float:
        """
        Calculate motion duration based on maximum joint/EEF position error.

        This follows the SDK's reset_to_home logic:
        duration = max(max_pos_error, min_duration)

        Args:
            target: Target position (7 elements: 6 joints/pose + gripper)
            min_duration: Minimum duration in seconds (default: 1.0)
            speed_factor: Multiplier for speed adjustment (default: 2.0)

        Returns:
            Calculated duration in seconds
        """
        # Always use Joint space for duration calculation (consistent units in radians)
        # This follows SDK's reset_to_home logic which uses joint position error
        state = self.arm.get_joint_state()
        current = np.concatenate([state.pos(), [state.gripper_pos]])

        # Calculate maximum position error (excluding gripper)
        max_error = np.abs(current[:6] - target[:6]).max()

        # Duration = max(max_error, min_duration) * speed_factor
        duration = max(max_error, min_duration) * speed_factor
        logger.info(f"Calculated motion duration: {duration:.1f} seconds")
        return duration
    

    def smooth_go_start(
        self, duration: float | None = None, easing: str = "ease_in_out_quad"
    ) -> None:
        """
        Smoothly move the arm to the start position using trajectory interpolation.

        For Joint mode:
        1. Switches to normal position control mode
        2. Moves the arm to start position over the specified duration
        3. Switches back to gravity compensation mode

        For Cartesian mode:
        1. Moves the arm to start EEF position over the specified duration
        (No mode switching needed - already in position control)

        Args:
            duration: Duration in seconds for the movement. If None, automatically
                calculated based on distance to target (like SDK's reset_to_home).
            easing: Easing profile to apply ("ease_in_out_quad" or "linear")

        Raises:
            DeviceNotConnectedError: If the robot is not connected.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Calculate duration if not provided
        if duration is None:
            target = np.array(self._start_position)
            duration = self._calculate_motion_duration(target)

        logger.info(f"Smoothly going to start position over {duration:.1f} seconds...")

        if self.config.control_mode == 'cartesian_control':
            # Cartesian mode: use EEF trajectory
            logger.info("Cartesian mode: use EEF trajectory interpolation.")
            state = self.arm.get_eef_state()
            current_cmd = arx5.EEFState(state.pose_6d(), state.gripper_pos)
            # Must set a future timestamp (SDK requires timestamp > current_time for interpolation)
            current_cmd.timestamp = self.arm.get_timestamp() + 0.01
            self.arm.set_eef_cmd(current_cmd)

            # Now safe to switch to normal cartesian control
            self.set_to_normal_cartesian_control()

            self.move_eef_trajectory(
                target_eef_poses=self._start_position_eef.copy(),
                durations=duration,
                easing=easing,
            )
            logger.info(f"✅ Successfully going to start position in {self.config.control_mode} mode")
        else:
            # Joint mode: use joint trajectory interpolation
            logger.info("Joint mode: use joint trajectory interpolation.")
            # First, set current position as target to avoid large position error
            state = self.arm.get_joint_state()

            # Set current position as command to avoid SDK protection
            current_cmd = arx5.JointState(self.robot_config.joint_dof)
            current_cmd.pos()[:] = state.pos()
            current_cmd.gripper_pos = state.gripper_pos

            self.arm.set_joint_cmd(current_cmd)

            # Now safe to switch to normal position control
            self.set_to_normal_position_control()

            # Execute smooth trajectory to start position
            self.move_joint_trajectory(
                target_joint_poses=self._start_position.copy(),
                durations=duration,
                easing=easing,
            )
            logger.info(f"✅ Successfully going to start position in {self.config.control_mode} mode")

    def smooth_go_home(
        self, duration: float | None = None, easing: str = "ease_in_out_quad"
    ) -> None:
        """
        Smoothly move the arm to the home position using trajectory interpolation.

        For Joint mode:
        1. Switches to normal position control mode
        2. Moves the arm to home position over the specified duration
        3. Switches back to gravity compensation mode

        For Cartesian mode:
        1. Moves the arm to home EEF position over the specified duration
        (No mode switching needed - already in position control)

        Args:
            duration: Duration in seconds for the movement. If None, automatically
                calculated based on distance to target (like SDK's reset_to_home).
            easing: Easing profile to apply ("ease_in_out_quad" or "linear")

        Raises:
            DeviceNotConnectedError: If the robot is not connected.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Calculate duration if not provided
        if duration is None:
            target = np.array(self._home_position)
            duration = self._calculate_motion_duration(target)

        logger.info(
            f"Smoothly returning to home position over {duration:.1f} seconds..."
        )

        if self.config.control_mode == 'cartesian_control':
            # Cartesian mode: use EEF trajectory
            logger.info("Cartesian mode: use EEF trajectory interpolation.")

            # Set current position as command first (required for interpolator)
            state = self.arm.get_eef_state()
            current_cmd = arx5.EEFState(state.pose_6d(), state.gripper_pos)
            current_cmd.timestamp = self.arm.get_timestamp() + 0.01
            self.arm.set_eef_cmd(current_cmd)

            # Switch to normal cartesian control (if in gravity compensation mode)
            self.set_to_normal_cartesian_control()

            self.move_eef_trajectory(
                target_eef_poses=self._home_position_eef.copy(),
                durations=duration,
                easing=easing,
            )
            logger.info(
                f"✅ Successfully returned to home position in {self.config.control_mode} mode"
            )
        else:
            # Joint mode: need to switch modes
            # First, set current position as target to avoid large position error
            state = self.arm.get_joint_state()

            # Set current position as command to avoid SDK protection
            current_cmd = arx5.JointState(self.robot_config.joint_dof)
            current_cmd.pos()[:] = state.pos()
            current_cmd.gripper_pos = state.gripper_pos

            self.arm.set_joint_cmd(current_cmd)

            # Now safe to switch to normal position control
            self.set_to_normal_position_control()

            # Execute smooth trajectory to home position
            self.move_joint_trajectory(
                target_joint_poses=self._home_position.copy(),
                durations=duration,
                easing=easing,
            )

            # Switch back to gravity compensation mode (only for joint mode)
            self.set_to_gravity_compensation_mode()
            logger.info(
                "✅ Successfully returned to home position and switched to gravity compensation mode"
            )