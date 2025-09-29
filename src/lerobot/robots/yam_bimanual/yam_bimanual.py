#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from functools import cached_property
from typing import Any

from i2rt.robots.utils import GripperType

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_yam_bimanual import YamBimanualConfig
from i2rt.robots.get_robot import get_yam_robot

logger = logging.getLogger(__name__)


class DummyMotorClass:
    def __init__(self):
        # Motors will be dynamically determined based on robot configuration
        self.motors = {
            "left_motor_0": 0,
            "left_motor_1": 0,
            "left_motor_2": 0,
            "left_motor_3": 0,
            "left_motor_4": 0,
            "left_motor_5": 0,
            "left_motor_6": 0,
            "right_motor_0": 0,
            "right_motor_1": 0,
            "right_motor_2": 0,
            "right_motor_3": 0,
            "right_motor_4": 0,
            "right_motor_5": 0,
            "right_motor_6": 0,
        }

class YamBimanual(Robot):
    """
    - [Koch v1.0](https://github.com/AlexanderKoch-Koch/low_cost_robot), with and without the wrist-to-elbow
        expansion, developed by Alexander Koch from [Tau Robotics](https://tau-robotics.com)
    - [Koch v1.1](https://github.com/jess-moss/koch-v1-1) developed by Jess Moss
    """

    config_class = YamBimanualConfig
    name = "yam_bimanual"

    def __init__(self, config: YamBimanualConfig):
        super().__init__(config)
        self.config = config
        self.bus = DummyMotorClass()
        self.cameras = make_cameras_from_configs(config.cameras)
        self.robot_left = None
        self.robot_right = None
        self.has_connected = False

    @property
    def _motors_ft(self) -> dict[str, type]:
        # Dynamically determine motor features based on actual robot configuration
        features = {}
        # if hasattr(self, 'robot_left') and self.robot_left:
        #     # Assuming 7 DOF for each arm (6 joints + gripper)
        for i in range(7):
            features[f"left_motor_{i}.pos"] = float
        # if hasattr(self, 'robot_right') and self.robot_right:
        for i in range(7):
            features[f"right_motor_{i}.pos"] = float

        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.has_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.has_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.robot_left = get_yam_robot(channel=self.config.port, gripper_type=GripperType.LINEAR_4310, zero_gravity_mode=False)
        self.robot_right = get_yam_robot(channel=self.config.port_right, gripper_type=GripperType.LINEAR_4310, zero_gravity_mode=False)
        self.has_connected = True

        for cam in self.cameras.values():
            cam.connect()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def setup_motors(self) -> None:
        return

    def get_observation(self) -> dict[str, Any]:
        if not self.has_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        joint_pos_left = self.robot_left.get_joint_pos()
        joint_pos_right = self.robot_right.get_joint_pos()

        # Create properly prefixed motor names
        motors_left = [f'left_motor_{i}' for i in range(len(joint_pos_left))]
        motors_right = [f'right_motor_{i}' for i in range(len(joint_pos_right))]
        
        # Build observation dictionary with proper prefixes
        obs_dict = {}
        for motor, val in zip(motors_left, joint_pos_left):
            obs_dict[f"{motor}.pos"] = val
        for motor, val in zip(motors_right, joint_pos_right):
            obs_dict[f"{motor}.pos"] = val

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action (dict[str, float]): The goal positions for the motors.

        Returns:
            dict[str, float]: The action sent to the motors, potentially clipped.
        """
        if not self.has_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Separate left and right motor actions
        left_goal_pos = {}
        right_goal_pos = {}
        
        for key, val in action.items():
            if key.endswith(".pos"):
                motor_name = key.removesuffix(".pos")
                if motor_name.startswith("left_motor_"):
                    left_goal_pos[motor_name] = val
                elif motor_name.startswith("right_motor_"):
                    right_goal_pos[motor_name] = val

        # Convert to ordered lists
        left_positions = [left_goal_pos[f"left_motor_{i}"] for i in range(len(left_goal_pos))]
        right_positions = [right_goal_pos[f"right_motor_{i}"] for i in range(len(right_goal_pos))]

        # Send goal positions to the arms
        self.robot_left.command_joint_pos(left_positions)
        self.robot_right.command_joint_pos(right_positions)
        return action

    def get_joint_positions(self) -> list[float]:
        arr = []
        left_pos = self.robot_left.get_joint_pos()
        right_pos = self.robot_right.get_joint_pos()
        for i in range(len(left_pos)):
            arr.append(left_pos[i])
        for i in range(len(right_pos)):
            arr.append(right_pos[i])
        return arr

    def command_joint_pos(self, joint_pos: list[float]) -> None:
        self.robot_left.command_joint_pos(joint_pos[:7])
        self.robot_right.command_joint_pos(joint_pos[7:])

    def disconnect(self):
        if not self.has_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.robot_left.close()
        self.robot_right.close()

        # self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
