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
from .config_yam_follower import YamFollowerConfig
from i2rt.robots.get_robot import get_yam_robot

logger = logging.getLogger(__name__)


class DummyMotorClass:
    def __init__(self):
        self.motors = {
            "motor_0": 0,
            "motor_1": 0,
            "motor_2": 0,
            "motor_3": 0,
            "motor_4": 0,
            "motor_5": 0,
            "motor_6": 0,
        }

class YamFollower(Robot):
    """
    - [Koch v1.0](https://github.com/AlexanderKoch-Koch/low_cost_robot), with and without the wrist-to-elbow
        expansion, developed by Alexander Koch from [Tau Robotics](https://tau-robotics.com)
    - [Koch v1.1](https://github.com/jess-moss/koch-v1-1) developed by Jess Moss
    """

    config_class = YamFollowerConfig
    name = "yam_follower"

    def __init__(self, config: YamFollowerConfig):
        super().__init__(config)
        self.config = config
        self.bus = DummyMotorClass()
        self.cameras = make_cameras_from_configs(config.cameras)
        self.robot = None
        self.has_connected = False

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

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

        self.robot = get_yam_robot(channel=self.config.port, gripper_type=GripperType.LINEAR_4310, zero_gravity_mode=False)
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
        joint_pos = self.robot.get_joint_pos()
        
        motors = ['motor_0', 'motor_1', 'motor_2', 'motor_3', 'motor_4', 'motor_5', 'motor_6']
        obs_dict = {f"{motor}.pos": val for motor, val in zip(motors, joint_pos)}

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

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        goal_pos = list(goal_pos.values())

        curr_pos = self.robot.get_joint_pos()

        # Send goal position to the arm
        self.robot.command_joint_pos(goal_pos)
        
        return action

    def disconnect(self):
        if not self.has_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.robot.close()

        # self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
