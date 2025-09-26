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

from i2rt.robots.utils import GripperType

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from i2rt.robots.get_robot import get_yam_robot

from ..teleoperator import Teleoperator
from .config_yam import YamLeaderConfig

logger = logging.getLogger(__name__)


class YamLeader(Teleoperator):
    """
    [Yam](https://i2rt.com/products/yam-manipulator) developed by I2RT
    """

    config_class = YamLeaderConfig
    name = "yam_leader"

    def __init__(self, config: YamLeaderConfig):
        super().__init__(config)
        self.config = config
        self.bus = {}
        self.has_connected = False
        self.robot = None

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.has_connected

    def connect(self, calibrate: bool = True):
        if self.has_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.robot = get_yam_robot(channel=self.config.port, gripper_type=GripperType.LINEAR_4310, zero_gravity_mode=True)
        self.has_connected = True

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def get_action(self) -> dict[str, float]:
        if not self.has_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        action = self.robot.get_joint_pos()
        motors = ['motor_0', 'motor_1', 'motor_2', 'motor_3', 'motor_4', 'motor_5', 'motor_6']
        action = {f"{motor}.pos": val for motor, val in zip(motors, action)}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.has_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.robot.close()

        logger.info(f"{self} disconnected.")

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        # self.robot.zero_torque_mode()
        # self.robot.zero_torque_mode()
        return

    def setup_motors(self) -> None:
        return