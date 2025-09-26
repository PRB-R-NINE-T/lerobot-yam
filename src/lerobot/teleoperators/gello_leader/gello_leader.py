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
import yaml
import os
import numpy as np

from i2rt.robots.utils import GripperType
from gello.robots.dynamixel import DynamixelRobot

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_gello_leader import GelloLeaderConfig

logger = logging.getLogger(__name__)


class GelloLeader(Teleoperator):
    """
    [Yam](https://i2rt.com/products/yam-manipulator) developed by I2RT
    """

    config_class = GelloLeaderConfig
    name = "gello_leader"

    def __init__(self, config: GelloLeaderConfig):
        super().__init__(config)
        self.config = config
        self.bus = {}
        self.has_connected = False
        self.robot = None
        self.robot_left = None
        self.robot_right = None
        
        # Load YAML config
        self._load_yaml_config()

    def _load_yaml_config(self):
        """Load and parse the YAML configuration file."""
        # Get the directory of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.yaml")
        
        with open(config_path, 'r') as file:
            self.yaml_config = yaml.safe_load(file)
        
        # Create robot instances from config
        self._create_robots()
    
    def _create_robots(self):
        """Create DynamixelRobot instances for left and right robots."""
        # Parse agent_left config
        agent_left_config = self.yaml_config['agent_left']
        dynamixel_left_config = agent_left_config['dynamixel_config']
        
        self.robot_left = DynamixelRobot(
            joint_ids=dynamixel_left_config['joint_ids'],
            joint_offsets=list(dynamixel_left_config['joint_offsets']),
            real=True,
            joint_signs=list(dynamixel_left_config['joint_signs']),
            port=agent_left_config['port'],
            gripper_config=tuple(dynamixel_left_config['gripper_config']),
            start_joints=np.array(agent_left_config['start_joints']),
        )
        
        # Parse agent_right config
        agent_right_config = self.yaml_config['agent_right']
        dynamixel_right_config = agent_right_config['dynamixel_config']
        
        self.robot_right = DynamixelRobot(
            joint_ids=dynamixel_right_config['joint_ids'],
            joint_offsets=list(dynamixel_right_config['joint_offsets']),
            real=True,
            joint_signs=list(dynamixel_right_config['joint_signs']),
            port=agent_right_config['port'],
            gripper_config=tuple(dynamixel_right_config['gripper_config']),
            start_joints=np.array(agent_right_config['start_joints']),
        )

    @property
    def action_features(self) -> dict[str, type]:
        # Return features for both left and right robots
        features = {}
        
        if hasattr(self, 'robot_left') and self.robot_left:
            num_joints_left = self.robot_left.num_dofs()
            for i in range(num_joints_left):
                features[f"left_motor_{i}.pos"] = float
                
        if hasattr(self, 'robot_right') and self.robot_right:
            num_joints_right = self.robot_right.num_dofs()
            for i in range(num_joints_right):
                features[f"right_motor_{i}.pos"] = float
                
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.has_connected

    def connect(self, calibrate: bool = True):
        if self.has_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # The robot_left and robot_right are already created in __init__ from YAML config
        # No additional setup needed here as DynamixelRobot handles connection internally
        
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
        
        # Get actions from both robots
        action_left = self.robot_left.get_joint_state()
        action_right = self.robot_right.get_joint_state()
        
        # Create motor names for left and right arms
        motors_left = [f'left_motor_{i}' for i in range(len(action_left))]
        motors_right = [f'right_motor_{i}' for i in range(len(action_right))]
        
        # Combine actions from both robots
        action = {}
        for motor, val in zip(motors_left, action_left):
            action[f"{motor}.pos"] = val
        for motor, val in zip(motors_right, action_right):
            action[f"{motor}.pos"] = val
            
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.has_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.robot_left:
            self.robot_left.close()
        if self.robot_right:
            self.robot_right.close()

        self.has_connected = False
        logger.info(f"{self} disconnected.")

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        # self.robot.zero_torque_mode()
        # self.robot.zero_torque_mode()
        return

    def setup_motors(self) -> None:
        return