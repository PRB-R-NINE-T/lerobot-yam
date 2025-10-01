import io
import logging
import threading
import time
import os
from datetime import datetime
from PIL import Image
import cv2
import numpy as np
import requests
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.robots import yam_bimanual
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.yam_bimanual.config_yam_bimanual import YamBimanualConfig

# Configure logging to show timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create output directory for images
output_dir = "outputs/captured_images"
os.makedirs(output_dir, exist_ok=True)

# Initialize cameras
# camera_1, camera_4 = initialize_cameras()
print("Cameras initialized")
num_steps = 5000
# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).
# 38.80.152.248:30982
client = websocket_client_policy.WebsocketClientPolicy("38.80.152.248", port=32318)
# Example state and task instruction (you should replace these with actual values)
state = np.zeros(14)  # Replace with actual robot state
task_instruction = "fold napkins"  # Replace with actual task instruction

cfg = YamBimanualConfig(
    port="can_left",
    port_right="can_right",
)

robot = make_robot_from_config(cfg)

robot.connect()

cameras = make_cameras_from_configs({
            "top": OpenCVCameraConfig(
                index_or_path=0,
                fps=30,
                width=640,
                height=480,
            ),
            "left": OpenCVCameraConfig(
                index_or_path=4,
                fps=30,
                width=640,
                height=480,
            ),
            "right": OpenCVCameraConfig(
                index_or_path=8,
                fps=30,
                width=640,
                height=480,
            ),
        })

for cam in cameras.values():
    cam.connect()

print(robot.command_joint_pos([0] * 14))

current_level = "TOP"  # Ask user for initial level
print(f"Starting with level: {current_level}")

item_for_next_two_chunks = None
skip = 0
last_json = None

def main():
    global skip, last_json, current_level
    for step in range(num_steps):
        step_start_time = datetime.now()
        logging.info(f"Step {step} started at: {step_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        
        # Fetch current frames from cameras
        top_cam = cameras["top"].async_read()
        left_cam = cameras["left"].async_read()
        right_cam = cameras["right"].async_read()
        
        # Save images every 10 steps
        if step % 10 == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            # Save original images
            cv2.imwrite(f"{output_dir}/opencv__dev_video{step}_top.png", top_cam)
            cv2.imwrite(f"{output_dir}/opencv__dev_video{step}_left.png", left_cam)
            cv2.imwrite(f"{output_dir}/opencv__dev_video{step}_right.png", right_cam)
            
            logging.info(f"Saved images for step {step}")
                        
        # Resize images on the client side to minimize bandwidth / latency. Always return images in uint8 format.
        # We provide utilities for resizing images + uint8 conversion so you match the training routines.
        # The typical resize_size for pre-trained pi0 models is 224.
        # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.
        current_state = robot.get_joint_positions()
        current_state = np.array(current_state)

        # Log observation construction time
        observation_time = datetime.now()
        logging.info(f"Observation constructed at: {observation_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

        observation = {
            "observation.images.top": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(top_cam, 224, 224)
            ),
            "observation.images.left": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(left_cam, 224, 224)
            ),
            "observation.images.right": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(right_cam, 224, 224)
            ),
            "observation.state": current_state,
            "prompt": task_instruction,
        }

        print(current_state, "curr state")

        # Call the policy server with the current observation.
        # This returns an action chunk of shape (action_horizon, action_dim).
        # Note that you typically only need to call the policy every N steps and execute steps
        # from the predicted action chunk open-loop in the remaining steps.
        policy_start_time = datetime.now()
        logging.info(f"Policy inference started at: {policy_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        
        action_chunk = client.infer(observation)["actions"]
        
        policy_end_time = datetime.now()
        logging.info(f"Policy inference completed at: {policy_end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        
        action_chunk = action_chunk[:30]
        for action in action_chunk:
            # if action[1] < 0.03:
            #     continue
            robot.command_joint_pos(np.array(action[:14]))
            time.sleep(0.04)

        # Execute the actions in the environment.
        # Add your action execution logic here
        step_end_time = datetime.now()
        step_duration = (step_end_time - step_start_time).total_seconds()
        logging.info(f"Step {step} completed at: {step_end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} (Duration: {step_duration:.3f}s)")
        print(f"Step {step}: Got action chunk with shape {action_chunk.shape}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        print(f"Cleanup completed. Images saved to: {output_dir}")