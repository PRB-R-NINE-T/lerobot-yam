import os
import shutil
from threading import Thread
import time
from datetime import datetime
import requests
from sympy import continued_fraction
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig
from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.teleoperators.so100_leader.so100_leader import SO100Leader
from lerobot.utils.control_utils import init_keyboard_listener, sanity_check_dataset_name, sanity_check_dataset_robot_compatibility
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun
from lerobot.record import record_loop

# Constants
FPS = 30
NUM_EPISODES = 50
EPISODE_TIME_SEC = 30
RESET_TIME_SEC = 10


def generate_dataset_repo_id(robot_id: str, hf_username: str = "robot_user") -> str:
    """Generate a dataset repository ID based on robot ID and day of year.
    
    Args:
        robot_id: The robot identifier (e.g. "follower_arm")
        hf_username: The HuggingFace username to use for the repo
        
    Returns:
        A repo_id in the format "{hf_username}/{robot_id}_{day_of_year}"
        where day_of_year is a single integer (1-366)
    """
    # Get current day of year as a single integer
    day_of_year = datetime.now().timetuple().tm_yday
    
    # Create dataset name: robotid_dayofyear
    dataset_name = f"{robot_id}_{day_of_year}"
    
    # Return full repo_id
    return f"{hf_username}/{dataset_name}"


def read_robot_mode(status_file_path="STATUS.txt"):
    """Read the robot mode and task from STATUS.txt file.
    
    Expected format: mode task (two columns separated by space)
    
    Returns:
        list: [mode, task] where mode is one of ('Teleop', 'Autonomous', 'idle')
    """
    try:
        if os.path.exists(status_file_path):
            with open(status_file_path, 'r') as f:
                line = f.read().strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        mode, task = parts[0], parts[1]
                        if mode in ['Teleop', 'Autonomous', 'idle']:
                            return [mode, task]
                        else:
                            log_say(f"Invalid mode '{mode}' in STATUS.txt, defaulting to ['idle', 'default']")
                            return ['idle', 'default']
                    elif len(parts) == 1:
                        mode = parts[0]
                        if mode in ['Teleop', 'Autonomous', 'idle']:
                            log_say(f"Only mode '{mode}' found, using default task")
                            return [mode, 'default']
                        else:
                            log_say(f"Invalid mode '{mode}' in STATUS.txt, defaulting to ['idle', 'default']")
                            return ['idle', 'default']
                    else:
                        log_say("Empty STATUS.txt file, defaulting to ['idle', 'default']")
                        return ['idle', 'default']
                else:
                    log_say("Empty STATUS.txt file, defaulting to ['idle', 'default']")
                    return ['idle', 'default']
        else:
            log_say("STATUS.txt not found, defaulting to ['idle', 'default']")
            return ['idle', 'default']
    except Exception as e:
        log_say(f"Error reading STATUS.txt: {e}, defaulting to ['idle', 'default']")
        return ['idle', 'default']


def autonomous_mode_logic(robot, teleop, dataset, fps, episode_time_sec, task_description):
    """Placeholder for autonomous mode logic.
    
    This function will be filled with autonomous behavior later.
    """
    log_say(f"Autonomous mode activated - placeholder logic for task: {task_description}")
    # TODO: Implement autonomous behavior here
    pass


# Create the robot and teleoperator configurations
# camera_config = {"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS)}
camera_config = {}
robot_config = SO100FollowerConfig(
    port="/dev/ttyACM0", id="follower_arm", cameras=camera_config
)
teleop_config = SO100LeaderConfig(port="/dev/ttyACM1", id="leader_arm")

# Initialize the robot and teleoperator
robot = SO100Follower(robot_config)
teleop = SO100Leader(teleop_config)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}



# Generate dataset repo_id based on robot ID and day of year
dataset_repo_id = generate_dataset_repo_id(robot.id, hf_username="pierre8181")
log_say(f"Creating dataset with repo_id: {dataset_repo_id}")
print(f"Creating dataset with repo_id: {dataset_repo_id}")

# Check if the huggingface cache directory exists
cache_path = os.path.expanduser(f"~/.cache/huggingface/lerobot/{dataset_repo_id}")
cache_exists = os.path.exists(cache_path)
log_say(f"HuggingFace cache path exists: {cache_exists} (path: {cache_path})")
print(f"HuggingFace cache path exists: {cache_exists} (path: {cache_path})")

# Create the dataset
if cache_exists:
    dataset = LeRobotDataset(
        dataset_repo_id,
    )

    if hasattr(robot, "cameras") and len(robot.cameras) > 0:
        dataset.start_image_writer(
            num_processes=0,
            num_threads=12,
        )
    sanity_check_dataset_robot_compatibility(dataset, robot, 30, dataset_features)
else:
    # Try to download the dummy dataset from HF Hub if current dataset doesn't exist locally
    dummy_repo_id = "pierre818191/lerobot_ds_format_dummy"
    dummy_cache_path = os.path.expanduser(f"~/.cache/huggingface/lerobot/{dummy_repo_id}")
    dummy_cache_exists = os.path.exists(dummy_cache_path)
    
    if not dummy_cache_exists:
        log_say(f"Dataset {dataset_repo_id} not found locally. Downloading dummy dataset {dummy_repo_id} from HF Hub...")
        print(f"Dataset {dataset_repo_id} not found locally. Downloading dummy dataset {dummy_repo_id} from HF Hub...")
        
        dummy_dataset = LeRobotDataset(dummy_repo_id)
        log_say(f"Successfully downloaded dummy dataset {dummy_repo_id}")
        print(f"Successfully downloaded dummy dataset {dummy_repo_id}")

    # Now copy the dummy dataset to the target dataset name
    if os.path.exists(dummy_cache_path):
        log_say(f"Copying dummy dataset to target dataset name: {dataset_repo_id}")
        print(f"Copying dummy dataset to target dataset name: {dataset_repo_id}")
        
        try:
            # Copy the dummy dataset to the target dataset path
            shutil.copytree(dummy_cache_path, cache_path)
            
            # Load the dataset with the target repo_id
            dataset = LeRobotDataset(
                dataset_repo_id,
            )
            
            if hasattr(robot, "cameras") and len(robot.cameras) > 0:
                dataset.start_image_writer(
                    num_processes=0,
                    num_threads=12,
                )
            log_say(f"Successfully loaded dummy dataset as {dataset_repo_id}")
            print(f"Successfully loaded dummy dataset as {dataset_repo_id}")
            
        except Exception as e:
            log_say(f"Failed to copy dummy dataset: {e}. Creating empty dataset instead.")
            print(f"Failed to copy dummy dataset: {e}. Creating empty dataset instead.")
            # Fall back to creating empty dataset
            dataset = LeRobotDataset.create(
                dataset_repo_id,
                30,
                robot_type=robot.name,
                features=dataset_features,
                image_writer_processes=0,
                image_writer_threads=12,
            )
    else:
        # Create empty dataset or load existing saved episodes
        dataset = LeRobotDataset.create(
            dataset_repo_id,
            30,
            robot_type=robot.name,
            features=dataset_features,
            image_writer_processes=0,
            image_writer_threads=12,
        )

# Initialize the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
_init_rerun(session_name="recording")

# Connect the robot and teleoperator
robot.connect()
teleop.connect()

episode_idx = 0
log_say("Starting robot control system with mode switching")

class Robot:
    def __init__(self, robot, teleop, dataset, base_url="http://localhost:8000"):
        self.robot = robot
        self.teleop = teleop
        self.dataset = dataset
        self.thread = None
        self.current_mode = ""
        self.base_url = os.environ.get("SERVER_BASE_URL", base_url)

    def start_thread(self):
        self.thread = Thread(target=self.update_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def update_loop(self):
        while True:
            print("Updating robot mode")
            try:
                # Construct the URL for the robot status endpoint
                url = f"{self.base_url}/robot/status/{self.robot.id}"
                
                # Make the HTTP GET request to fetch robot status
                response = requests.get(url, timeout=5)
                response.raise_for_status()  # Raises HTTPError for bad responses
                
                # Parse the JSON response
                status_data = response.json()
                
                # Extract and save the status to current_mode
                if 'status' in status_data:
                    self.current_mode = status_data['status']
                    log_say(f"Updated robot mode from remote database: {self.current_mode}")
                else:
                    log_say("Warning: 'status' field not found in response")
                    
            except requests.exceptions.RequestException as e:
                log_say(f"Error fetching robot status from database: {e}")
                # Keep current mode unchanged on error
                
            except ValueError as e:
                log_say(f"Error parsing JSON response: {e}")
                # Keep current mode unchanged on error
                
            except Exception as e:
                log_say(f"Unexpected error in update_loop: {e}")
                # Keep current mode unchanged on error
            
            # Wait before next update (adjust frequency as needed)
            time.sleep(0.5)

    def operation_loop(self):
        while True:
            # Read current mode and task from STATUS.txt
            mode_and_task = read_robot_mode()
            current_mode, current_task = mode_and_task[0], mode_and_task[1]
            log_say(f"Current mode: {current_mode}, Current task: {current_task}")
            
            if current_mode == "Teleop":
                # Execute original teleoperation logic
                print("Current mode is teleop")
                while self.current_mode == "Teleop":
                    start_loop_t = time.perf_counter()

                    action = teleop.get_action()
                    observation = robot.get_observation()
                    sent_action = robot.send_action(action)

                    observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")
                    action_frame = build_dataset_frame(dataset.features, action, prefix="action")

                    frame = {**observation_frame, **action_frame}
                    dataset.add_frame(frame, task=current_task)

                    dt_s = time.perf_counter() - start_loop_t
                    busy_wait(1 / 30 - dt_s)
                
                print("Current mode is not teleop anymore")
                    
            elif current_mode == "Autonomous":
                print("Current mode is autonomous")
                while self.current_mode == "Autonomous":
                    current_ip = ""
                    # Execute autonomous mode logic (placeholder for now)
                    print("Executing autonomous mode logic")
                    continue
                    autonomous_mode_logic(robot, teleop, dataset, FPS, EPISODE_TIME_SEC, current_task)
                    # Sleep briefly before checking mode again
                    # time.sleep(1)
                
            elif current_mode == "idle":
                # Idle mode - keep iterating and checking for mode changes
                log_say("Robot in idle mode - waiting for mode change")
                print("Robot in idle mode - waiting for mode change")
                time.sleep(1)  # Check every second

if __name__ == "__main__":
    robot = Robot(robot, teleop, dataset)
    robot.start_thread()
    robot.operation_loop()