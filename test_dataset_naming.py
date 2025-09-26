#!/usr/bin/env python3

"""
Test script to demonstrate the robot_id + day_of_year dataset naming functionality.
"""

from datetime import datetime


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


def test_dataset_naming():
    """Test the dataset naming function with different robot IDs."""
    
    # Test cases
    test_robots = [
        "follower_arm",
        "leader_arm", 
        "so100_robot",
        "bi_manipulator",
        "test_bot_123"
    ]
    
    print("Dataset Naming Test")
    print("=" * 50)
    print(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Day of year: {datetime.now().timetuple().tm_yday}")
    print()
    
    for robot_id in test_robots:
        repo_id = generate_dataset_repo_id(robot_id)
        print(f"Robot ID: '{robot_id}' -> Dataset repo_id: '{repo_id}'")
    
    print()
    print("Custom username example:")
    custom_repo_id = generate_dataset_repo_id("follower_arm", "my_username")
    print(f"Robot ID: 'follower_arm' with username 'my_username' -> '{custom_repo_id}'")
    
    print("\nDataset naming format: {hf_username}/{robot_id}_{day_of_year}")
    print("Where day_of_year is a single integer from 1-366")


if __name__ == "__main__":
    test_dataset_naming()
