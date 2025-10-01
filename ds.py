from lerobot.datasets.lerobot_dataset import LeRobotDataset

eps = [i for i in range(64)]
eps.remove(40)
ds = LeRobotDataset("pierre-safe-sentinels-inc/yam-second-run-bck", episodes=eps)

ds.batch_encode_videos(0, 63)