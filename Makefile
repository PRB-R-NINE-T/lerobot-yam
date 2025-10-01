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

.PHONY: tests

PYTHON_PATH := $(shell which python3)

# If uv is installed and a virtual environment exists, use it
UV_CHECK := $(shell command -v uv)
ifneq ($(UV_CHECK),)
	VENV_PYTHON := $(shell .venv/bin/python 2>/dev/null)
	ifneq ($(VENV_PYTHON),)
		PYTHON_PATH := $(VENV_PYTHON)
	endif
endif

export PATH := $(dir $(PYTHON_PATH)):$(PATH)

DEVICE ?= cpu

build-user:
	docker build -f docker/Dockerfile.user -t lerobot-user .

build-internal:
	docker build -f docker/Dockerfile.internal -t lerobot-internal .

test-end-to-end:
	${MAKE} DEVICE=$(DEVICE) test-act-ete-train
	${MAKE} DEVICE=$(DEVICE) test-act-ete-train-resume
	${MAKE} DEVICE=$(DEVICE) test-act-ete-eval
	${MAKE} DEVICE=$(DEVICE) test-diffusion-ete-train
	${MAKE} DEVICE=$(DEVICE) test-diffusion-ete-eval
	${MAKE} DEVICE=$(DEVICE) test-tdmpc-ete-train
	${MAKE} DEVICE=$(DEVICE) test-tdmpc-ete-eval
	${MAKE} DEVICE=$(DEVICE) test-smolvla-ete-train
	${MAKE} DEVICE=$(DEVICE) test-smolvla-ete-eval

test-act-ete-train:
	$(PYTHON_PATH) -m lerobot.scripts.train \
		--policy.type=act \
		--policy.dim_model=64 \
		--policy.n_action_steps=20 \
		--policy.chunk_size=20 \
		--policy.device=$(DEVICE) \
		--policy.push_to_hub=false \
		--env.type=aloha \
		--env.episode_length=5 \
		--dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
		--dataset.image_transforms.enable=true \
		--dataset.episodes="[0]" \
		--batch_size=2 \
		--steps=4 \
		--eval_freq=2 \
		--eval.n_episodes=1 \
		--eval.batch_size=1 \
		--save_freq=2 \
		--save_checkpoint=true \
		--log_freq=1 \
		--wandb.enable=false \
		--output_dir=tests/outputs/act/

test-act-ete-train-resume:
	$(PYTHON_PATH) -m lerobot.scripts.train \
		--config_path=tests/outputs/act/checkpoints/000002/pretrained_model/train_config.json \
		--resume=true

test-act-ete-eval:
	$(PYTHON_PATH) -m lerobot.scripts.eval \
		--policy.path=tests/outputs/act/checkpoints/000004/pretrained_model \
		--policy.device=$(DEVICE) \
		--env.type=aloha \
		--env.episode_length=5 \
		--eval.n_episodes=1 \
		--eval.batch_size=1

test-diffusion-ete-train:
	$(PYTHON_PATH) -m lerobot.scripts.train \
		--policy.type=diffusion \
		--policy.down_dims='[64,128,256]' \
		--policy.diffusion_step_embed_dim=32 \
		--policy.num_inference_steps=10 \
		--policy.device=$(DEVICE) \
		--policy.push_to_hub=false \
		--env.type=pusht \
		--env.episode_length=5 \
		--dataset.repo_id=lerobot/pusht \
		--dataset.image_transforms.enable=true \
		--dataset.episodes="[0]" \
		--batch_size=2 \
		--steps=2 \
		--eval_freq=2 \
		--eval.n_episodes=1 \
		--eval.batch_size=1 \
		--save_checkpoint=true \
		--save_freq=2 \
		--log_freq=1 \
		--wandb.enable=false \
		--output_dir=tests/outputs/diffusion/

test-diffusion-ete-eval:
	$(PYTHON_PATH) -m lerobot.scripts.eval \
		--policy.path=tests/outputs/diffusion/checkpoints/000002/pretrained_model \
		--policy.device=$(DEVICE) \
		--env.type=pusht \
		--env.episode_length=5 \
		--eval.n_episodes=1 \
		--eval.batch_size=1

test-tdmpc-ete-train:
	$(PYTHON_PATH) -m lerobot.scripts.train \
		--policy.type=tdmpc \
		--policy.device=$(DEVICE) \
		--policy.push_to_hub=false \
		--env.type=xarm \
		--env.task=XarmLift-v0 \
		--env.episode_length=5 \
		--dataset.repo_id=lerobot/xarm_lift_medium \
		--dataset.image_transforms.enable=true \
		--dataset.episodes="[0]" \
		--batch_size=2 \
		--steps=2 \
		--eval_freq=2 \
		--eval.n_episodes=1 \
		--eval.batch_size=1 \
		--save_checkpoint=true \
		--save_freq=2 \
		--log_freq=1 \
		--wandb.enable=false \
		--output_dir=tests/outputs/tdmpc/

test-tdmpc-ete-eval:
	$(PYTHON_PATH) -m lerobot.scripts.eval \
		--policy.path=tests/outputs/tdmpc/checkpoints/000002/pretrained_model \
		--policy.device=$(DEVICE) \
		--env.type=xarm \
		--env.episode_length=5 \
		--env.task=XarmLift-v0 \
		--eval.n_episodes=1 \
		--eval.batch_size=1


test-smolvla-ete-train:
	$(PYTHON_PATH) -m lerobot.scripts.train \
		--policy.type=smolvla \
		--policy.n_action_steps=20 \
		--policy.chunk_size=20 \
		--policy.device=$(DEVICE) \
		--policy.push_to_hub=false \
		--env.type=aloha \
		--env.episode_length=5 \
		--dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
		--dataset.image_transforms.enable=true \
		--dataset.episodes="[0]" \
		--batch_size=2 \
		--steps=4 \
		--eval_freq=2 \
		--eval.n_episodes=1 \
		--eval.batch_size=1 \
		--save_freq=2 \
		--save_checkpoint=true \
		--log_freq=1 \
		--wandb.enable=false \
		--output_dir=tests/outputs/smolvla/

test-smolvla-ete-eval:
	$(PYTHON_PATH) -m lerobot.scripts.eval \
		--policy.path=tests/outputs/smolvla/checkpoints/000004/pretrained_model \
		--policy.device=$(DEVICE) \
		--env.type=aloha \
		--env.episode_length=5 \
		--eval.n_episodes=1 \
		--eval.batch_size=1

teleop:
	$(PYTHON_PATH) -m lerobot.teleoperate \
		--robot.type=yam_follower \
		--robot.port=can_leader_l \
		--robot.id=follower \
		--teleop.type=yam_leader \
		--teleop.port=can_follower_l \
		--teleop.id=leader \
		--display_data=true

record-data:
	rm -rf /home/p/.cache/huggingface/lerobot/pierre-safe-sentinels-inc/yam-pi-zero-five-test
	$(PYTHON_PATH) -m lerobot.record \
		--robot.type=yam_follower \
		--robot.port=can_leader_l \
		--robot.id=follower \
		--teleop.type=yam_leader \
		--teleop.port=can_follower_l \
		--teleop.id=leader \
		--display_data=true \
		--robot.cameras="{}" \
		--dataset.repo_id=pierre-safe-sentinels-inc/yam-pi-zero-five-test \
		--dataset.num_episodes=2 \
		--dataset.single_task="Grab the black cube"

token:
	hf-token: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

lr_teleop:
	$(PYTHON_PATH) -m lerobot.teleoperate --robot.type=yam_bimanual --robot.port=can0 --robot.port_right=can1 --robot.id=follower --teleop.type=gello_leader --teleop.id=leader --teleop.port=/dev/ttyUSB0 --teleop.port_right=/dev/ttyUSB1

record:
	@if ! ip link show can_left | grep -q "state UP"; then \
		echo "Setting up can_left interface..."; \
		sudo ip link set can_left up type can bitrate 1000000; \
	else \
		echo "can_left interface is already up"; \
	fi
	@if ! ip link show can_right | grep -q "state UP"; then \
		echo "Setting up can_right interface..."; \
		sudo ip link set can_right up type can bitrate 1000000; \
	else \
		echo "can_right interface is already up"; \
	fi
	. .venv/bin/activate && \
	LOGLEVEL=INFO PYTHONPATH=/home/p/Desktop/lerobot-yam:$$PYTHONPATH lerobot-record --robot.type=yam_bimanual --robot.port=can_left \
	--robot.port_right=can_right --robot.id=follower --teleop.type=gello_leader \
	--teleop.id=leader --teleop.port=/dev/ttyUSB2 --teleop.port_right=/dev/ttyUSB1 \
	--dataset.repo_id=pierre-safe-sentinels-inc/yam-fold-napkins \
	--dataset.num_episodes=1 --dataset.single_task="fold napkins" --dataset.episode_time_s=150 \
	--robot.cameras="{ top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, left :{type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, right: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}}" \
	--dataset.reset_time_s=20 --dataset.video_encoding_batch_size=1  --resume=true

record_second:
	@if ! ip link show can_left | grep -q "state UP"; then \
		echo "Setting up can_left interface..."; \
		sudo ip link set can_left up type can bitrate 1000000; \
	else \
		echo "can_left interface is already up"; \
	fi
	@if ! ip link show can_right | grep -q "state UP"; then \
		echo "Setting up can_right interface..."; \
		sudo ip link set can_right up type can bitrate 1000000; \
	else \
		echo "can_right interface is already up"; \
	fi
	. .venv/bin/activate && \
	LOGLEVEL=INFO PYTHONPATH=/home/p/Desktop/lerobot-yam:$$PYTHONPATH lerobot-record --robot.type=yam_bimanual --robot.port=can_left \
	--robot.port_right=can_right --robot.id=follower --teleop.type=gello_leader \
	--teleop.id=leader --teleop.port=/dev/ttyUSB2 --teleop.port_right=/dev/ttyUSB1 \
	--dataset.repo_id=pierre-safe-sentinels-inc/yam-second-run \
	--dataset.num_episodes=5 --dataset.single_task="fold napkins" --dataset.episode_time_s=150 \
	--robot.cameras="{ top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, right :{type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, left: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}}" \
	--dataset.reset_time_s=20 --dataset.video=False --resume=true

find_cameras:
	. .venv/bin/activate && \
	PYTHONPATH=/home/p/Desktop/lerobot-yam:$$PYTHONPATH lerobot-find-cameras opencv

hf_token:
	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