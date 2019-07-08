import gym
import torch

from agents.c51 import C51Agent
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from common.utils import mini_batch_train_frames

env_id = "PongNoFrameskip-v4"
env    = make_atari(env_id)
env    = wrap_deepmind(env)
env    = wrap_pytorch(env)

MAX_FRAMES = 1000000
BATCH_SIZE = 32

agent = C51Agent(env, use_conv=False)
if torch.cuda.is_available():
    agent.model.cuda()

episode_rewards = mini_batch_train_frames(env, agent, MAX_FRAMES, BATCH_SIZE)
