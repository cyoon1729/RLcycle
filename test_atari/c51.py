import gym

from agents.c51 import C51Agent
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from common.utils import run_environment

env_id = "PongNoFrameskip-v4"
env    = make_atari(env_id)
env    = wrap_deepmind(env)
env    = wrap_pytorch(env)

MAX_EPISODES = 1000
MAX_STEPS = 500
BATCH_SIZE = 32

agent = C51Agent(env, use_conv=False)
run_environment(env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)