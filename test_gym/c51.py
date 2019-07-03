import gym

from agents.c51 import C51Agent
from common.utils import run_environment

env_id = "CartPole-v0"
MAX_EPISODES = 1000
MAX_STEPS = 500
BATCH_SIZE = 32

agent = C51Agent(env, use_conv=False)
episode_rewards = run_environment(env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)