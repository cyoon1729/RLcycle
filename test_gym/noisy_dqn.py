import gym

from agents.noisy_dqn import NoisyDQNAgent
from common.utils import run_environment

env_id = "CartPole-v0"
MAX_EPISODES = 1000
MAX_STEPS = 500
BATCH_SIZE = 32

agent = NoisyDQNAgent(env, use_conv=False)
episode_rewards = run_environment(env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)