import gym

from agents.dueling_ddqn import DuelingAgent
from common.utils import run_environment

env_id = "CartPole-v0"
MAX_EPISODES = 1000
MAX_STEPS = 500
BATCH_SIZE = 32

agent = DuelingAgent(env, use_conv=False)
episode_rewards = run_environment(env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)