import gym

from perDQN.per_dqn import PERAgent
from common.utils import mini_batch_train

env_id = "CartPole-v0"
MAX_EPISODES = 1000
MAX_STEPS = 500
BATCH_SIZE = 32

env = gym.make(env_id)
agent = PERAgent(env, use_conv=False)
episode_rewards = mini_batch_train(env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)
