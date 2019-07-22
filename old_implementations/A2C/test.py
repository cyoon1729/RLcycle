import gym
from agent import Agent


env_id = "CartPole-v0"
env = gym.make(env_id)
agent = Agent(env)
agent.train(1500, 300)