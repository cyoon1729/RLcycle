from agent import Agent
import gym

env_id = "CartPole-v0"
env = gym.make(env_id)
agent = Agent(env)

if __name__ == "__main__":
    agent.train(5000, 200, 32)