import gym
from a3c import A3CAgent


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    gamma = 0.99
    lr = 1e-3
    GLOBAL_MAX_EPISODE = 1000

    agent = A3CAgent(env, gamma, lr, GLOBAL_MAX_EPISODE)
    agent.train()
    agent.save_model()