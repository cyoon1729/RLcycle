import gym
from a3c import A3CAgent, DecoupledA3CAgent

def train_twoHeaded():
    agent = A3CAgent(env, gamma, lr, GLOBAL_MAX_EPISODE)
    agent.train()
    agent.save_model()

def train_decoupled():
    agent = DecoupledA3CAgent(env, gamma, lr, GLOBAL_MAX_EPISODE)
    agent.train()
    agent.save_model()


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    gamma = 0.99
    lr = 1e-3
    GLOBAL_MAX_EPISODE = 1000

    #train_twoHeaded()
    train_decoupled()
