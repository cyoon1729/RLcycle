import torch
import torch.optim as optim
import torch.multiprocessing as mp  
import gym

from models import TwoHeadNetwork
from worker import Worker


class A3CAgent:
    
    def __init__(self, env, gamma, lr, global_max_episode):
        self.env = env

        self.gamma = gamma
        self.lr = lr
        self.global_episode = mp.Value('i', 0)
        self.GLOBAL_MAX_EPISODE = global_max_episode

        self.global_network = TwoHeadNetwork(self.env.observation_space.shape[0], self.env.action_space.n)
        self.global_optimizer = optim.Adam(self.global_network.parameters(), lr=lr) 
        self.workers = [Worker(i, env, self.gamma, self.global_network, self.global_optimizer, self.global_episode, self.GLOBAL_MAX_EPISODE) for i in range(mp.cpu_count())]
    
    def train(self):
        print("Training on {} cores".format(mp.cpu_count()))
        input("Enter to start")

        [worker.start() for worker in self.workers]
        [worker.join() for worker in self.workers]
    
    def save_model(self):
        torch.save(self.global_network.state_dict(), "a3c_model.pth")


