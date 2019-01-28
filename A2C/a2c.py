import sys
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# neural network parameters
hidden_size = 256
lr = 3e-4
num_steps = 5

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x 
    
    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_size, learning_rate=3e-4):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x 
    
def a2c(env):
    num_inputs = envs.observation_space.shape[0]
    num_outputs = envs.action_space.n
    
    policy_network = PolicyNetwork(num_inputss, num_outputs, hidden_size)
    value_network = ValueNetwork(num_inputs, hidden)

    state = env.reset()
    for episode in range(20000):
        log_probs = []
        values = []
        rewards = []

        action, log_prob = policy_network.get_action(state)
        value = value_network.forward(state)
        new_state, reward, done, _ = env.step(action)

        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value)

        state = new_state
        




    
    