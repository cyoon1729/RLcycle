import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
import torch.optim as optim

class Critic(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, init_w = 3e-3):
        super(Critic, self)__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

        self.linear3.weight.data_uniform_(-init_w, init_w)
        self.linear3.bias.data_uniform_(-init_w, init_w)
    
    def forward(self, state, action):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = Variable(torch.from_numpy(action).float().unsqueeze(0))
        
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

class Actor(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, learning_rate = 3e-4):
        super(Actor, self)__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
        self.linear3.weight.data_uniform_(-init_w, init_w)
        self.linear3.bias.data_uniform_(-init_w, init_w)
        
    def forward(self, x):
        x = Variable(torch.from_numpy(x).float().unsqueeze(0))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))

        return x

    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.forward(state).detach().cpu().numpy()[0, 0]
        