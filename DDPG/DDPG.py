import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from model import *
from utils import *

class DDPGagent:
    def __init__(self, env, hidden_size=128, learning_rate=1e-4, gamma=0.99, tau=1e-3):
        # Params
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau

        # Networks
        self.actor = Actor(self.num_states, hidden_size, self.num_actions)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Noise
        self.noise = OUNoise(env.action_space)
        self.noise.reset()

        # Training
        #self.memory = ReplayBuffer(128)        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
    
    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor.forward(state)
        action = action.detach().numpy()[0,0]
        #return action
        return self.noise.get_action(action)

    def get_target_action(self, state):  # sample action from target policy network
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor_target.forward(state)
        action = action.detach().numpy()[0,0]
        #return action
        return self.noise.get_action(action)
    
    """
    def update():
        state, action, reward, next_state = experience

        # Critic update
        Q = self.critic.forward(state, action)
        next_action = self.get_target_action(next_state)
        next_Q = self.critic_target.forward(next_state, next_action)
        Qprime = reward + learning_rate * next_Q # Bellman Update
        critic_loss = self.critc_criterion(Qprime, Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        # Actor update
        policy_loss = -(self.critic.forward(state, self.actor_forward(state)))
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


    def update():
        state, action, reward, next_state = self.replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)
        next_state = torch.FloatTensor(next_state)

        Q_values = self.critic.forward(state, action)
        next_actions = self.get_target_action(next_state)
        next_Q = self.criic_targe.forward(next_state, next_action)
        Qprime = reward + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qprime, Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        policy_loss = -(self.critic.forward(state, self.actor_forward(state)))
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    """



