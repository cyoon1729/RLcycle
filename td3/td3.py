import torch
import torch.nn as nn 
import torch.optim as optim
from .model import Critic, Actor
from common.replay_buffers import BasicBuffer


# Single Agent TD3
class TD3Agent:

    def __init__(self, env, gamma, tau, buffer_maxlen, delay_step):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.deplay_step = deplay_step
        
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        self.critic1 = Critic(self.obs_dim, self.action_dim)
        self.critic2 = Critic(self.obs_dim, self.action_dim)
        self.critic1_target = Critic(self.obs_dim, self.action_dim)
        self.critic2_target = Critic(self.obs_dim, self.action_dim)
        
        self.actor = Actor(self.obs_dim, self.action_dim)
        self.actor_target = Actor(self.obs_dim, self.action_dim)
    
        # Copy critic target parameters
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data)
        
        self.replay_buffer = BasicBuffer(buffer_maxlen)        
        self.MSELoss  = nn.MSELoss()
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_learning_rate) 
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
    
    def get_action(self, obs):
        state = torch.FloatTensor(obs).unsqueeze(0)
        action = self.actor.forward(state)
        action = torch.argmax(action).item()

        return action
    
    def update(self, batch_size, update_step):
        state_batch, action_batch, reward_batch, next_state_batch, masks = self.replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.FloatTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        masks = torch.FloatTensor(masks)
        
        # update critics
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        
        curr_Q1 = self.critic1.forward(state_batch, action_batch)
        curr_Q2 = self.critic2.forward(state_batch, action_batch)
        

        action_space_noise = self.generate_action_space_noise(action_batch)
        next_actions = self.actor.forward(state_batch) + action_space_noise)
        next_Q1 = self.critic1.forward(next_state_batch, next_actions)
        next_Q2 = self.critic2.forward(next_state_batch, next_actions)
        expected_Q = reward_batch + self.gamma * torch.min(next_Q1, next_Q2)

        critic1_loss = self.MSELoss(curr_Q1, expected_Q.detach())
        critic1_loss.backward()
        self.critic1_optimizer.step()

        critic2_loss = self.MSELoss(curr_Q2, expected_Q.detach())
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # update actor & target networks  
        if(update_step % delay_step == 0):
            # actor
            self.actor_optimizer.zero_grad()
            policy_gradient = -self.critic1(state_batch, self.actor(state_batch)).sum().mean()
            policy_gradient.backward()
            self.actor_optimizer.step()

            # target networks
            self.update_targets()

    def generate_action_space_noise(self, action_batch, bound):
        noise = torch.randn(action_batch.size(0), self.env.action_space.n).clamp(-bound, bound)
        return noise

    def update_targets(self):
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))