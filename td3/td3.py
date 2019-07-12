import torch
import torch.nn as nn 
import torch.optim as optim

from td3.models import Critic, Actor
from common.replay_buffers import BasicBuffer


class TD3Agent:

    def __init__(self, env, gamma, tau, buffer_maxlen, delay_step, noise_std, noise_bound, critic_learning_rate, actor_learning_rate):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.delay_step = delay_step
        self.noise_std = noise_std
        self.noise_bound = noise_bound
        
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
	self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        self.critic1 = Critic(self.obs_dim, self.action_dim).to(self.device)
        self.critic2 = Critic(self.obs_dim, self.action_dim).to(self.device)
        self.critic1_target = Critic(self.obs_dim, self.action_dim).to(self.device)
        self.critic2_target = Critic(self.obs_dim, self.action_dim).to(self.device)
        
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.actor_target = Actor(self.obs_dim, self.action_dim).to(self.device)
    
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
        state = Variable(torch.FloatTensor(obs).unsqueeze(0))
        action = self.actor.forward(state)
        action = action.squeeze(0).cpu().detach().numpy()

        return action
    
    def update(self, batch_size, update_step):
        state_batch, action_batch, reward_batch, next_state_batch, masks = self.replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device)
        
        # update critics
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        
        curr_Q1 = self.critic1.forward(state_batch, action_batch)
        curr_Q2 = self.critic2.forward(state_batch, action_batch)
        

        action_space_noise = self.generate_action_space_noise(action_batch)
        next_actions = self.actor.forward(state_batch) + action_space_noise
        next_Q1 = self.critic1_target.forward(next_state_batch, next_actions)
        next_Q2 = self.critic2_target.forward(next_state_batch, next_actions)
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
            policy_gradient = -self.critic1(state_batch, self.actor(state_batch)).mean()
            policy_gradient.backward()
            self.actor_optimizer.step()

            # target networks
            self.update_targets()

    def generate_action_space_noise(self, action_batch):
        noise = torch.normal(torch.zeros(action_batch.size()), self.noise_std).clamp(-self.noise_bound, self.noise_bound).to(self.device)
        return noise

    def update_targets(self):
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
