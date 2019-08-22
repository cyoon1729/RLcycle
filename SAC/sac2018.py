import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from models import ValueNetwork, SoftQNetwork, GaussianPolicy
from buffer import Buffer


class SACAgent:
  
    def __init__(self, env, gamma, tau, v_lr, q_lr, policy_lr, buffer_maxlen):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.env = env        
        self.action_range = [env.action_space.low, env.action_space.high]
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # hyperparameters
        self.gamma = gamma
        self.tau = tau       

        # initialize networks 
        self.value_net = ValueNetwork(self.obs_dim, 1).to(self.device)
        self.target_value_net = ValueNetwork(self.obs_dim, 1).to(self.device)
        self.q_net1 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.q_net2 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.policy_net = GaussianPolicy(self.obs_dim, self.action_dim).to(self.device)
        
        # copy params to target param
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param)
            
        # initialize optimizers 
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=v_lr)
        self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        self.replay_buffer = Buffer(buffer_maxlen)
        
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.policy_net.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        action = action.cpu().detach().squeeze(0).numpy()

        return action
    

    def update(self, batch_size):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        dones = dones.view(dones.size(0), -1)
        
        _, _, next_zs, next_log_pi = self.policy_net.sample(next_states)
        next_actions = torch.tanh(next_zs)
        next_q1 = self.q_net1(next_states, next_actions)
        next_q2 = self.q_net2(next_states, next_actions)
        next_v = self.target_value_net(next_states)
        
        # value Loss
        next_v_target = torch.min(next_q1, next_q2) - next_log_pi
        curr_v = self.value_net.forward(states)
        v_loss = F.mse_loss(curr_v, next_v_target.detach())
        
        # q loss
        curr_q1 = self.q_net1.forward(states, actions)
        curr_q2 = self.q_net2.forward(states, actions)
        expected_q = rewards + (1 - dones) * self.gamma * next_v
        q1_loss = F.mse_loss(curr_q1, expected_q.detach())
        q2_loss = F.mse_loss(curr_q2, expected_q.detach())
        
        # update value network and q networks
        self.value_optimizer.zero_grad()
        v_loss.backward()
        self.value_optimizer.step()
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        _, _, new_zs, log_pi = self.policy_net.sample(states)
        new_actions = torch.tanh(new_zs)
        min_q = torch.min(
            self.q_net1.forward(states, new_actions),
            self.q_net2.forward(states, new_actions)
        )
        policy_loss = (log_pi - min_q).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
    
        # target networks
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)