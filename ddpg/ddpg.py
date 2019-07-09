import torch
import torch.optim as optim
import torch.autograd as autograd


class DDPGAgent:
    
    def __init__(self, env, gamma, tau, buffer_maxlen, critic_learning_rate, actor_learning_rate):
        self.env = env
        self.noise = OUNoise(self.env.action_space)
        
        self.env = env
        self.gamma = gamma
        self.tau = tau
        
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        self.device = "cpu"
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.device = "cuda"
            
        self.critic = Critic(self.obs_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.obs_dim, self.action_dim).to(self.device)
        
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.actor_target = Actor(self.obs_dim, self.action_dim).to(self.device)
    
        # Copy critic target parameters
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        self.replay_buffer = BasicBuffer(buffer_maxlen)        
        self.MSELoss  = nn.MSELoss()
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
    
    def get_action(self, obs):
        state = autograd.Variable(torch.FloatTensor(obs).unsqueeze(0)).to(self.device)
        action = self.actor.forward(state)
        action = action.squeeze(0).cpu().detach().numpy()

        return action
    
    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.replay_buffer.sample(batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, masks = self.replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device)
   
        # update critic
        self.critic_optimizer.zero_grad()
        
        curr_Q = self.critic.forward(state_batch, action_batch)
        next_actions = self.actor_target.forward(next_state_batch)
        next_Q = self.critic_target.forward(next_state_batch, next_actions.detach())
        expected_Q = reward_batch + self.gamma * next_Q
        critic_loss = self.MSELoss(curr_Q, expected_Q.detach())
        critic_loss.backward() 
        self.critic_optimizer.step()

        # update actor
        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic.forward(state_batch, self.actor.forward(state_batch)).mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))