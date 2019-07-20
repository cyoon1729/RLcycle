import torch
import torch.nn.functional as F 
import torch.multiprocessing as mp
from torch.distributions import Categorical

from models import TwoHeadNetwork


class Worker(mp.Process):

    def __init__(self, id, env, gamma, global_network, global_optimizer, global_episode, GLOBAL_MAX_EPISODE):
        super(Worker, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = "w%i" % id
        
        self.env = env
        self.env.seed(id)
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.gamma = gamma
        self.local_network = TwoHeadNetwork(self.obs_dim, self.action_dim) 

        self.global_network = global_network
        self.global_episode = global_episode
        self.global_optimizer = global_optimizer
        self.GLOBAL_MAX_EPISODE = GLOBAL_MAX_EPISODE
    
    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        logits, _ = self.local_network.forward(state)
        dist = F.softmax(logits, dim=0)
        probs = Categorical(dist)

        return probs.sample().cpu().detach().item()
    
    def compute_loss(self, trajectory):
        states = torch.FloatTensor([sars[0] for sars in trajectory]).to(self.device)
        actions = torch.LongTensor([sars[1] for sars in trajectory]).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor([sars[2] for sars in trajectory]).to(self.device)
        next_states = torch.FloatTensor([sars[3] for sars in trajectory]).to(self.device)
        dones = torch.FloatTensor([sars[4] for sars in trajectory]).view(-1, 1).to(self.device)
        
        # compute discounted rewards
        discounted_rewards = [torch.sum(torch.FloatTensor([self.gamma**i for i in range(rewards[j:].size(0))])\
             * rewards[j:]) for j in range(rewards.size(0))]  # sorry, not the most readable code.
        
        logits, values = self.local_network.forward(states)
        dists = F.softmax(logits, dim=1)
        probs = Categorical(dists)
        
        # compute value loss
        value_targets = rewards.view(-1, 1) + torch.FloatTensor(discounted_rewards).view(-1, 1).to(self.device)
        value_loss = F.mse_loss(values, value_targets.detach())
        
        # compute entropy bonus
        entropy = []
        for dist in dists:
            entropy.append(-torch.sum(dist.mean() * torch.log(dist)))
        entropy = torch.stack(entropy).sum()
        
        # compute policy loss
        advantage = value_targets - values
        policy_loss = -probs.log_prob(actions.view(actions.size(0))).view(-1, 1) * advantage.detach()
        policy_loss = policy_loss.mean()
        
        total_loss = policy_loss + value_loss - 0.001 * entropy 
        return total_loss

    def update_global(self, trajectory):
        loss = self.compute_loss(trajectory)
        
        self.global_optimizer.zero_grad()
        loss.backward()
        # propagate local gradients to global parameters
        for local_params, global_params in zip(self.local_network.parameters(), self.global_network.parameters()):
            global_params._grad = local_params._grad
        self.global_optimizer.step()

    def sync_with_global(self):
        self.local_network.load_state_dict(self.global_network.state_dict())

    def run(self):
        state = self.env.reset()
        trajectory = [] # [[s, a, r, s', done], [], ...]
        episode_reward = 0
        
        while self.global_episode.value < self.GLOBAL_MAX_EPISODE:
            action = self.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            trajectory.append([state, action, reward, next_state, done])
            episode_reward += reward

            if done:
                with self.global_episode.get_lock():
                    self.global_episode.value += 1
                
                print(self.name + " | episode: "+ str(self.global_episode.value) + " " + str(episode_reward))

                self.update_global(trajectory)
                self.sync_with_global()

                trajectory = []
                episode_reward = 0
                state = self.env.reset()
            
            state = next_state