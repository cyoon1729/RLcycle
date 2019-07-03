import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import math
import gym

# KL divergence of two univariate Gaussian distributions
def KL_divergence_mean_std(p_mean, p_std, q_mean, q_std):
    kld = torch.log(q_std/p_std) + (torch.pow(p_std) + torch.pow(p_mean - q_mean, 2))/(2 * torch.pow(q_std)) - 0.5
    return kld

# compute KL divergence of two distributions
def KL_divergence_two_dist(dist_p, dist_q):
    kld = torch.sum(dist_p * (torch.log(dist_p) - torch.log(dist_q)))
    return kld

# project value distribution onto atoms as in Categorical Algorithm
def dist_projection(optimal_dist, rewards, dones, gamma, n_atoms, Vmin, Vmax, support):
    batch_size = rewards.size(0)
    m = torch.zeros(batch_size, n_atoms)
    delta_z = (Vmax - Vmin) / (n_atoms - 1)

    for sample_idx in range(batch_size):
        reward = rewards[sample_idx]
        done = dones[sample_idx]

        for atom in range(n_atoms):
            # compute projection of Tz_j
            Tz_j = reward + (1 - done) * gamma * support[atom]
            Tz_j = torch.clamp(Tz_j, Vmin, Vmax)
            b_j = (Tz_j - Vmin) / delta_z
            l = torch.floor(b_j).long().item()
            u = torch.ceil(b_j).long().item()

            # distribute probability of Tz_j
            m[sample_idx][l] = m[sample_idx][l] + optimal_dist[sample_idx][atom] * (u - b_j)
            m[sample_idx][u] = m[sample_idx][u] + optimal_dist[sample_idx][atom] * (b_j - l)

    #print(m)
    return m

# run environment
def run_environment(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)

            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))

    return episode_rewards

# process episode rewards for multiple trials
def process_episode_rewards(many_episode_rewards):
    minimum = [np.min(episode_reward) for episode_reward in episode_rewards]
    maximum = [np.max(episode_reward) for episode_reward in episode_rewards]
    mean = [np.mean(episode_reward) for episode_reward in episode_rewards]

    return minimum, maximum, mean


class NoisyLinear(nn.Module):

    def __init__(self, num_in, num_out, is_training=True):
        super(NoisyLinear, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.is_training = is_training

        self.mu_weight = nn.Parameter(torch.FloatTensor(num_out, num_in))
        self.mu_bias = nn.Parameter(torch.FloatTensor(num_out))
        self.sigma_weight = nn.Parameter(torch.FloatTensor(num_out, num_in))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(num_out))
        self.register_buffer("epsilon_weight", torch.FloatTensor(num_out, num_in)) 
        self.register_buffer("epsilon_bias", torch.FloatTensor(num_out))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        self.reset_noise()

        if self.is_training:
            weight = self.mu_weight + self.sigma_weight.mul(autograd.Variable(self.epsilon_weight)) 
            bias = self.mu_bias + self.sigma_bias.mul(autograd.Variable(self.epsilon_bias))
        else:
            weight = self.mu_weight
            buas = self.mu_bias

        y = F.linear(x, weight, bias)
        
        return y

    def reset_parameters(self):
        std = math.sqrt(3 / self.num_in)
        self.mu_weight.data.uniform_(-std, std)
        self.mu_bias.data.uniform_(-std,std)

        self.sigma_weight.data.fill_(0.017)
        self.sigma_bias.data.fill_(0.017)

    def reset_noise(self):
        self.epsilon_weight.data.normal_()
        self.epsilon_bias.data.normal_()

    
class FactorizedNoisyLinear(nn.Module):

    def __init__(self, num_in, num_out, is_training=True):
        super(FactorizedNoisyLinear, self).__init__()
        self.num_in = num_in
        self.num_out = num_out 
        self.is_training = is_training

        self.mu_weight = nn.Parameter(torch.FloatTensor(num_out, num_in))
        self.mu_bias = nn.Parameter(torch.FloatTensor(num_out)) 
        self.sigma_weight = nn.Parameter(torch.FloatTensor(num_out, num_in))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(num_out))
        self.register_buffer("epsilon_i", torch.FloatTensor(num_in))
        self.register_buffer("epsilon_j", torch.FloatTensor(num_out))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        self.reset_noise()
        
        if self.is_training:
            epsilon_weight = self.epsilon_j.ger(self.epsilon_i)
            epsilon_bias = self.epsilon_j
            weight = self.mu_weight + self.sigma_weight.mul(autograd.Variable(epsilon_weight))
            bias = self.mu_bias + self.sigma_bias.mul(autograd.Variable(epsilon_bias))
        else:
            weight = self.mu_weight
            bias = self.mu_bias

        y = F.linear(x, weight, bias)
        
        return y

    def reset_parameters(self):
        std = 1 / math.sqrt(self.num_in)
        self.mu_weight.data.uniform_(-std, std)
        self.mu_bias.data.uniform_(-std, std)

        self.sigma_weight.data.fill_(0.5 / math.sqrt(self.num_in))
        self.sigma_bias.data.fill_(0.5 / math.sqrt(self.num_in))

    def reset_noise(self):
        eps_i = torch.randn(self.num_in)
        eps_j = torch.randn(self.num_out)
        self.epsilon_i = eps_i.sign() * (eps_i.abs()).sqrt()
        self.epsilon_j = eps_j.sign() * (eps_j.abs()).sqrt()