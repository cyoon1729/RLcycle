from .networks import softQ, policy, value
from .utils import NormalizedActions, ReplayBuffer

import sys
import numpy as np
import torch  
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
import matplotlib.pyplot as plt

class SoftActorCritic:

    def __init__(self, config, env):
        # super(SoftActorCritic, self).__init__(env, batch_size, gama, mean_lambda, std_lambda, z_lambda, soft_tau):

        self.batch_size = config.batch_size
        self.gamma = config.gamma
        self.mean_lambda = config.mean_lambda
        self.std_lambda = config.std_lambda
        self.z_lambda = config.z_lambda
        self.soft_tau = config.soft_tau

        # CUDA availability:
        # use_cuda = torch.cuda.is_available()
        # self.device = torch.device("cuda" if use_cuda else "cpu")
        self.device = torch.device("cpu")
        
        # Neural Network Parameters
        self.env = env
        self.action_dim = env.action_dim
        self.state_dim = env.state_dim
        hidden_dim = 256

        # Network initialization
        self.value_net = value.ValueNetwork(env.state_dim, hidden_dim).to(self.device)
        self.target_value_net = value.ValueNetwork(env.state_dim, hidden_dim).to(self.device)
        self.soft_q_net = softQ.SoftQNetwork(env.state_dim, env.action_dim, hidden_dim).to(self.device)
        self.policy_net = policy.PolicyNetwork(self.device, env.state_dim, env.action_dim, hidden_dim).to(self.device)

        # Optimizers
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)
    

        self.value_criterion  = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()

        value_lr  = 3e-4
        soft_q_lr = 3e-4
        policy_lr = 3e-4

        self.value_optimizer  = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        # Replay buffer
        replay_buffer_size = 1000000
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.batch_size  = 128

    def soft_q_update(self):

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        expected_q_value = self.soft_q_net(state, action)
        expected_value   = self.value_net(state)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)


        target_value = self.target_value_net(next_state)
        next_q_value = reward + (1 - done) * self.gamma * target_value
        q_value_loss = self.soft_q_criterion(expected_q_value, next_q_value.detach())

        expected_new_q_value = self.soft_q_net(state, new_action)
        next_value = expected_new_q_value - log_prob
        value_loss = self.value_criterion(expected_value, next_value.detach())

        log_prob_target = expected_new_q_value - expected_value
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
        

        mean_loss = self.mean_lambda * mean.pow(2).mean()
        std_loss  = self.std_lambda  * log_std.pow(2).mean()
        z_loss    = self.z_lambda    * z.pow(2).sum(1).mean()

        policy_loss += mean_loss + std_loss + z_loss

        self.soft_q_optimizer.zero_grad()
        q_value_loss.backward()
        self.soft_q_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )
        
    def try_to_solve_mdp(self):
        
        max_episode = 750
        max_steps = 200 #cancer =  8
        max_num_updates = 40000
        num_updates = 0
        rewards = []
        rewards_plt = []
        avg_episode_reward_plt = []
        episode_index = 0

        best_traj = [[],[]]
        best_action = []
        
        while episode_index < max_episode and num_updates < max_num_updates :
            #state, state_copy = self.env.reset()
            state = self.env.reset()
            episode_reward = 0

            for step in range(max_steps):

                best_traj[0].append(state[0])
                best_traj[1].append(state[1])
                

                #action = self.policy_net.get_action(state_copy)
                action = self.policy_net.get_action(state)
                #action[0] = action[0] * self.env.action_bound
                action = action * self.env.action_bound
                action = np.clip(action, self.env.action_space_low, self.env.action_space_high)
                best_action.append(action)
                next_state, reward, done = self.env.step(state, action)
                #next_state, next_state_copy, reward, done = self.env.step(state, action)
                
                # next_state, reward, done, _ = self.env.step(action)

                #self.replay_buffer.push(state_copy, action, reward, next_state_copy, done)
                self.replay_buffer.push(state, action, reward, next_state, done)
                if len(self.replay_buffer) > self.batch_size: # and episode_index < 500:
                    self.soft_q_update()
                
                state = next_state
                #state_copy = next_state_copy
                episode_reward = episode_reward + reward
                num_updates = num_updates + 1


                if done:
                    break
            
            rewards.append(episode_reward)
            rewards_plt.append(episode_reward)
            avg_episode_reward_plt.append(np.mean(rewards[-10:]))
            
            if episode_reward < 150:
                best_traj = [[],[]]
                best_action = []
            #     print(state)
            # if episode_reward > 150:
            #     print(best_traj)
            #     print(best_action)
            #     break
  
            sys.stdout.write("episode: {}, total numsteps: {}, reward: {}, average reward: {}\n".format(episode_index, num_updates, rewards[-1], np.mean(rewards[-10:])))
            episode_index = episode_index + 1

        # print(best_traj)
        # print(best_action)
        # plt.plot(rewards_plt)
        # plt.ylabel('Episode Reward')
        # plt.xlabel('Episode')
        # plt.show()

        # plt.plot(avg_episode_reward_plt)
        # plt.ylabel('Avg Episod Reward')
        # plt.xlabel('Episode')
        # plt.show()
        
        # plt.plot(best_traj[0])
        # plt.plot(best_traj[1])
        # plt.ylabel('State')
        # plt.xlabel('Time (months)')
        # plt.show() 
        
        # plt.plot(best_action)
        # plt.ylabel('Dose Level')
        # plt.xlabel('Time (months)')
        # plt.show()