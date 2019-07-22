import sys
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import obstacle_env

# Constants
GAMMA = 0.99

class PolicyNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size=256):
        super(PolicyNetwork, self).__init__()
        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x 
    
class Agent:

    def __init__(self, env, learning_rate=3e-4):
        self.env = env
        self.num_actions = self.env.action_space.n
        self.policy_network = PolicyNetwork(self.env.observation_space.shape[0], self.env.action_space.n)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy_network.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob

    def update_policy(self, rewards, log_probs):
        discounted_rewards = []

        for t in range(len(rewards)):
            Gt = 0 
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + GAMMA**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)
            
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards

        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)
        
        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer.step()

    def train(self, max_episode=3000, max_step=200):
        for episode in range(max_episode):
            state = env.reset()
            log_probs = []
            rewards = []
            episode_reward = 0

            for steps in range(max_step):
                action, log_prob = self.get_action(state)
                new_state, reward, done, _ = self.env.step(action)
                
                log_probs.append(log_prob)
                rewards.append(reward)
                episode_reward += reward

                if done:
                    self.update_policy(rewards, log_probs)
                    if episode % 10 == 0:
                        print("episode " + str(episode) + ": " + str(episode_reward))

                    break
                
                state = new_state     
# def main():
#     env = gym.make('obstacle-v0')
#     policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n, 128)
    
#     max_episode_num = 5000
#     max_steps = 10000
#     numsteps = []
#     avg_numsteps = []
#     all_rewards = []

#     for episode in range(max_episode_num):
#         state = env.reset()
#         log_probs = []
#         rewards = []

#         for steps in range(max_steps):
#             env.render()
#             action, log_prob = policy_net.get_action(state)
#             new_state, reward, done, _ = env.step(action)
#             log_probs.append(log_prob)
#             rewards.append(reward)

#             if done:
#                 update_policy(policy_net, rewards, log_probs)
#                 numsteps.append(steps)
#                 avg_numsteps.append(np.mean(numsteps[-10:]))
#                 all_rewards.append(np.sum(rewards))
#                 if episode % 1 == 0:
#                     sys.stdout.write("episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(episode, np.round(np.sum(rewards), decimals = 3),  np.round(np.mean(all_rewards[-10:]), decimals = 3), steps))

#                 break
            
#             state = new_state
        
#     plt.plot(numsteps)
#     plt.plot(avg_numsteps)
#     plt.xlabel('Episode')
#     plt.show()
        
if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = Agent(env)
    agent.train(3000,200)


    
