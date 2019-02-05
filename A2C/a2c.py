import sys
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# hyperparameters
hidden_size = 256
learning_rate = 3e-4

# Constants
GAMMA = 0.9
num_steps = 1000
max_episodes = 2500

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)
    
    def forward(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)
        
        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(policy_dist.detach().numpy()))
        log_prob = torch.log(policy_dist.squeeze(0)[highest_prob_action])

        return value, highest_prob_action, log_prob

def a2c(env):
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n
    
    actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    all_lengths = []
    average_lengths = []
    
    for episode in range(max_episodes):
        log_probs = []
        values = []
        rewards = []

        state = env.reset()
        for steps in range(num_steps):
            value, action, log_prob = actor_critic.forward(state)
            new_state, reward, done, _ = env.step(action)
    
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            state = new_state
            
            if done:
                all_lengths.append(steps)
                average_lengths.append(np.mean(all_lengths[-10:]))
                if episode % 10 == 0:                    
                    sys.stdout.write("episode: {}, total length: {}, average length: {} \n".format(episode, steps, average_lengths[-1]))
                break
        
        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = 0 
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + GAMMA**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        #update actor critic
        values = torch.stack(values)
        discounted_rewards = torch.Tensor(discounted_rewards)
        log_probs = torch.stack(log_probs)
        
        advantage = discounted_rewards - value
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss

        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()

    plt.plot(all_lengths)
    plt.plot(average_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode length')
    plt.show()
    
    
if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    a2c(env)    
    
