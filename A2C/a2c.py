import sys
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# neural network parameters
hidden_size = 256
learning_rate = 3e-4
num_steps = 100

# Constants
GAMMA = 0.9

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(num_inputs, hidden_size)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)

        """self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size)
            nn.Relu()
        ) """
    
    def forward(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        # get value
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value, dim=1)
        
        # get policy distribution
        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])

        return value, highest_prob_action, log_prob



class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        #self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x 
    
    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_size, learning_rate=3e-4):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        #self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x 

def nstep_a2c(env):
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n
    
    actor_critic = ActorCritic(num_inputs, num_actions, hidden_size)
    value_criterion = nn.MSELoss()
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    state = env.reset()
    for episode in range(20000):
        log_probs = []
        values = []
        rewards = []

        for steps in range(num_steps):
            value, action, log_prob = actor_critic.forward(state)
            new_state, reward, done, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)

            state = new_state
            
            if done:
                sys.stdout.write("episode: {}, total length: {} \n".format(episode, steps))
                break
        
        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = 0 
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + GAMMA**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        value_loss = []
        policy_gradient = []
        #update actor critic

        values = torch.cat(values)
        rewards = torch.cat(rewards).detach()
        log_probs = torch.cat(log_probs)
        
        advantage = returns - value
        actor_loss = (-log_probs * advantage.detach()).mean()
        critic_loss = value_criterion(returns, value).mean()
        #critic_loss = advantage.pow(2).mean()
        ac_loss = actor_loss + 0.5 * critic_loss
        
        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()


def a2c(env):
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n
    
    policy_network = PolicyNetwork(num_inputs, num_outputs, hidden_size)
    value_network = ValueNetwork(num_inputs, hidden_size)

    value_criterion  = nn.MSELoss()
    value_optimizer  = optim.Adam(value_network.parameters(), lr=learning_rate)
    policy_optimizer = optim.Adam(policy_network.parameters(), lr=learning_rate)

    state = env.reset()
    for episode in range(20000):
        log_probs = []
        values = []
        rewards = []

        for steps in range(num_steps):
            action, log_prob = policy_network.get_action(state)
            value = value_network.forward(state)
            new_state, reward, done, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)

            state = new_state
    
            if done:
                sys.stdout.write("episode: {}, total length: {} \n".format(episode, steps))
                break
    
        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = 0 
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + GAMMA**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)
        
        policy_gradient = []
        value_loss = []
        
        for log_prob, returns, value in zip(log_probs, discounted_rewards, values):
            advantage = returns - value
            #policy_gradient.append(-log_prob * advantage)
            #value_loss = value_criterion(returns, value)
            """policy_optimizer.zero_grad()
            policy_loss = -log_prob * advantage
            policy_loss.backward(retain_graph=True)
            policy_optimizer.step()

            value_optimizer.zero_grad()
            value_loss = value_criterion(returns, value)
            value_loss.backward()
            value_optimizer.step()"""
        

        #policy_gradient = torch.stack(policy_gradient).sum().mean()
        policy_loss = torch.FloatTensor(policy_gradient)
        value_loss = torch.stack(value_loss)
        value_loss = torch.sum(value_loss).mean()
        policy_gradient = torch.stack(policy_gradient)
        policy_gradient = torch.sum(policy_gradient).mean()
        # Actor update
        policy_optimizer.zero_grad()
        value_optimizer.zero_grad()
        policy_gradient.backward(retain_graph=True)
        value_loss.backward()
        policy_optimizer.step()

        # Critic update
        
        
        value_optimizer.step()
      
if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    a2c(env)


    




    
    