# This is a draft for my blog post, which is now published here:
https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b
# Deep Deterministic Policy Gradients explained 

**Reinforcement Learning in Continuous Action Spaces**

This post is a **thorough** review of Deepmind's publication *"Continuous Control With Deep Reinforcement Learning"* (Lillicrap et al. 2016), in which the Deep Deterministic Policy Gradients (DDPG) is presented, and is written for people who wish to understand the DDPG algorithm. If you wish to just look at the implementation, you can skip to the final section of this post. 



*This post is written with the assumption that the readers are familiar with basic reinforcement learning concepts, value & policy learning, and actor critic methods. Familiarity with python and PyTorch will also be really helpful for reading through this post. If you are not familiar with PyTorch, try to follow the code snippets as if they are pseudo-code.*



## From Discrete Action Spaces to Continuous Action Spaces







## Going through the paper

### Network Schematics

DDPG uses four neural networks: a Q network, a deterministic policy network, a target Q network, and a target policy network. 
$$
\text{Parameters} \\
\begin{aligned}
\theta^{Q}&: \text{Q network} \\
\theta^{\mu}&: \text{Deterministic policy function} \\
\theta^{Q'}&: \text{target Q network} \\
\theta^{\mu'}&: \text{target policy network} \\
\end{aligned}
$$
The target networks are time-delayed copies of their original networks that slowly track the learned networks. In prior methods that do not use target networks, the update equations of the network are interdependent on the values calculated by the network itself, which makes it prone to divergence. For example:

[Q learning]

Greatly improving the stability of learning. 

So, we have the standard Actor & Critic architecture for the deterministic policy network and the Q network:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, init_w=3e-3):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
    
    def forward(self, state, action):
        value = torch.cat([state, action], 1)
        value = F.relu(self.linear1(x))
        value = F.relu(self.linear2(x))
        value = self.linear3(x)
        return value

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 3e-4, init_w=3e-3):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        action = F.relu(self.linear1(state))
        action = F.relu(self.linear2(x))
        action = torch.tanh(self.linear3(x))
        return action
```



And we initialize the networks and target networks as:

```python
actor = Actor(num_states, hidden_size, num_actions)
actor_target = Actor(num_states, hidden_size, num_actions)
critic = Critic(num_states + num_actions, hidden_size, num_actions)
critic_target = Critic(num_states + num_actions, hidden_size, num_actions)

# We initialize the target networks as copies of the original networks
for target_param, param in zip(actor_target.parameters(), actor.parameters()):
    target_param.data.copy_(param.data)
for target_param, param in zip(critic_target.parameters(), critic.parameters()):
    target_param.data.copy_(param.data)
        
```

### Learning

#### Experience Replay

As used in Deep Q learning (and many other RL algorithms), DDPG uses experience replay to update neural network parameters. During each trajectory roll-out, we save all the experience tuples (state, action, reward, next_state) and store them in a finite sized cache--a "replay buffer." Then, we sample random mini-batches of experience from the replay buffer when we update the value and policy networks. 

Here's how the replay buffer looks like:

```Python
import random
from collections import deque

class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)
```



#### Policy & Value Network Updates

The value network is updated similarly as is done in Q-learning. The updated Q value is obtained by the Bellman equation: 
$$
y_{i} = r_{i} + \gamma Q'(s_{i+1}, \mu'(s_{i+1} \vert \theta^{\mu'}) \vert \theta^{Q'})
$$
However, in DDPG, the **next-state Q values are calculated with the target value network and target policy network**. Then, we minimize the mean-squared loss between the updated Q value and the original Q value:
$$
Loss = \frac{1}{N} \sum \limits_{i}(y_{i} - Q(s_{i}, a_{i}\vert \theta^{Q}))^{2}
$$
**\* Note that the original Q value is calculated with the value network, not the target value network. **

In code, this looks like:

```python
Qvals = critic.forward(states, actions)
next_actions = actor_target.forward(next_states)
next_Q = critic_target.forward(next_states, next_actions.detach())
Qprime = rewards + gamma * next_Q
critic_loss = nn.MSELoss(Qvals, Qprime)

critic_optimizer.zero_grad()
critic_loss.backward() 
critic_optimizer.step()
```



For the policy function, our objective is to maximize the expected return:
$$
J(\theta) = \mathbb{E}[Q(s, a)\vert_{s = s_{t}, a_{t} = \mu(s_{t})}]
$$
To calculate the policy loss, we take the derivative of the objective function with respect to the policy parameter. Keep in mind that the actor (policy) function is differentiable, so we have to apply the chain rule. 
$$
\nabla_{\theta^{\mu}} J(\theta) \approx \nabla_{a} Q(s, a) \nabla_{\theta^{\mu}} \mu(s | \theta^{\mu})
$$
But since we are updating the policy in an off-policy way with batches of experience, we take the mean of the sum of gradients calculated from the mini-batch:
$$
\nabla_{\theta^{\mu}} J(\theta) \approx \frac{1}{N} \sum \limits_{i}[\nabla_{a}Q(s, a \vert \theta^{Q}) \vert_{s = s_{i}, a = \mu(s_{i})} \nabla_{\theta^{\mu}}\mu(s|\theta^{\mu}) \vert_{s = s_{i}}]
$$
In code, this looks like:

```python
policy_loss = -critic.forward(states, actor.forward(states)).mean()

actor_optimizer.zero_grad()
policy_loss.backward()
actor_optimizer.step()
```

Where the optimizers use Adaptive Moment Estimation (ADAM):

```
actor_optimizer = optim.Adam(actor.parameters(), lr=actor_learning_rate)
critic_optimizer = optim.Adam(critic.parameters(), lr=critic_learning_rate)
```

#### Target Network Updates

 We make a copy of the target network parameters and have them slowly track those of the learned networks via  "soft updates," as illustrated below:
$$
\theta^{Q'} \gets \tau \theta^{Q} + (1-\tau)\theta^{Q'} \\
\theta^{\mu'} \gets \tau \theta^{\mu} + (1-\tau)\theta^{\mu'} \\
\text{where} \quad\tau \ll 1
$$
This can be implemented very simply:

```python
# update target networks 
for target_param, param in zip(actor_target.parameters(), actor.parameters()):
    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
       
for target_param, param in zip(critic_target.parameters(), critic.parameters()):
    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

```

### Exploration

In Reinforcement learning for discrete action spaces, exploration was done via probabilistically selecting a random action (such as epsilon-greedy or Boltzmann exploration). For continuous action spaces, exploration is done via adding the *Ornstein-Uhlenbeck* noise to the action output (Uhlenbeck & Ornstein, 1930):
$$
\mu'(s_{t}) = \mu(s_{t} \vert \theta^{\mu}_{t}) + \mathcal{N}
$$
The *Ornstein-Uhlenbeck Process* generates noise that is correlated with the previous noise, as to prevent the noise from canceling out or freezing the overall dynamics. Wikipedia provides a thorough explanation of the *Ornstein-Uhlenbeck Process*.

Here's a python implementation written by Pong et al:

``` python
"""
Taken from https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
"""
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0): 
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)
```

So we input the action produced by the actor network into `get_action()` and get a new action to which the temporally correlated noise is added. 

We are all set now!

## Putting them all together 

We have here the Replay Buffer, the Ornstein-Uhlenbeck Process, and the normalized Action Wrapper for OpenAI Gym continuous control environments in *utils.py*:

[Gist]

And the Actor & Critic networks in *models.py*:

[Gist]

And the DDPG agent in *ddpg.py*:

[Gist]

And the test in *main.py*:

[Gist]

 

And we can see if the DDPG agent learns optimal policy for the classic Inverted Pendulum task:

[Image]



Awesome! That's it for DDPG!

## References



