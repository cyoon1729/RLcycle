# Policy-Gradient-Methods

Author: Chris Yoon

Implementations of important policy gradient algorithms in deep reinforcement learning.



## Implementations

- **Advantage Actor-Critic (A2C)**
  
  Paper: ["Asynchronous Methods for Deep Reinforcement Learning" (Mnih et al., 2016)](https://arxiv.org/pdf/1602.01783.pdf)
 
- **Asynchronous Advantage Actor-Critic (A3C)**
  
  Paper: ["Asynchronous Methods for Deep Reinforcement Learning" (Mnih et al., 2016)](https://arxiv.org/pdf/1602.01783.pdf)

- **Deep Deterministic Policy Gradients (DDPG)**

  Paper: ["Continuous control with deep reinforcement learning" (Lillicrap et al., 2015)](https://arxiv.org/abs/1509.02971)

- **Twin Dueling Deep Deterministic Policy Gradients (TD3)**

  Paper: ["Addressing Function Approximation Error in Actor-Critic Methods" (Fujimoto et al., 2018)](https://arxiv.org/abs/1802.09477)
  
- **Soft Actor Critic (SAC)**

  - Paper (`sac2018.py`): ["Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor" (Haarnoja et al., 2018)](https://arxiv.org/abs/1801.01290)
  - Paper (`sac2019.py`): ["Soft Actor-Critic Algorithms and Applications" (Haarnoja et al., 2019)](https://arxiv.org/abs/1812.05905)
  - Algorithm in 2018 paper uses value network, double Q networks, and Gaussian policy. Algorithm in 2019 paper uses double Q networks and Gaussian policy, and adds automatic entropy tuning.
  - TODO: SAC for discrete action space 

More implementations will be added soon.

## Known Dependencies

- Python 3.6
- PyTorch 1.0.2
- gym 0.12.5

## How to run:

Install package

```bash
git clone git@github.com:cyoon1729/Policy-Gradient-Methods.git
cd Policy-Gradient-Methods
pip install .
```

Example:

```python
import gym

from policygradients.common.utils import mini_batch_train  # import training function
from policygradients.td3.td3 import TD3Agent  # import agent from algorithm of interest

# Create Gym environment
env = gym.make("Pendulum-v0")

# check agent class for initialization parameters and initialize agent
gamma = 0.99
tau = 1e-2
noise_std = 0.2
bound = 0.5
delay_step = 2
buffer_maxlen = 100000
critic_lr = 1e-3
actor_lr = 1e-3

agent = TD3Agent(env, gamma, tau, buffer_maxlen, delay_step, noise_std, bound, critic_lr, actor_lr)

# define training parameters
max_episodes = 100
max_steps = 500
batch_size = 32

# train agent with mini_batch_train function
episode_rewards = mini_batch_train(env, agent, max_episodes, max_steps, batch_size)
```

