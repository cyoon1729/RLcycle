from sac2018 import SACAgent
#from sac2019 import SACAgent
from common.utils import mini_batch_train
import gym

env = gym.make("Pendulum-v0")

#SAC 2018 Params
tau = 0.005
gamma = 0.99
value_lr = 3e-3
q_lr = 3e-3
policy_lr = 3e-3
buffer_maxlen = 1000000

# SAC 2019 Params
# gamma = 0.99
# tau = 0.01
# alpha = 0.2
# a_lr = 3e-4
# q_lr = 3e-4
# p_lr = 3e-4
# buffer_maxlen = 1000000

state = env.reset()
#2018 agent
agent = SACAgent(env, gamma, tau, value_lr, q_lr, policy_lr, buffer_maxlen)

#2019 agent
# agent = SACAgent(env, gamma, tau, alpha, q_lr, p_lr, a_lr, buffer_maxlen)

# train
episode_rewards = mini_batch_train(env, agent, 50, 500, 64)