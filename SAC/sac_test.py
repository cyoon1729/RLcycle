#from sac2018 import SACAgent
from sac2019 import SACAgent
import gym

env = gym.make("Pendulum-v0")

#SAC 2018 Params
# tau = 0.005
# gamma = 0.99
# value_lr = 3e-3
# q_lr = 3e-3
# policy_lr = 3e-3
# buffer_maxlen = 1000000

# SAC 2019 Params
gamma = 0.99
tau = 0.01
alpha = 0.2
a_lr = 3e-4
q_lr = 1e-3
p_lr = 1e-3
buffer_maxlen = 1000000

#2018 agent
#agent = SACAgent(env, gamma, tau, value_lr, q_lr, policy_lr, buffer_maxlen)

#2019 agent
agent = SACAgent(env, gamma, tau, alpha, q_lr, p_lr, a_lr, buffer_maxlen)

def rescale_action(action):
        return action * (agent.action_range[1] - agent.action_range[0]) / 2.0 +\
            (agent.action_range[1] + agent.action_range[0]) / 2.0

def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(rescale_action(action))
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)

            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state

    return episode_rewards

episode_rewards = mini_batch_train(env, agent, 80, 300, 64)
