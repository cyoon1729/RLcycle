import gym

from a2c import A2CAgent


env = gym.make("CartPole-v0")
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
MAX_EPISODE = 1500
MAX_STEPS = 500

lr = 1e-4
gamma = 0.99

agent = A2CAgent(env, gamma, lr)

def run():
    for episode in range(MAX_EPISODE):
        state = env.reset()
        trajectory = [] # [[s, a, r, s', done], [], ...]
        episode_reward = 0
        for steps in range(MAX_STEPS):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            trajectory.append([state, action, reward, next_state, done])
            episode_reward += reward

            if done:
                break
                
            state = next_state
        if episode % 10 == 0:
            print("Episode " + str(episode) + ": " + str(episode_reward))
        agent.update(trajectory)

run()