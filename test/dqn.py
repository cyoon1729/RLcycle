import gym
import numpy as np

from agents.dqn import DQNAgent

MAX_EPISODES = 500
MAX_STEPS = 300
BATCH_SIZE = 32

env_id = "CartPole-v0"
env = gym.make(env_id)
agent = DQNAgent(env, use_conv=False)

if __name__ == "__main__":
    episode_rewards = []

    for episode in range(MAX_EPISODES):
        state = env.reset()
        episode_reward = 0
        for step in range(MAX_STEPS):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)

            if done:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break
            else:
                if(len(agent.replay_buffer) > BATCH_SIZE):
                    agent.update_model(BATCH_SIZE)
                state = next_state
                episode_reward += reward
