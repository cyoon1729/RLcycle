from collections import deque
from typing import Callable, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig

from rlcycle.common.abstract.action_selector import ActionSelector
from rlcycle.common.abstract.agent import Agent
from rlcycle.dqn_base.learner import DQNLearner


class DQNBaseAgent(Agent):
    """Configurable DQN base agent; works with Dueling DQN, C51, QR-DQN, etc

    Attributes:
        env (gym.ENV): Gym environment for RL agent
        learner (LEARNER): Carries and updates agent value/policy models
        replay_buffer (ReplayBuffer): Replay buffer for experience replay (PER as wrapper)
        action_selector (ActionSelector): Callable for DQN action selection (EpsGreedy as wrapper)
        use_n_step (bool): Indication of using n-step updates
        transition_queue (Deque): deque for tracking and preprocessing n-step transitions

    """

    def __init__(
        self,
        experiment_info: DictConfig,
        hyper_params: DictConfig,
        model_cfg: DictConfig,
    ):
        Agent.__init__(self, experiment_info, hyper_params, model_cfg)
        self.use_n_step = self.hyper_params.n_step > 1
        self.transition_queue = deque(maxlen=self.hyper_params.n_step)

    def _initialize(self):
        """Initialize agent components"""
        self.env = build_env(self.experiment_info.env)
        self.learner = build_learner(
            self.experiment_info, self.hyper_params, self.model_cfg
        )
        self.replay_buffer = ReplayBuffer(self.hyper_params)
        if self.hyper_params.use_per:
            self.learner = PERLearner(self.learner)
            self.replay_buffer = PrioritizedReplayBuffer(
                self.replay_buffer, self.hyper_params
            )
        self.action_selector = build_action_selector(self.experiment_info)
        self.action_selector = EpsGreedy(
            self.action_selector, self.env.action_space, self.hyper_params
        )

    def step(
        self, state: np.ndarray, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.float64, bool]:
        """Carry out single env step and return info

        Params:
            state (np.ndarray): current env state
            action (np.ndarray): action to be executed

        """
        next_state, reward, done, _ = self.env.step(action)
        return state, action, reward, next_state, done

    def train(self):
        """Main training loop"""
        for episode_i in range(self.experiment_info.total_num_episodes):
            state = self.env.reset()
            episode_reward = 0

            while not done:
                if self.args.render:
                    self.env.render()

                action = self.action_selector(state)
                state, action, reward, next_state, done = self.step(state, action)
                episode_reward = episode_reward + reward

                if self.use_n_step:
                    transition = [state, action, reward, next_state, done]
                    self.transition_queue.append(transition)
                    if len(self.transition_queue) > self.hyper_params.n_step:
                        n_step_transition = preprocess_n_step(self.transition_queue)
                        self.replay_buffer.add(*n_step_transition)
                else:
                    self.replay_buffer.add(state, action, reward, next_state, done)

                if len(self.replay_buffer) > self.hyper_params.update_starting_point:
                    experience = self.replay_buffer.sample()
                    info = self.learner.update_model(experience)

                    if self.hyper_params.use_per:
                        q_loss, indices, new_priorities = info
                        self.replay_buffer.update_priorities(indices, new_priorities)
                    else:
                        q_loss = info

            if episode_i % self.experiment_info.test_interval:
                self.test(episode_i)
                self.learner.save_params()

    def test(self, step: int):
        """Test policy without random exploration a number of times
        
        Params:
            step (int): step information, by episode number of model update step

        """
        print("====TEST START====")
        self.action_selector.exploration = False
        episode_rewards = []
        for test_i in range(self.experiment_info.test_num):
            episode_reward = 0
            while not done:
                self.env.render()
                action = self.action_selector(state)
                state, action, reward, next_state, done = self.step(state, action)
                episode_reward = episode_reward + reward

            print(f"step: {step} \ttest: {test_i} \tepisode reward: {episode_reward}")
            episode_rewards.append(episode_reward)

        print(f"STEP: {step} \tMEAN REWARD: {np.mean(episode_rewards)}")
        print("====TEST END====")
