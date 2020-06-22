from collections import deque
from typing import Callable, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig

from rlcycle.build import build_action_selector, build_learner, build_loss
from rlcycle.common.abstract.agent import Agent
from rlcycle.common.buffer.prioritized_replay_buffer import \
    PrioritizedReplayBuffer
from rlcycle.common.buffer.replay_buffer import ReplayBuffer
from rlcycle.common.utils.common_utils import np2tensor, preprocess_nstep
from rlcycle.dqn_base.action_selector import EpsGreedy
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
        self.update_step = 0

        self._initialize()

    def _initialize(self):
        """Initialize agent components"""
        # set env specific model params
        self.model_cfg.params.model_cfg.state_dim = self.env.observation_space.shape
        self.model_cfg.params.model_cfg.action_dim = self.env.action_space.n

        # Build learner
        self.learner = build_learner(
            self.experiment_info, self.hyper_params, self.model_cfg
        )

        # Build replay buffer, wrap with PER buffer if using it
        self.replay_buffer = ReplayBuffer(self.hyper_params)
        if self.hyper_params.use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                self.replay_buffer, self.hyper_params
            )

        # Build action selector, wrap with e-greedy exploration
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
        step = 0
        for episode_i in range(self.experiment_info.total_num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                if self.experiment_info.train_render:
                    self.env.render()

                action = self.action_selector(self.learner.network, state)
                state, action, reward, next_state, done = self.step(state, action)
                episode_reward = episode_reward + reward
                step = step + 1

                if self.use_n_step:
                    transition = [state, action, reward, next_state, done]
                    self.transition_queue.append(transition)
                    if len(self.transition_queue) == self.hyper_params.n_step:
                        n_step_transition = preprocess_nstep(
                            self.transition_queue, self.hyper_params.gamma
                        )
                        self.replay_buffer.add(*n_step_transition)
                else:
                    self.replay_buffer.add(state, action, reward, next_state, done)

                if len(self.replay_buffer) >= self.hyper_params.update_starting_point:
                    if step % self.hyper_params.train_freq == 0:
                        experience = self.replay_buffer.sample()
                        info = self.learner.update_model(
                            self._preprocess_experience(experience)
                        )
                        self.update_step = self.update_step + 1

                        if self.hyper_params.use_per:
                            q_loss, indices, new_priorities = info
                            self.replay_buffer.update_priorities(
                                indices, new_priorities
                            )
                        else:
                            q_loss = info

                        self.action_selector.decay_epsilon()

                state = next_state

            print(
                f"[TRAIN] episode num: {episode_i} | update step: {self.update_step} |"
                f"episode reward: {episode_reward} | epsilon: {self.action_selector.eps}"
            )

            if episode_i % self.experiment_info.test_interval == 0:
                policy_copy = self.learner.get_policy(self.device)
                self.test(
                    policy_copy, self.action_selector, episode_i, self.update_step
                )
                # self.learner.save_params()

    def _preprocess_experience(self, experience: Tuple[np.ndarray]):
        states, actions, rewards, next_states, dones = experience[:5]
        if self.hyper_params.use_per:
            indices, weights = experience[-2:]

        states = np2tensor(states, self.device)
        actions = np2tensor(actions.reshape(-1, 1), self.device)
        rewards = np2tensor(rewards.reshape(-1, 1), self.device)
        next_states = np2tensor(next_states, self.device)
        dones = np2tensor(dones.reshape(-1, 1), self.device)

        experience = (states, actions.long(), rewards, next_states, dones)

        if self.hyper_params.use_per:
            weights = np2tensor(weights.reshape(-1, 1), self.device)
            experience = experience + (indices, weights,)

        return experience
