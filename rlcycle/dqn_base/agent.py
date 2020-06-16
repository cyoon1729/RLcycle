from collections import deque
from typing import Callable, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig

from rlcycle.build import (build_action_selector, build_env, build_learner,
                           build_loss)
from rlcycle.common.abstract.agent import Agent
from rlcycle.common.buffer.prioritized_replay_buffer import \
    PrioritizedReplayBuffer
from rlcycle.common.buffer.replay_buffer import ReplayBuffer
from rlcycle.dqn_base.action_selector import EpsGreedy
from rlcycle.dqn_base.learner import DQNLearner
from rlcycle.common.utils.common_utils import np2tensor

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
        # Build env and env specific model params
        self.env = build_env(self.experiment_info)
        self.model_cfg.params.model_cfg.state_dim = self.env.observation_space.shape
        self.model_cfg.params.model_cfg.action_dim = self.env.action_space.n

        # Build learner
        self.learner = build_learner(
            self.experiment_info, self.hyper_params, self.model_cfg
        )

        # Build replay buffer, wrap with PER buffer if using it
        self.replay_buffer = ReplayBuffer(self.hyper_params.replay_buffer_size)
        if self.hyper_params.use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                self.replay_buffer, self.experiment_info, self.hyper_params
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
        for episode_i in range(self.experiment_info.total_num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                if self.experiment_info.render:
                    self.env.render()

                action = self.action_selector(self.learner.network, state)
                state, action, reward, next_state, done = self.step(state, action)
                episode_reward = episode_reward + reward

                if self.use_n_step:
                    transition = [state, action, reward, next_state, done]
                    self.transition_queue.append(transition)
                    if len(self.transition_queue) > self.hyper_params.n_step:
                        #TODO: Implement preprocess_n_step
                        n_step_transition = preprocess_n_step(self.transition_queue)
                        self.replay_buffer.add(*n_step_transition)
                else:
                    self.replay_buffer.add(state, action, reward, next_state, done)

                if len(self.replay_buffer) >= self.hyper_params.update_starting_point:
                    experience = self.replay_buffer.sample()
                    info = self.learner.update_model(
                        self._preprocess_experience(experience)
                    )
                    self.update_step = self.update_step + 1

                    if self.hyper_params.use_per:
                        q_loss, indices, new_priorities = info
                        self.replay_buffer.update_priorities(indices, new_priorities)
                    else:
                        q_loss = info
                    
                    self.action_selector.decay_epsilon()

            print(f"[TRAIN] episode num: {episode_i} \tepisode reward: {episode_reward}")
            
            if episode_i % self.experiment_info.test_interval == 0:
                self.test(episode_i, self.update_step)
                # self.learner.save_params()

    def test(self, episode_i: int, update_step: int):
        """Test policy without random exploration a number of times
        
        Params:
            step (int): step information, by episode number of model update step

        """
        print("====TEST START====")
        self.action_selector.exploration = False
        episode_rewards = []
        for test_i in range(self.experiment_info.test_num):
            state = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                self.env.render()
                action = self.action_selector(self.learner.network, state)
                state, action, reward, next_state, done = self.step(state, action)
                episode_reward = episode_reward + reward

            print(f"episode num: {episode_i} \tupdate step: {update_step} \ttest: {test_i} \tepisode reward: {episode_reward}")
            episode_rewards.append(episode_reward)

        print(f"EPISODE NUM: {episode_i} \tUPDATE STEP: {update_step} \tMEAN REWARD: {np.mean(episode_rewards)}")
        print("====TEST END====")
        self.action_selector.exploration = True

    def _preprocess_experience(self, experience: Tuple[np.ndarray]):
        states, actions, rewards, next_states, dones, indices, weights = tuple([np2tensor(np_arr, self.device) for np_arr in experience])
        actions = actions.long().view(-1, 1)
        rewards = rewards.view(-1, 1)
        dones = dones.view(-1, 1)
        indices = indices.long().detach().cpu().numpy() # indices need not be tensor
        weights = weights.view(-1, 1)
        return states, actions, rewards, next_states, dones, indices, weights