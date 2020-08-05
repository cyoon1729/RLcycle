from collections import deque
from typing import Tuple

import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

from rlcycle.build import build_action_selector, build_learner
from rlcycle.common.abstract.agent import Agent
from rlcycle.common.buffer.prioritized_replay_buffer import PrioritizedReplayBuffer
from rlcycle.common.buffer.replay_buffer import ReplayBuffer
from rlcycle.common.utils.common_utils import np2tensor, preprocess_nstep
from rlcycle.common.utils.logger import Logger
from rlcycle.ddpg.action_selector import RandomActionsStarts


class SACAgent(Agent):
    """Soft Actor Critic agent

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
        self.experiment_info.env.state_dim = self.env.observation_space.shape[0]
        self.experiment_info.env.action_dim = self.env.action_space.shape[0]
        self.experiment_info.env.action_range = [
            self.env.action_space.low.tolist(),
            self.env.action_space.high.tolist(),
        ]

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

        # Build action selector
        self.action_selector = build_action_selector(
            self.experiment_info, self.use_cuda
        )
        self.action_selector = RandomActionsStarts(
            self.action_selector,
            max_exploratory_steps=self.hyper_params.max_exploratory_steps,
        )

        # Build logger
        if self.experiment_info.log_wandb:
            experiment_cfg = OmegaConf.create(
                dict(
                    experiment_info=self.experiment_info,
                    hyper_params=self.hyper_params,
                    model=self.learner.model_cfg,
                )
            )
            self.logger = Logger(experiment_cfg)

    def step(
        self, state: np.ndarray, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.float64, np.ndarray, bool]:
        """Carry out single env step and return info

        Params:
            state (np.ndarray): current env state
            action (np.ndarray): action to be executed

        """
        next_state, reward, done, _ = self.env.step(
            self.action_selector.rescale_action(action)
        )
        return state, action, reward, next_state, done

    def train(self):
        """Main training loop"""
        step = 0
        for episode_i in range(self.experiment_info.total_num_episodes):
            # Test when we have to
            if episode_i % self.experiment_info.test_interval == 0:
                policy_copy = self.learner.get_policy(self.use_cuda)
                average_test_score = self.test(
                    policy_copy, self.action_selector, episode_i, self.update_step
                )
                if self.experiment_info.log_wandb:
                    self.logger.write_log(
                        log_dict=dict(average_test_score=average_test_score),
                    )
                self.learner.save_params()

            # Run episode
            state = self.env.reset()
            losses = dict(
                critic1_loss=[], critic2_loss=[], actor_loss=[], alpha_loss=[]
            )
            episode_reward = 0
            done = False

            while not done:
                if self.experiment_info.train_render:
                    self.env.render()

                action = self.action_selector(self.learner.actor, state, episode_i)
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

                state = next_state

                if len(self.replay_buffer) >= self.hyper_params.update_starting_point:
                    experience = self.replay_buffer.sample()
                    info = self.learner.update_model(
                        self._preprocess_experience(experience)
                    )
                    self.update_step = self.update_step + 1
                    critic1_loss, critic2_loss, actor_loss, alpha_loss = info[:4]
                    losses["critic1_loss"].append(critic1_loss)
                    losses["critic2_loss"].append(critic2_loss)
                    losses["actor_loss"].append(actor_loss)
                    losses["alpha_loss"].append(alpha_loss)

                    if self.hyper_params.use_per:
                        indices, new_priorities = info[-2:]
                        self.replay_buffer.update_priorities(indices, new_priorities)

            print(
                f"[TRAIN] episode num: {episode_i} | update step: {self.update_step} |"
                f" episode reward: {episode_reward}"
            )

            if self.experiment_info.log_wandb:
                log_dict = dict(episode_reward=episode_reward)
                if self.update_step > 0:
                    log_dict["critic1_loss"] = np.mean(losses["critic1_loss"])
                    log_dict["critic2_loss"] = np.mean(losses["critic2_loss"])
                    log_dict["actor_loss"] = np.mean(losses["actor_loss"])
                    log_dict["alpha_loss"] = np.mean(losses["alpha_loss"])
                self.logger.write_log(log_dict)

    def _preprocess_experience(
        self, experience: Tuple[np.ndarray]
    ) -> Tuple[torch.Tensor]:
        """Convert collected experience to pytorch tensors."""
        states, actions, rewards, next_states, dones = experience[:5]
        if self.hyper_params.use_per:
            indices, weights = experience[-2:]

        states = np2tensor(states, self.use_cuda)
        actions = np2tensor(actions, self.use_cuda)
        rewards = np2tensor(rewards.reshape(-1, 1), self.use_cuda)
        next_states = np2tensor(next_states, self.use_cuda)
        dones = np2tensor(dones.reshape(-1, 1), self.use_cuda)

        experience = (states, actions, rewards, next_states, dones)

        if self.hyper_params.use_per:
            weights = np2tensor(weights.reshape(-1, 1), self.use_cuda)
            experience = experience + (indices, weights,)

        return experience
