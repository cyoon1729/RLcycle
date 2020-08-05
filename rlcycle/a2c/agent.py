import time
from typing import Tuple

import numpy as np
from omegaconf import DictConfig, OmegaConf
import ray
import torch

from rlcycle.a2c.worker import TrajectoryRolloutWorker
from rlcycle.build import build_action_selector, build_learner
from rlcycle.common.abstract.agent import Agent
from rlcycle.common.utils.common_utils import np2tensor
from rlcycle.common.utils.logger import Logger


class A2CAgent(Agent):
    """Synchronous Advantage Actor Critic (A2C; data parallel) agent

    Attributes:
        learner (Learner): learner for A2C
        update_step (int): update step counter
        action_selector (ActionSelector): action selector for testing
        logger (Logger): WandB logger

    """

    def __init__(
        self,
        experiment_info: DictConfig,
        hyper_params: DictConfig,
        model_cfg: DictConfig,
    ):
        Agent.__init__(self, experiment_info, hyper_params, model_cfg)
        self.update_step = 0

        self._initialize()

    def _initialize(self):
        """Set env specific configs and build learner."""
        self.experiment_info.env.state_dim = self.env.observation_space.shape[0]
        if self.experiment_info.env.is_discrete:
            self.experiment_info.env.action_dim = self.env.action_space.n
        else:
            self.experiment_info.env.action_dim = self.env.action_space.shape[0]
            self.experiment_info.env.action_range = [
                self.env.action_space.low.tolist(),
                self.env.action_space.high.tolist(),
            ]

        self.learner = build_learner(
            self.experiment_info, self.hyper_params, self.model_cfg
        )

        self.action_selector = build_action_selector(
            self.experiment_info, self.use_cuda
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
        """Carry out one environment step"""
        # A2C only uses this for test
        next_state, reward, done, _ = self.env.step(action)
        return state, action, reward, next_state, done

    def train(self):
        """Run data parellel training (A2C)."""
        ray.init()
        workers = []
        for worker_id in range(self.experiment_info.num_workers):
            worker = ray.remote(num_cpus=1)(TrajectoryRolloutWorker).remote(
                worker_id, self.experiment_info, self.learner.model_cfg.actor
            )
            workers.append(worker)

        print("Starting training...")
        time.sleep(1)
        while self.update_step < self.experiment_info.max_update_steps:
            # Run and retrieve trajectories
            trajectory_infos = ray.get(
                [worker.run_trajectory.remote() for worker in workers]
            )

            # Run update step with multiple trajectories
            trajectories_tensor = [
                self._preprocess_trajectory(traj["trajectory"])
                for traj in trajectory_infos
            ]
            info = self.learner.update_model(trajectories_tensor)
            self.update_step = self.update_step + 1

            # Synchronize worker policies
            policy_state_dict = self.learner.actor.state_dict()
            for worker in workers:
                worker.synchronize_policy.remote(policy_state_dict)

            if self.experiment_info.log_wandb:
                worker_average_score = np.mean(
                    [traj["score"] for traj in trajectory_infos]
                )
                log_dict = dict(episode_reward=worker_average_score)
                if self.update_step > 0:
                    log_dict["critic_loss"] = info[0]
                    log_dict["actor_loss"] = info[1]
                self.logger.write_log(log_dict)

            if self.update_step % self.experiment_info.test_interval == 0:
                policy_copy = self.learner.get_policy(self.use_cuda)
                average_test_score = self.test(
                    policy_copy,
                    self.action_selector,
                    self.update_step,
                    self.update_step,
                )
                if self.experiment_info.log_wandb:
                    self.logger.write_log(
                        log_dict=dict(average_test_score=average_test_score),
                    )

                self.learner.save_params()

    def _preprocess_trajectory(
        self, trajectory: Tuple[np.ndarray, ...]
    ) -> Tuple[torch.Tensor]:
        """Preprocess trajectory for pytorch training"""
        states, actions, rewards = trajectory

        states = np2tensor(states, self.use_cuda)
        actions = np2tensor(actions.reshape(-1, 1), self.use_cuda)
        rewards = np2tensor(rewards.reshape(-1, 1), self.use_cuda)

        if self.experiment_info.is_discrete:
            actions = actions.long()

        trajectory = (states, actions, rewards)

        return trajectory
