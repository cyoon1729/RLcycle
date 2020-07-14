from typing import Dict, List, Tuple

import numpy as np
from omegaconf import DictConfig
import torch

from rlcycle.a2c.worker import TrajectoryRolloutWorker
from rlcycle.build import build_loss, build_model
from rlcycle.common.utils.common_utils import np2tensor


class ComputesGradients:
    """TrajectorRolloutWorker wrapper for gradient parallelization

    Attributes:
        worker (TrajectoryRolloutWorker): worker that is wrapped
        hyper_params (DictConfig): algorithm hyperparameters
        critic (BaseModel): critic network
        critic_loss_fn (Loss): critic (value) loss function
        actor_loss_fn (Loss): actor (policy) loss function

    """

    def __init__(
        self,
        worker: TrajectoryRolloutWorker,
        hyper_params: DictConfig,
        model_cfg: DictConfig,
    ):
        self.worker = worker
        self.hyper_params = hyper_params
        self.model_cfg = model_cfg

        # Build critic
        self.critic = build_model(self.model_cfg.critic, self.worker.use_cuda)

        # Build loss functions
        self.critic_loss_fn = build_loss(
            self.worker.experiment_info.critic_loss,
            self.hyper_params,
            self.worker.use_cuda,
        )

        self.actor_loss_fn = build_loss(
            self.worker.experiment_info.actor_loss,
            self.hyper_params,
            self.worker.use_cuda,
        )

    def compute_grads_with_traj(self) -> Tuple[List[torch.Tensor], ...]:
        trajectory_info = self.worker.run_trajectory()
        trajectory_tensors = self._preprocess_trajectory(trajectory_info["trajectory"])

        # Compute loss
        critic_loss_element_wise, values = self.critic_loss_fn(
            (self.critic), trajectory_tensors,
        )
        critic_loss = critic_loss_element_wise.mean()

        trajectory_tensors = trajectory_tensors + (values,)
        actor_loss_element_wise = self.actor_loss_fn(
            (self.worker.actor), trajectory_tensors,
        )
        actor_loss = actor_loss_element_wise.mean()

        # Compute and save gradients
        self.critic.zero_grad()
        critic_loss.backward()
        critic_grads = []
        for param in self.critic.parameters():
            critic_grads.append(param.grad)

        self.worker.actor.zero_grad()
        actor_loss.backward()
        actor_grads = []
        for param in self.worker.actor.parameters():
            actor_grads.append(param.grad)

        computed_grads = (critic_grads, actor_grads)
        step_info = dict(
            worker_rank=self.worker.rank,
            critic_loss=critic_loss,
            actor_loss=actor_loss,
            score=trajectory_info["score"],
        )

        return computed_grads, step_info

    def synchronize(self, state_dicts: Dict[str, dict]):
        self.critic.load_state_dict(state_dicts["critic"])
        self.worker.synchronize_policy(state_dicts["actor"])

    def _preprocess_trajectory(
        self, trajectory: Tuple[np.ndarray, ...]
    ) -> Tuple[torch.Tensor]:
        """Preprocess trajectory for pytorch training"""
        states, actions, rewards = trajectory

        states = np2tensor(states, self.worker.use_cuda)
        actions = np2tensor(actions.reshape(-1, 1), self.worker.use_cuda)
        rewards = np2tensor(rewards.reshape(-1, 1), self.worker.use_cuda)

        if self.worker.experiment_info.is_discrete:
            actions = actions.long()

        trajectory = (states, actions, rewards)

        return trajectory
