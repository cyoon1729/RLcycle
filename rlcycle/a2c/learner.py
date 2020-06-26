from copy import deepcopy
from typing import List, Tuple

import torch
import torch.optim as optim
from omegaconf import DictConfig
from torch.nn.utils import clip_grad_norm_

from rlcycle.build import build_loss, build_model
from rlcycle.common.abstract.learner import Learner
from rlcycle.common.models.base import BaseModel


class A2CLearner(Learner):
    """Advantage Actor Critic Learner (can also be used for A3C).

    Attributes:
        critic (BaseModel): critic network
        critic_optimizer (torch.Optimizer): critic optimizer
        critic loss_fn (Loss): critic loss function
        actor (BaseModel): actor network
        actor_optimizer (torch.Optimizer): actor optimizer
        actor loss_fn (Loss): actor loss function
        update_step (int): counter for update step

    """

    def __init__(
        self,
        experiment_info: DictConfig,
        hyper_params: DictConfig,
        model_cfg: DictConfig,
    ):
        Learner.__init__(self, experiment_info, hyper_params, model_cfg)
        self.update_step = 0

        self._initialize()

    def _initialize(self):
        """Initialize networks, optimizer, loss function."""
        # Set env-specific input dims and output dims for models

        self.model_cfg.critic.params.model_cfg.state_dim = (
            self.model_cfg.actor.params.model_cfg.state_dim
        ) = self.experiment_info.env.state_dim
        self.model_cfg.critic.params.model_cfg.action_dim = (
            self.model_cfg.actor.params.model_cfg.action_dim
        ) = self.experiment_info.env.action_dim

        # Initialize critic models, optimizers, and loss function
        self.critic = build_model(self.model_cfg.critic, self.device)

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.hyper_params.critic_learning_rate
        )
        self.critic_loss_fn = build_loss(
            self.experiment_info.critic_loss,
            self.hyper_params,
            self.experiment_info.device,
        )

        # Initialize actor model, optimizer, and loss function
        self.actor = build_model(self.model_cfg.actor, self.device)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.hyper_params.actor_learning_rate
        )
        self.actor_loss_fn = build_loss(
            self.experiment_info.actor_loss,
            self.hyper_params,
            self.experiment_info.device,
        )

    def update_model(
        self, trajectories: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        """Update model."""
        critic_loss = actor_loss = 0
        for trajectory in trajectories:
            # Compute loss
            critic_loss_element_wise, values = self.critic_loss_fn(
                (self.critic), trajectory,
            )
            critic_loss += critic_loss_element_wise.mean()

            trajectory = trajectory + (values,)
            actor_loss_element_wise = self.actor_loss_fn((self.actor), trajectory,)
            actor_loss += actor_loss_element_wise.mean()

        # Take mean of losses computed from all trajectories
        critic_loss = critic_loss / len(trajectories)
        actor_loss = actor_loss / len(trajectories)

        # Update critic networks
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(
            self.critic.parameters(), self.hyper_params.critic_gradient_clip
        )
        self.critic_optimizer.step()

        # Update actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor.parameters(), self.hyper_params.actor_gradient_clip)
        self.actor_optimizer.step()

        info = (critic_loss, actor_loss)
        return info

    def get_policy(self, target_device: torch.device) -> BaseModel:
        """Return policy mapped to target device"""
        policy_copy = deepcopy(self.network)
        policy_copy.to(target_device)
        return policy_copy