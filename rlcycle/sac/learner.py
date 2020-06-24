from copy import deepcopy
from typing import Tuple

import torch
import torch.optim as optim
from omegaconf import DictConfig
from torch.nn.utils import clip_grad_norm_

from rlcycle.build import build_loss, build_model
from rlcycle.common.abstract.learner import Learner
from rlcycle.common.utils.common_utils import hard_update, soft_update


class SACLearner(Learner):
    """Learner for Soft Actor Critic

    Attributes:
        critic1 (BaseModel): critic network
        target_critic1 (BaseModel): target network for critic1
        critic2 (BaseModel): second critic network to reduce overestimation
        target_critic2 (BaseModel): target network for critic2
        critic1_optimizer (torch.Optimizer): critic1 optimizer
        critic2_optimizer (torch.Optimizer): critic2 optimizer
        critic loss_fn (Loss): critic loss function
        actor (BaseModel): actor network
        actor_optimizer (torch.Optimizer): actor optimizer
        actor loss_fn (Loss): actor loss function
        use_per (bool): indicatation of using prioritized experience replay
        update_step (int): counter for update step
        alpha (torch.Tensor): entropy temperature
        target_entropy (torch.Tensor): target for entropy temperature
        log_alpha (torch.Tensor): log(alpha)
        alpha_optimizer: optimizer for entropy temperature
        use_per (bool): indicatation of using prioritized experience replay
        update_step (int): counter for update step

    """

    def __init__(
        self,
        experiment_info: DictConfig,
        hyper_params: DictConfig,
        model_cfg: DictConfig,
    ):
        Learner.__init__(self, experiment_info, hyper_params, model_cfg)
        self.use_per = self.hyper_params.use_per
        self.update_step = 0

        self._initialize()

    def _initialize(self):
        """initialize networks, optimizer, loss function, alpha (entropy temperature)"""
        # Set env-specific input dims and output dims for models
        self.model_cfg.critic.params.model_cfg.state_dim = (
            self.model_cfg.actor.params.model_cfg.state_dim
        ) = self.experiment_info.env.state_dim
        self.model_cfg.critic.params.model_cfg.action_dim = (
            self.model_cfg.actor.params.model_cfg.action_dim
        ) = self.experiment_info.env.action_dim

        # Initialize critic models, optimizers, and loss function
        self.critic1 = build_model(self.model_cfg.critic, self.device)
        self.target_critic1 = build_model(self.model_cfg.critic, self.device)
        self.critic2 = build_model(self.model_cfg.critic, self.device)
        self.target_critic2 = build_model(self.model_cfg.critic, self.device)
        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(), lr=self.hyper_params.critic_learning_rate
        )
        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(), lr=self.hyper_params.critic_learning_rate
        )
        self.critic_loss_fn = build_loss(
            self.experiment_info.critic_loss,
            self.hyper_params,
            self.experiment_info.device,
        )

        hard_update(self.critic1, self.target_critic1)
        hard_update(self.critic2, self.target_critic2)

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

        # entropy temperature
        self.alpha = self.hyper_params.alpha
        self.target_entropy = -torch.prod(
            torch.Tensor(self.experiment_info.env.action_dim).to(self.device)
        ).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam(
            [self.log_alpha], lr=self.hyper_params.alpha_learning_rate
        )

    def update_model(
        self, experience: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        if self.use_per:
            indices, weights = experience[-2:]
            experience = experience[0:-2]

        # Compute value loss
        critic1_loss_element_wise, critic2_loss_element_wise = self.critic_loss_fn(
            (
                self.critic1,
                self.target_critic1,
                self.critic2,
                self.target_critic2,
                self.actor,
            ),
            self.alpha,
            experience,
            self.hyper_params,
        )

        if self.use_per:
            critic1_loss = (critic1_loss_element_wise * weights).mean()
            critic2_loss = (critic2_loss_element_wise * weights).mean()
        else:
            critic1_loss = critic1_loss_element_wise.mean()
            critic2_loss = critic2_loss_element_wise.mean()

        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        clip_grad_norm_(
            self.critic1.parameters(), self.hyper_params.critic_gradient_clip
        )
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        clip_grad_norm_(
            self.critic2.parameters(), self.hyper_params.critic_gradient_clip
        )
        self.critic2_optimizer.step()

        soft_update(self.critic1, self.target_critic1, self.hyper_params.tau)
        soft_update(self.critic2, self.target_critic2, self.hyper_params.tau)

        # Compute actor (policy) loss
        actor_loss, log_pi = self.actor_loss_fn(
            (self.critic1, self.critic2, self.actor), self.alpha, experience,
        )

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor.parameters(), self.hyper_params.actor_gradient_clip)
        self.actor_optimizer.step()

        # Update alpha
        alpha_loss = (self.log_alpha * (-log_pi - self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        info = (critic1_loss, critic2_loss, actor_loss, alpha_loss)
        if self.use_per:
            new_priorities = torch.clamp(critic1_loss_element_wise.view(-1), min=1e-6)
            new_priorities = new_priorities.cpu().detach().numpy()
            info = info + (indices, new_priorities,)

        return info

    def get_policy(self, target_device: torch.device):
        policy_copy = deepcopy(self.actor)
        policy_copy.to(target_device)
        return policy_copy
