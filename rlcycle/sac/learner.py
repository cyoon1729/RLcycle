from typing import Tuple

import torch
import torch.optim as optim
from omegaconf import DictConfig
from rlcycle.build import build_loss, build_model
from rlcycle.common.abstract.learner import Learner
from rlcycle.common.utils.common_utils import hard_update, soft_update
from torch.nn.utils import clip_grad_norm_


class SACLearner(Learner):
    """Learner object for Soft Actor Critic algorithm"""

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
        # Initialize critic and related
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
        self.critic_loss_fn = build_loss(self.experiment_info.critic_loss)

        hard_update(self.critic1, self.target_critic1)
        hard_update(self.critic2, self.target_critic2)

        # Initialize actor and related
        self.actor = build_model(self.model_cfg.actor, self.device)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.hyper_params.policy_learning_rate
        )
        self.actor_loss_fn = build_loss(self.experiment_info.actor_loss)

        # entropy temperature
        self.alpha = self.hyper_params.alpha
        self.target_entropy = -torch.prod(
            torch.Tensor(self.experiment_info.env.action_space.shape).to(self.device)
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

        # Compute policy loss
        policy_loss = self.policy_loss_fn(
            (self.critic1, self.critic2, self.actor),
            self.alpha,
            experience,
            self.hyper_params,
        )

        # Update actor
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        clip_grad_norm_(self.actor.parameters(), self.hyper_params.actor_gradient_clip)
        self.actor_optimizer.step()

        # Update alpha
        alpha_loss = (self.log_alpha * (-log_pi - self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()
