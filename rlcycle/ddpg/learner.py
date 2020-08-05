from copy import deepcopy
import os
from typing import Tuple

from omegaconf import DictConfig
import torch
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

from rlcycle.build import build_loss, build_model
from rlcycle.common.abstract.learner import Learner
from rlcycle.common.models.base import BaseModel
from rlcycle.common.utils.common_utils import hard_update, soft_update


class DDPGLearner(Learner):
    """Learner for DDPG Agent

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
        """Initialize networks, optimizer, loss function."""
        # Set env-specific input dims and output dims for models
        self.model_cfg.critic.params.model_cfg.state_dim = (
            self.model_cfg.actor.params.model_cfg.state_dim
        ) = self.experiment_info.env.state_dim
        self.model_cfg.critic.params.model_cfg.action_dim = (
            self.model_cfg.actor.params.model_cfg.action_dim
        ) = self.experiment_info.env.action_dim

        # Initialize critic models, optimizers, and loss function
        self.critic1 = build_model(self.model_cfg.critic, self.use_cuda)
        self.target_critic1 = build_model(self.model_cfg.critic, self.use_cuda)
        self.critic2 = build_model(self.model_cfg.critic, self.use_cuda)
        self.target_critic2 = build_model(self.model_cfg.critic, self.use_cuda)

        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(), lr=self.hyper_params.critic_learning_rate
        )
        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(), lr=self.hyper_params.critic_learning_rate
        )
        self.critic_loss_fn = build_loss(
            self.experiment_info.critic_loss, self.hyper_params, self.use_cuda,
        )

        hard_update(self.critic1, self.target_critic1)
        hard_update(self.critic2, self.target_critic2)

        # Initialize actor model, optimizer, and loss function
        self.actor = build_model(self.model_cfg.actor, self.use_cuda)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.hyper_params.actor_learning_rate
        )
        self.actor_loss_fn = build_loss(
            self.experiment_info.actor_loss, self.hyper_params, self.use_cuda,
        )

    def update_model(
        self, experience: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:

        # Separate indices and weights from experience if using PER
        if self.use_per:
            indices, weights = experience[-2:]
            experience = experience[0:-2]

        # Compute critic loss
        critic1_loss_element_wise, critic2_loss_element_wise = self.critic_loss_fn(
            (
                self.critic1,
                self.target_critic1,
                self.critic2,
                self.target_critic2,
                self.actor,
            ),
            experience,
        )

        # Compute new priorities and correct importance sampling bias
        if self.use_per:
            critic1_loss = (critic1_loss_element_wise * weights).mean()
            critic2_loss = (critic2_loss_element_wise * weights).mean()
        else:
            critic1_loss = critic1_loss_element_wise.mean()
            critic2_loss = critic2_loss_element_wise.mean()

        # Update critic networks
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        clip_grad_norm_(
            self.critic1.parameters(), self.hyper_params.critic_gradient_clip
        )
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        clip_grad_norm_(
            self.critic1.parameters(), self.hyper_params.critic_gradient_clip
        )
        self.critic2_optimizer.step()

        soft_update(self.critic1, self.target_critic1, self.hyper_params.tau)
        soft_update(self.critic2, self.target_critic2, self.hyper_params.tau)

        # Compute actor loss
        policy_loss = self.actor_loss_fn((self.critic1, self.actor), experience,)

        # Update actor network
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        clip_grad_norm_(self.actor.parameters(), self.hyper_params.actor_gradient_clip)
        self.actor_optimizer.step()

        critic1_loss = float(critic1_loss.detach().cpu().item())
        critic2_loss = float(critic2_loss.detach().cpu().item())
        policy_loss = float(policy_loss.detach().cpu().item())
        info = (
            critic1_loss,
            critic2_loss,
            policy_loss,
        )

        if self.use_per:
            new_priorities = torch.clamp(critic1_loss_element_wise.view(-1), min=1e-6)
            new_priorities = new_priorities.cpu().detach().numpy()
            info = info + (indices, new_priorities,)

        return info

    def get_policy(self, to_cuda: bool) -> BaseModel:
        policy_copy = deepcopy(self.actor)
        if to_cuda:
            return policy_copy.cuda()
        else:
            return policy_copy.cpu()

    def save_params(self):
        ckpt = self.ckpt_path + f"/update-step-{self.update_step}"
        os.makedirs(ckpt, exist_ok=True)
        path = os.path.join(ckpt + ".pt")

        torch.save(self.critic1.state_dict(), path)
        torch.save(self.critic2.state_dict(), path)
        torch.save(self.actor.state_dict(), path)
        torch.save(self.critic1_optimizer.state_dict(), path)
        torch.save(self.critic2_optimizer.state_dict(), path)
        torch.save(self.actor_optimizer.state_dict(), path)
