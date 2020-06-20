from typing import Tuple

import torch
import torch.optim as optim
from omegaconf import DictConfig
from rlcycle.build import build_loss, build_model
from rlcycle.common.abstract.learner import Learner
from rlcycle.common.utils.common_utils import hard_update, soft_update
from torch.nn.utils import clip_grad_norm_


class DDPGLearner(Learner):
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
        """initialize networks, optimizer, loss function"""
        self.critic = build_model(self.model_cfg.critic, self.device)
        self.target_critic = build_model(self.model_cfg.critic, self.device)
        hard_update(self.critic, self.target_critic)
        self.actor = build_model(self.model_cfg.actor, self.device)

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.hyper_params.critic_learning_rate
        )
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.hyper_params.actor_learning_rate
        )

        self.critic_loss_fn = build_loss(self.experiment_info)

    def update_model(
        self, experience: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:

        # Separate indices and weights from experience if using PER
        if self.use_per:
            indices, weights = experience[-2:]
            experience = experience[0:-2]

        critic_loss_element_wise = self.critic_loss_fn(
            (self.critic, self.target_critic), experience, self.hyper_params,
        )

        # Compute new priorities and correct importance sampling bias
        if self.use_per:
            critic_loss = (critic_loss_element_wise * weights).mean()
        else:
            critic_loss = critic_loss_element_wise.mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), self.hyper_params.gradient_clip)
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        clip_grad_norm_(self.actor.parameters(), self.hyper_params.gradient_clip)
        self.actor_optimizer.step()

        soft_update(self.critic, self.target_critic, self.hyper_params.tau)

        info = (loss,)
        if self.use_per:
            new_priorities = torch.clamp(q_loss_element_wise.view(-1), min=1e-6)
            new_priorities = new_priorities.cpu().detach().numpy()
            info = info + (indices, new_priorities,)

        return info
