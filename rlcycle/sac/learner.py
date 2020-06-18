from typing import Tuple

import torch
import torch.optim as optim
from omegaconf import DictConfig
from torch.nn.utils import clip_grad_norm_

from rlcycle.build import build_loss, build_model
from rlcycle.common.abstract.learner import Learner
from rlcycle.common.utils.common_utils import hard_update, soft_update


class SACLearner(Learner):
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
        self.critic1 = build_model(self.model_cfg.critic, self.device)
        self.target_critic1 = build_model(self.model_cfg.critic, self.device)
        self.critic2 = build_model(self.model_cfg.critic, self.device)
        self.target_critic2 = build_model(self.model_cfg.critic, self.device)
        hard_update(self.critic1, self.target_critic1)
        hard_update(self.critic2, self.target_critic2)

        self.policy = build_model(self.model_cfg.actor, self.device)

        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(), lr=self.hyper_params.critic_learning_rate
        )
        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(), lr=self.hyper_params.critic_learning_rate
        )
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(), lr=self.hyper_params.policy_learning_rate
        )

        self.critic_loss_fn = build_loss(self.experiment_info)

    def update_model(
        self, experience: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        if self.use_per:
            indices, weights = experience[-2:]
            experience = experience[0:-2]

        critic1_loss_element_wise, critic2_loss_element_wise = self.critic_loss_fn(
            (self.critic, self.target_critic, self.policy),
            experience,
            self.hyper_params,
        )

        if self.use_per:
            critic1_loss = (critic1_loss_element_wise * weights).mean()
            critic2_loss = (critic2_loss_element_wise * weights).mean()
        else:
            critic1_loss = critic1_loss_element_wise.mean()
            critic2_loss = critic2_loss_element_wise.mean()

        # update critic networks
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # delayed update for policy network and target q networks
        _, _, new_zs, log_pi = self.policy_net.sample(states)
        new_actions = torch.tanh(new_zs)
        min_q = torch.min(
            self.q_net1.forward(states, new_actions),
            self.q_net2.forward(states, new_actions),
        )
        policy_loss = (self.alpha * log_pi - min_q).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        soft_update(self.critic1, self.target_critic1, self.hyper_params.tau)
        soft_update(self.critic2, self.target_critic2, self.hyper_params.tau)
