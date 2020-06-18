from typing import Tuple

import torch
import torch.optim as optim
from omegaconf import DictConfig
from torch.nn.utils import clip_grad_norm_

from rlcycle.build import build_loss, build_model
from rlcycle.common.abstract.learner import Learner
from rlcycle.common.utils.common_utils import hard_update, soft_update


class DQNLearner(Learner):
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
        self.network = build_model(self.model_cfg, self.device)
        self.target_network = build_model(self.model_cfg, self.device)
        hard_update(self.network, self.target_network)

        self.optimizer = optim.Adam(
            self.network.parameters(), lr=self.hyper_params.learning_rate
        )

        self.loss_fn = build_loss(self.experiment_info)

    def update_model(
        self, experience: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:

        # Separate indices and weights from experience if using PER
        if self.use_per:
            indices, weights = experience[-2:]
            experience = experience[0:-2]

        q_loss_element_wise = self.loss_fn(
            (self.network, self.target_network), experience, self.hyper_params,
        )

        # Compute new priorities and correct importance sampling bias
        if self.use_per:
            q_loss = (q_loss_element_wise * weights).mean()
        else:
            q_loss = q_loss_element_wise.mean()

        dqn_reg = torch.norm(q_loss, 2).mean() * self.hyper_params.q_reg_coeff
        loss = q_loss + dqn_reg

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.network.parameters(), self.hyper_params.gradient_clip)
        self.optimizer.step()

        soft_update(self.network, self.target_network, self.hyper_params.tau)

        info = (loss,)
        if self.use_per:
            new_priorities = torch.clamp(q_loss_element_wise.view(-1), min=1e-6)
            new_priorities = new_priorities.cpu().detach().numpy()
            info = info + (indices, new_priorities,)

        return info
