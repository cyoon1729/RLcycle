from copy import deepcopy
from typing import Tuple

import torch
import torch.optim as optim
from omegaconf import DictConfig
from rlcycle.build import build_loss, build_model
from rlcycle.common.abstract.learner import Learner
from rlcycle.common.utils.common_utils import hard_update, soft_update
from torch.nn.utils import clip_grad_norm_


class DQNLearner(Learner):
    """Learner for DQN base agent

    Attributes:
        network (BaseModel): a dqn-based network
        target_network (BaseModel): target network to network, for unbiased estimates
        optimizer (torch.Optimizer): network optimizer, using ADAM
        loss_fn (Loss): loss function specific to dqn network architecture
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
        """initialize networks, optimizer, loss function"""
        self.network = build_model(self.model_cfg, self.device)
        self.target_network = build_model(self.model_cfg, self.device)
        hard_update(self.network, self.target_network)

        self.optimizer = optim.Adam(
            self.network.parameters(), lr=self.hyper_params.learning_rate
        )

        self.loss_fn = build_loss(
            self.experiment_info.loss, self.hyper_params, self.experiment_info.device
        )

    def update_model(
        self, experience: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:

        # Separate indices and weights from experience if using PER
        if self.use_per:
            indices, weights = experience[-2:]
            experience = experience[:-2]

        q_loss_element_wise = self.loss_fn(
            (self.network, self.target_network), experience
        )

        # Compute new priorities and correct importance sampling bias
        if self.use_per:
            q_loss = torch.mean((q_loss_element_wise * weights))
        else:
            q_loss = torch.mean(q_loss_element_wise)

        dqn_reg = torch.norm(q_loss, 2).mean() * self.hyper_params.q_reg_coeff
        loss = q_loss + dqn_reg

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.network.parameters(), self.hyper_params.gradient_clip)
        self.optimizer.step()

        soft_update(self.network, self.target_network, self.hyper_params.tau)

        info = (q_loss,)
        if self.use_per:
            new_priorities = torch.clamp(q_loss_element_wise.view(-1), min=1e-6)
            new_priorities = new_priorities.cpu().detach().numpy()
            info = info + (indices, new_priorities,)

        return info

    def get_policy(self, target_device: torch.device):
        """Return policy mapped to target device"""
        policy_copy = deepcopy(self.network)
        policy_copy.to(target_device)
        return policy_copy
