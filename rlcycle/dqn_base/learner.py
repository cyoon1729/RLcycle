from typing import Tuple

import torch
import torch.optim as optim
from rlcycle.build import build_loss, build_model
from rlcycle.common.abstract.learner import Learner
from rlcycle.common.utils.common_utils import hard_update, soft_update
from torch.nn.utils import clip_grad_norm_


class DQNLearner(Learner):
    def __init__(self, args: dict, hyper_params: dict, model_cfg: dict):
        Learner.__init__(args, hyper_params, model_cfg)
        self.use_per = self.hyper_params.use_per
        self.update_step = 0

        self._initialize()

    def _initialize(self):
        """initialize networks, optimizer, loss function"""
        self.network = build_model(self.args, self.hyper_params, self.model_cfg)
        self.target_network = build_model(self.args, self.hyper_params, self.model_cfg)
        hard_update(self.network, self.target_network)
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=self.hyper_params.learning_rate
        )
        self.loss_fn = build_loss(args, hyper_params)

    def update_model(
        self, experience: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:

        # Separate indices and weights from experience if using PER
        if self.use_per:
            indices, weights = experience[-3:-1]
            experience = experience[0:5]

        q_loss, q_vals = self.loss_fn(
            (self.network, self.target_network),
            self.optimizer,
            experience,
            self.hyper_params,
        )

        # Compute new priorities and correct importance sampling bias
        if self.use_per:
            new_priorities = q_loss.detach().cpu().numpy()
            q_loss = (q_loss * weights).mean()

        dqn_reg = torch.norm(q_loss, 2).mean() * self.hyper_params.q_regularization
        loss = q_loss + dqn_reg

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.network.parameters, self.hyper_params.gradient_clip)
        self.optimizer.step()

        soft_update(self.network, self.target_network, self.hyper_params.tau)

        info = (loss,)
        if self.use_per:
            info = info + (indices.cpu().numpy(), new_priorities,)

        return info
