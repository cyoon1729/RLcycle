from typing import Tuple

import torch
import torch.optim as optim
from rlcycle.common.abstract.learner import Learner
from rlcycle.common.build import build_loss, build_model
from rlcycle.common.utils.common_utils import hard_update, soft_update
from torch.nn.utils import clip_grad_norm_


class DQNLearner(Learner):
    def __init__(self, args: dict, hyper_params: dict, model_cfg: dict):
        Learner.__init__(args, hyper_params, model_cfg)

        self._initialize()

    def _initialize(self):
        """initialize networks, optimizer, loss function"""
        self.network = build_model(self.args, self.hyper_params, self.model_cfg)
        self.target_network = build_model(self.args, self.hyper_params, self.model_cfg)
        hard_update(self.network, self.target_network)
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=self.hyper_params["dqn_learning_rate"]
        )
        self.loss_fn = build_loss(args, hyper_params)

    def update_model(
        self, experience: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        q_loss = self.loss_fn(
            (self.network, self.target_network),
            self.optimizer,
            experience,
            self.hyper_params,
        )

        dqn_reg = torch.norm(q_loss, 2).mean() * self.hyper_params["q_regularization"]
        loss = q_loss + dqn_reg

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.network.parameters, self.hyper_params["gradient_clip"])
        self.optimizer.step()

        soft_update(self.network, self.target_network, self.hyper_params["tau"])

        return loss


class PERLearner(DQNLearner):
    def __init__(self, dqn_learner: DQNLearner):
        self.learner = dqn_learner

    def update_model(
        self, experience: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        indices, weights = experience[-3:-1]
        indices = indices.detach().cpu().numpy()
        experience = experience[0:6]

        q_loss = self.learner.loss_fn(
            (self.learner.network, self.learner.target_network),
            self.learner.optimizer,
            experience,
            self.learner.hyper_params,
        )
        weighted_q_loss = torch.mean(q_loss * weights)

        dqn_reg = (
            torch.norm(weighted_q_loss, 2).mean()
            * self.learner.hyper_params["q_regularization"]
        )
        loss = weighted_q_loss + dqn_reg

        self.learner.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(
            self.learner.network.parameters, self.hyper_params["gradient_clip"]
        )
        self.learner.optimizer.step()

        soft_update(
            self.learner.network,
            self.learner.target_network,
            self.learner.hyper_params["tau"],
        )
        new_priorities = q_loss.detach().cpu().numpy() + 1e-6

        return loss, indices, new_priorities
