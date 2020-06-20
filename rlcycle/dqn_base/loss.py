from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from rlcycle.common.abstract.loss import Loss


class DQNLoss(Loss):
    def __call__(
        self,
        networks: Tuple[nn.Module, ...],
        data: Tuple[torch.Tensor, ...],
        hyper_params: dict,
    ) -> Tuple[Any, ...]:
        dqn, target_dqn = networks

        states, actions, rewards, next_states, dones = data
        q_value = dqn.forward(states).gather(1, actions)
        next_q = torch.max(target_dqn.forward(next_states), 1)[0].unsqueeze(1)

        n_step_gamma = hyper_params.gamma ** hyper_params.n_step
        target_q = rewards + (1 - dones) * n_step_gamma * next_q

        element_wise_loss = F.smooth_l1_loss(
            q_value, target_q.detach(), reduction="none"
        )
        return element_wise_loss


class QRLoss(Loss):
    def __call__(
        self,
        networks: Tuple[nn.Module, ...],
        data: Tuple[torch.Tensor, ...],
        hyper_params: dict,
    ) -> Tuple[Any, ...]:
        dqn, target_dqn = networks
        states, actions, rewards, next_states, dones = data

        q_value = dqn.forward(states).gather(1, actions)
        next_q = target_dqn.forward(next_states)

        n_step_gamma = hyper_params.gamma ** hyper_params.n_step
        target_q = rewards + (1 - dones) * n_step_gamma * next_q

        distance = q_targets.detach() - curr_q
        element_wise_loss = F.smooth_l1_loss(
            curr_q, q_targets.detach(), reduction="none"
        ) * torch.abs((dqn.tau - (distance.detach() < 0).float()))

        return element_wise_loss


class C51Loss(Loss):
    def __call__(
        self,
        networks: Tuple[nn.Module, ...],
        data: Tuple[torch.Tensor, ...],
        hyper_params: dict,
    ) -> Tuple[Any, ...]:
        pass
