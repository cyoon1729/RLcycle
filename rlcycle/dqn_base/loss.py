from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rlcycle.common.abstract.loss import Loss
from rlcycle.common.utils.common_utils import soft_update


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
        next_q = torch.max(target_dqn.forward(next_states), 1)[0].view(-1, 1)
        target_q = rewards + (1 - dones) * hyper_params.gamma * next_q

        loss = F.smooth_l1_loss(q_value, target_q.detach(), reduction='none')
        return loss


class QRLoss(Loss):
    def __call__(
        self,
        networks: Tuple[nn.Module, ...],
        data: Tuple[np.ndarray, ...],
        hyper_params: dict,
    ) -> Tuple[Any, ...]:
        pass


class C51Loss(Loss):
    def __call__(
        self,
        networks: Tuple[nn.Module, ...],
        data: Tuple[np.ndarray, ...],
        hyper_params: dict,
    ) -> Tuple[Any, ...]:
        pass
