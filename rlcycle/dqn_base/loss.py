from typing import Tuple, List

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlcycle.common.abstract.loss import Loss


class DQNLoss(Loss):
    """Compute double DQN loss"""

    def __init__(self, hyper_params: DictConfig, device: torch.device):
        Loss.__init__(self, hyper_params, device)

    def __call__(
        self, networks: Tuple[nn.Module, ...], data: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        dqn, target_dqn = networks

        states, actions, rewards, next_states, dones = data
        q_value = dqn.forward(states).gather(1, actions)
        next_q = torch.max(target_dqn.forward(next_states), 1)[0].unsqueeze(1)

        n_step_gamma = self.hyper_params.gamma ** self.hyper_params.n_step
        target_q = rewards + (1 - dones) * n_step_gamma * next_q

        element_wise_loss = F.smooth_l1_loss(
            q_value, target_q.detach(), reduction="none"
        )
        return element_wise_loss


class QRLoss(Loss):
    """Compute quantile regression loss"""

    def __init__(self, hyper_params: DictConfig, device: torch.device):
        Loss.__init__(self, hyper_params, device)

    def __call__(
        self, networks: Tuple[nn.Module, ...], data: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        qrdqn, target_qrdqn = networks
        states, actions, rewards, next_states, dones = data

        z_dists = qrdqn.forward(states)
        z_dists = z_dists[list(range(states.size(0))), actions.view(-1)]

        with torch.no_grad():
            next_z = target_qrdqn.forward(next_states)
            next_actions = torch.max(next_z.mean(2), dim=1)[1]
            next_z_max = next_z[list(range(states.size(0))), next_actions]

            n_step_gamma = self.hyper_params.gamma ** self.hyper_params.n_step
            target_z = rewards + (1 - dones) * n_step_gamma * next_z_max

        distance = target_z - z_dists
        element_wise_loss = torch.mean(
                self.quantile_huber_loss(distance) * (qrdqn.tau - (distance.detach() < 0).float()).abs(),
                dim=1
        )

        return element_wise_loss

    @staticmethod
    def quantile_huber_loss(x: List[torch.Tensor], k: float = 1.0):
        return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


class C51Loss(Loss):
    """Compute C51 loss"""

    def __init__(self, hyper_params: DictConfig, device: torch.device):
        Loss.__init__(self, hyper_params, device)

    def __call__(
        self, networks: Tuple[nn.Module, ...], data: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        pass
