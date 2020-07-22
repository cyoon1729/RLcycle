from typing import List, Tuple

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlcycle.common.abstract.loss import Loss


class DQNLoss(Loss):
    """Compute double DQN loss"""

    def __init__(self, hyper_params: DictConfig, use_cuda: bool):
        Loss.__init__(self, hyper_params, use_cuda)

    def __call__(
        self, networks: Tuple[nn.Module, ...], data: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        network, target_network = networks
        states, actions, rewards, next_states, dones = data

        q_value = network.forward(states).gather(1, actions)

        with torch.no_grad():
            next_q = torch.max(target_network.forward(next_states), 1)[0].unsqueeze(1)
            n_step_gamma = self.hyper_params.gamma ** self.hyper_params.n_step
            target_q = rewards + (1 - dones) * n_step_gamma * next_q

        element_wise_loss = F.smooth_l1_loss(
            q_value, target_q.detach(), reduction="none"
        )

        return element_wise_loss


class QRLoss(Loss):
    """Compute quantile regression loss"""

    def __init__(self, hyper_params: DictConfig, use_cuda: bool):
        Loss.__init__(self, hyper_params, use_cuda)

    def __call__(
        self, networks: Tuple[nn.Module, ...], data: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        network, target_network = networks
        states, actions, rewards, next_states, dones = data

        z_dists = network.forward(states)
        z_dists = z_dists[list(range(states.size(0))), actions.view(-1)]

        with torch.no_grad():
            next_z = target_network.forward(next_states)
            next_actions = torch.max(next_z.mean(2), dim=1)[1]
            next_z = next_z[list(range(states.size(0))), next_actions]
            n_step_gamma = self.hyper_params.gamma ** self.hyper_params.n_step
            target_z = rewards + (1 - dones) * n_step_gamma * next_z

        distance = target_z - z_dists
        quantile_huber_loss = (
            network.tau - (distance.detach() < 0).float()
        ).abs() * self.huber_loss(distance)
        element_wise_loss = torch.mean(quantile_huber_loss, dim=1, keepdim=True)

        return element_wise_loss

    @staticmethod
    def huber_loss(x: List[torch.Tensor], k: float = 1.0):
        return torch.where(x.abs() <= k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


class CategoricalLoss(Loss):
    """Compute C51 loss"""

    def __init__(self, hyper_params: DictConfig, use_cuda: bool):
        Loss.__init__(self, hyper_params, use_cuda)

    def __call__(
        self, networks: Tuple[nn.Module, ...], data: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        network, target_network = networks
        states, actions, rewards, next_states, dones = data
        batch_size = states.size(0)
        offset = (
            torch.linspace(0, (batch_size - 1) * network.num_atoms, batch_size)
            .long()
            .unsqueeze(1)
            .expand(batch_size, network.num_atoms)
        )
        if self.use_cuda:
            offset = offset.cuda()

        z_dists = network.forward(states)
        z_dists = z_dists[list(range(states.size(0))), actions.view(-1)]

        with torch.no_grad():
            next_z = target_network.forward(next_states)
            next_actions = torch.max(next_z.mean(2), dim=1)[1]
            next_z = next_z[list(range(states.size(0))), next_actions]

            n_step_gamma = self.hyper_params.gamma ** self.hyper_params.n_step
            target_z = rewards + (1 - dones) * n_step_gamma * network.support
            target_z = torch.clamp(target_z, min=network.v_min, max=network.v_max)
            target_proj = self.dist_projection(network, next_z, target_z, offset)

        log_dist = torch.log(z_dists)
        element_wise_loss = -(target_proj * log_dist).sum(1)

        return element_wise_loss

    def dist_projection(
        self,
        network: nn.Module,
        next_z: torch.Tensor,
        target_z: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        b = (target_z - network.v_min) / network.delta_z
        lb = b.floor().long()
        ub = b.ceil().long()

        proj_dist = torch.zeros(next_z.size())
        if self.use_cuda:
            proj_dist = proj_dist.cuda()

        proj_dist.view(-1).index_add_(
            0, (lb + offset).view(-1), (next_z * (ub.float() - b)).view(-1)
        )
        proj_dist.view(-1).index_add_(
            0, (ub + offset).view(-1), (next_z * (b - lb.float())).view(-1)
        )

        return proj_dist
