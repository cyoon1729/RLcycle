from typing import Tuple

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlcycle.common.abstract.loss import Loss


class CriticLoss(Loss):
    """SAC critic loss as described in Haarnoja et al., 2019"""

    def __init__(self, hyper_params: DictConfig, use_cuda: bool):
        Loss.__init__(self, hyper_params, use_cuda)

    def __call__(
        self,
        networks: Tuple[nn.Module, ...],
        alpha: torch.Tensor,
        data: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        critic1, target_critic1, critic2, target_critic2, actor = networks

        states, actions, rewards, next_states, dones = data

        q_value1 = critic1.forward(states, actions)
        q_value2 = critic2.forward(states, actions)

        _, _, next_zs, next_log_pi = actor.sample(next_states)
        next_actions = torch.tanh(next_zs)
        next_q1 = target_critic1(next_states, next_actions)
        next_q2 = target_critic2(next_states, next_actions)
        next_q = torch.min(next_q1, next_q2) - alpha * next_log_pi

        n_step_gamma = self.hyper_params.gamma ** self.hyper_params.n_step
        target_q = rewards + (1 - dones) * n_step_gamma * next_q

        element_wise_q1_loss = F.mse_loss(q_value1, target_q.detach(), reduction="none")
        element_wise_q2_loss = F.mse_loss(q_value2, target_q.detach(), reduction="none")

        return element_wise_q1_loss, element_wise_q2_loss


class PolicyLoss(Loss):
    """SAC policy loss as described in Haarnoja et al., 2019"""

    def __init__(self, hyper_params: DictConfig, use_cuda: bool):
        Loss.__init__(self, hyper_params, use_cuda)

    def __call__(
        self,
        networks: Tuple[nn.Module, ...],
        alpha: torch.Tensor,
        data: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        critic1, critic2, actor = networks

        states, _, _, _, _ = data

        _, _, new_zs, log_pi = actor.sample(states)
        new_actions = torch.tanh(new_zs)
        min_q = torch.min(
            critic1.forward(states, new_actions), critic2.forward(states, new_actions)
        )
        policy_loss = (alpha * log_pi - min_q).mean()

        return policy_loss, log_pi
