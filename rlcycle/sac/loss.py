from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from rlcycle.common.abstract.loss import Loss


class CriticLoss(Loss):
    """SAC critic loss as described in Haarnoja et al., 2019"""

    def __init__(self, hyper_params: DictConfig, device: torch.device):
        Loss.__init__(self, hyper_params, device)

    def __call__(
        self,
        networks: Tuple[nn.Module, ...],
        alpha: torch.Tensor,
        data: Tuple[torch.Tensor, ...],
        hyper_params: DictConfig,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        critic1, target_critic1, critic2, target_critic2, actor = networks

        states, actions, rewards, next_states, dones = data

        q_value1 = critic1.forward(states, actions)
        q_value2 = critic2.forward(states, actions)

        _, _, next_zs, next_log_pi = actor.sample(next_states)
        next_actions = torch.tanh(next_zs)
        next_q1 = target_critic1(next_states, next_actions)
        next_q2 = target_critic2(next_states, next_actions)
        target_q = torch.min(next_q1, next_q2) - alpha * next_log_pi

        n_step_gamma = self.hyper_params.gamma ** self.hyper_params.n_step
        expected_q = rewards + (1 - dones) * n_step_gamma * target_q

        # q loss
        element_wise_q1_loss = F.smooth_l1_loss(
            q_value1, expected_q.detach(), reduction="none"
        )
        element_wise_q2_loss = F.smooth_l1_loss(
            q_value2, expected_q.detach(), reduction="none"
        )

        return element_wise_q1_loss, element_wise_q2_loss


class PolicyLoss(Loss):
    """SAC policy loss as described in Haarnoja et al., 2019"""

    def __init__(self, hyper_params: DictConfig, device: torch.device):
        Loss.__init__(self, hyper_params, device)

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
