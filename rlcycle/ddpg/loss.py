from typing import Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from rlcycle.common.abstract.loss import Loss


class CriticLoss(Loss):
    """Critic loss using clipped double q learning as in Fujimoto et al., 2018"""

    def __init__(self, hyper_params: DictConfig, device: torch.device):
        Loss.__init__(self, hyper_params, device)

    def __call__(
        self, networks: Tuple[nn.Module, ...], data: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        critic1, critic1_target, critic2, critic2_target, actor = networks

        states, actions, rewards, next_states, dones = data

        q1_value = critic1.forward(states, actions)
        q2_value = critic2.forward(states, actions)

        next_actions = actor.forward(states) + self._action_space_noise()
        next_q1 = critic1_target.forward(next_states, next_actions)
        next_q1 = critic2_target.forward(next_states, next_actions)

        n_step_gamma = self.hyper_params.gamma ** self.hyper_params.n_step
        target_q = rewards + n_step_gamma * torch.min(next_q1, next_q2)

        # critic loss
        element_wise_critic1_loss = F.smooth_l1_loss(q1_value, target_q.detach())
        element_wise_critic2_loss = F.smooth_l1_loss(q2_value, target_q.detach())

        return element_wise_critic1_loss, element_wise_critic2_loss

    def _action_space_noise(self) -> torch.Tensor:
        if self.hyper_params.use_policy_reg:
            zeros = torch.zeros(self.hyper_params.batch_size, 1)
            noise = torch.normal(zeros, self.hyper_params.noise_std).to(self.device)
            noise = torch.clamp(
                noise,
                min=-self.hyper_params.policy_noise_bound,
                max=self.hyper_params.policy_noise_bound,
            )
            return noise
        return 0


class ActorLoss(Loss):
    """Compute DDPG actor loss"""

    def __init__(self, hyper_params: DictConfig, device: torch.Device):
        Loss.__init__(self, hyper_params, device)

    def __call__(
        self, networks: Tuple[nn.Module, ...], data: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        critic1, actor = networks

        states, _, _, _, _ = data
        actions = actor.forward(states)

        policy_gradient = -critic1.forward(states, actions).mean()

        return policy_gradient
