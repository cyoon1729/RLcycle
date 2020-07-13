from typing import Tuple

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlcycle.common.abstract.loss import Loss


class CriticLoss(Loss):
    """Critic loss using clipped double q learning as in Fujimoto et al., 2018"""

    def __init__(self, hyper_params: DictConfig, use_cuda: bool):
        Loss.__init__(self, hyper_params, use_cuda)

    def __call__(
        self, networks: Tuple[nn.Module, ...], data: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        critic1, critic1_target, critic2, critic2_target, actor = networks

        states, actions, rewards, next_states, dones = data

        q_value1 = critic1.forward(states, actions)
        q_value2 = critic2.forward(states, actions)

        next_actions = actor.forward(states) + self._generate_action_space_noise()
        next_q1 = critic1_target.forward(next_states, next_actions)
        next_q2 = critic2_target.forward(next_states, next_actions)
        next_q = torch.min(next_q1, next_q2)

        n_step_gamma = self.hyper_params.gamma ** self.hyper_params.n_step
        target_q = rewards + (1 - dones) * n_step_gamma * next_q

        # q loss
        element_wise_q1_loss = F.mse_loss(q_value1, target_q.detach(), reduction="none")
        element_wise_q2_loss = F.mse_loss(q_value2, target_q.detach(), reduction="none")

        return element_wise_q1_loss, element_wise_q2_loss

    def _generate_action_space_noise(self) -> torch.Tensor:
        """Generate action space noise"""
        if self.hyper_params.use_policy_reg:
            zeros = torch.zeros(self.hyper_params.batch_size, 1)
            noise = torch.normal(zeros, self.hyper_params.noise_std)
            noise = torch.clamp(
                noise,
                min=-self.hyper_params.policy_noise_bound,
                max=self.hyper_params.policy_noise_bound,
            )
            if self.use_cuda:
                return noise.cuda()
            else:
                return noise.cpu()
        return 0


class ActorLoss(Loss):
    """Compute DDPG actor loss"""

    def __init__(self, hyper_params: DictConfig, use_cuda: bool):
        Loss.__init__(self, hyper_params, use_cuda)

    def __call__(
        self, networks: Tuple[nn.Module, ...], data: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        critic1, actor = networks

        states, _, _, _, _ = data
        actions = actor.forward(states)

        policy_gradient = -critic1.forward(states, actions).mean()

        return policy_gradient
