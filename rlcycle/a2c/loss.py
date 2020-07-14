from typing import Tuple

from omegaconf import DictConfig
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

from rlcycle.common.abstract.loss import Loss
from rlcycle.common.models.base import BaseModel


class DiscreteCriticLoss(Loss):
    """Copmute critic loss for softmax-ed policy in discrete action space."""

    def __init__(self, hyper_params: DictConfig, use_cuda: bool):
        Loss.__init__(self, hyper_params, use_cuda)

    def __call__(
        self, networks: Tuple[BaseModel, ...], data: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        critic = networks
        states, _, rewards = data

        # compute current values
        values = critic.forward(states)

        # Compute value targets
        value_targets = torch.zeros_like(rewards)
        if self.use_cuda:
            value_targets = value_targets.cuda(non_blocking=True)

        for t in reversed(range(rewards.size(0) - 1)):
            value_targets[t] = rewards[t] + self.hyper_params.gamma * value_targets[t]

        # Compute value targets
        critic_loss = F.mse_loss(values, value_targets.detach(), reduction="none")

        return critic_loss, values


class DiscreteActorLoss(Loss):
    """Copmute actor loss for softmax-ed policy in discrete action space."""

    def __init__(self, hyper_params: DictConfig, use_cuda: bool):
        Loss.__init__(self, hyper_params, use_cuda)

    def __call__(
        self, networks: Tuple[BaseModel, ...], data: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        actor = networks
        states, actions, rewards, values = data

        # Compute advantage
        q_values = values.clone()
        for t in reversed(range(rewards.size(0) - 1)):
            q_values[t] = rewards[t] + self.hyper_params.gamma * values[t]
        advantages = q_values - values

        # Compute entropy regularization
        policy_dists = actor.forward(states)
        categorical_dists = Categorical(policy_dists)
        entropies = torch.zeros_like(actions)
        for i in range(entropies.size(0)):
            entropies[i] = -torch.sum(policy_dists[i] * torch.log(policy_dists[i]))
        entropy_bonus = entropies.mean()

        # Compute policy loss
        policy_loss = (
            -categorical_dists.log_prob(actions.view(-1)).view(-1, 1)
            * advantages.detach()
        )
        policy_loss = policy_loss.mean() - self.hyper_params.alpha * entropy_bonus

        return policy_loss
