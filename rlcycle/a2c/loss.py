from typing import Tuple

import torch
from omegaconf import DictConfig
from rlcycle.common.abstract.loss import Loss
from rlcycle.common.models.base import BaseModel


class DiscreteCriticLoss(Loss):
    """Copmute critic loss for softmax-ed policy in discrete action space."""

    def __init__(self, hyper_params: DictConfig, device: torch.device):
        Loss.__init__(self, hyper_params, device)

    def __call__(
        self, networks: Tuple[BaseModel, ...], data: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        pass


class GaussianCriticLoss(Loss):
    """Compute critic loss for gaussian policy in continuous action space."""

    def __init__(self, hyper_params: DictConfig, device: torch.device):
        Loss.__init__(self, hyper_params, device)

    def __call__(
        self, networks: Tuple[BaseModel, ...], data: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        pass


class DiscreteActorLoss(Loss):
    """Copmute actor loss for softmax-ed policy in discrete action space."""

    def __init__(self, hyper_params: DictConfig, device: torch.device):
        Loss.__init__(self, hyper_params, device)

    def __call__(
        self, networks: Tuple[BaseModel, ...], data: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        pass


class GaussianActorLoss(Loss):
    """Compute actor loss for gaussian policy in continuous action space."""

    def __init__(self, hyper_params: DictConfig, device: torch.device):
        Loss.__init__(self, hyper_params, device)

    def __call__(
        self, networks: Tuple[BaseModel, ...], data: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        pass
