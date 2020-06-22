from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig


class Loss(ABC):
    """Abstract class for callable loss functions"""

    def __init__(self, hyper_params: DictConfig, device: str):
        self.hyper_params = hyper_params
        self.device = torch.device(device)

    @abstractmethod
    def __call__(
        self, networks: Tuple[nn.Module, ...], data: Tuple[np.ndarray, ...],
    ) -> Tuple[torch.Tensor, ...]:
        pass
