from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn as nn


class Loss(ABC):
    """Abstract class for callable loss functions

    Attributes:
        hyper_params (DictConfig): algorithm hyperparameters
        use_cuda (bool): true if using gpu

    """

    def __init__(self, hyper_params: DictConfig, use_cuda: bool):
        self.hyper_params = hyper_params
        self.use_cuda = use_cuda

    @abstractmethod
    def __call__(
        self, networks: Tuple[nn.Module, ...], data: Tuple[np.ndarray, ...],
    ) -> Tuple[torch.Tensor, ...]:
        pass
