from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch.nn as nn


class ActionSelector(ABC):
    """Abstract base class for callable action selection methods

    Attributes:
        use_cuda (bool): true if using gpu
        exploration (bool): turn on/off exploratory scheme
    """

    def __init__(self, use_cuda: bool):
        self.use_cuda = use_cuda
        self.exploration = True

    @abstractmethod
    def __call__(self, policy: nn.Module, state: np.ndarray) -> Tuple[np.ndarray, ...]:
        pass
