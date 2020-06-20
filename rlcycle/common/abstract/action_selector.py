from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class ActionSelector(ABC):
    """Abstract base class for callable action selection methods"""

    def __init__(self, device: str):
        self.device = torch.device(device)
        self.exploration = True

    @abstractmethod
    def __call__(self, policy: nn.Module, state: np.ndarray) -> Tuple[np.ndarray, ...]:
        pass
