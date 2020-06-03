from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch.nn as nn


class ActionSelector(ABC):
    """Abstract base class for callable action selection methods"""

    @abstractmethod
    def __call__(self, network: nn.Module, state: np.ndarray) -> Tuple[np.ndarray, ...]:
        pass
