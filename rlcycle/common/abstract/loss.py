from abc import ABC, abstractmethod
from typing import Any, Tuple, Union

import numpy as np
import torch.nn as nn
import torch.optim as optim


class Loss(ABC):
    """Abstract class for callable loss functions"""

    @abstractmethod
    def __call__(
        self,
        networks: Tuple[nn.Module, ...],
        optimizers: Tuple[optim.Optim, ...],
        data: Tuple[np.ndarray, ...],
    ) -> Tuple[Any, ...]:
        pass
