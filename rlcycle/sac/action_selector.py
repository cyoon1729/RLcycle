from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from rlcycle.common.abstract.action_selector import ActionSelector
from rlcycle.common.utils.common_utils import np2tensor


class SACActionSelector(ActionSelector):
    def __init__(self, device: torch.device):
        self.device = device

    def __call__(
        self, policy: nn.Module, state: np.ndarray
    ) -> Tuple[torch.Tensor, ...]:
        mu, sigma, z, log_pi = policy.sample(np2tensor(state, self.device))
        action = torch.tanh(z)
        action_np = action.cpu().detach().view(-1).numpy()
        return action_np
