from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from rlcycle.common.abstract.action_selector import ActionSelector
from rlcycle.common.utils.common_utils import np2tensor


class SACActionSelector(ActionSelector):
    """Action selector for (vanilla) DDPG policy

    Attributes:
        device (torch.device): map location for tensor computations
        action_min (np.ndarray): lower bound for continuous actions
        action_max (np.ndarray): upper bound for continuous actions

    """

    def __init__(self, action_range: list, device: str):
        self.device = torch.device(device)
        self.action_min = np.array(action_range[0])
        self.action_max = np.array(action_range[1])

    def __call__(
        self, policy: nn.Module, state: np.ndarray
    ) -> Tuple[torch.Tensor, ...]:
        """Generate action via policy"""
        if state.ndim == 1:
            state = state.reshape(1, -1)
        mu, sigma, z, log_pi = policy.sample(np2tensor(state, self.device))
        action = torch.tanh(z)
        action_np = action.cpu().detach().view(-1).numpy()
        return action_np

    def rescale_action(self, action: np.ndarray):
        """Rescale actions to fit continuous action spaces"""
        action_rescaled = (
            action * (self.action_max - self.action_min) / 2.0
            + (self.action_max + self.action_min) / 2.0
        )
        return action_rescaled
