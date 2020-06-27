from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from rlcycle.common.abstract.action_selector import ActionSelector
from rlcycle.common.utils.common_utils import np2tensor


class DDPGActionSelector(ActionSelector):
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

    def __call__(self, policy: nn.Module, state: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Generate action via policy"""
        if state.ndim == 1:
            state = state.reshape(1, -1)
        action = policy.forward(np2tensor(state, self.device))
        action_np = action.cpu().detach().view(-1).numpy()
        return action_np

    def rescale_action(self, action: np.ndarray):
        """Rescale actions to fit continuous action spaces"""
        action_rescaled = (
            action * (self.action_max - self.action_min) / 2.0
            + (self.action_max + self.action_min) / 2.0
        )
        return action_rescaled


# Ornstein-Ulhenbeck Noise
# Adpated from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(ActionSelector):
    """Ornstein-Ulhenbeck noise wrapper to wrap deterministic continuous policies"""

    def __init__(
        self,
        action_selector: ActionSelector,
        action_space: list,
        mu: float = 0.0,
        theta: float = 0.15,
        max_sigma: float = 0.3,
        min_sigma: float = 0.3,
        decay_period: int = 100000,
    ):
        self.action_selector = action_selector
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high

        self.exploration = True

        self._reset()

    def _reset(self):
        """Reset"""
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        """Evolve Ornstein-Ulhenbeck noise state"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def __call__(
        self, policy: nn.Module, state: np.ndarray, t: float = 0.0
    ) -> Tuple[np.ndarray, ...]:
        """Add Ornstein-Ulhenbeck Noise to generated action"""
        action = self.action_selector(policy, state)
        if not self.exploration:
            return action

        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, t / self.decay_period
        )
        return np.clip(action + ou_state, self.low, self.high)
