from typing import Tuple

import numpy as np
import torch.nn as nn

from rlcycle.common.abstract.action_selector import ActionSelector
from rlcycle.common.utils.common_utils import np2tensor


class RandomActionsStarts(ActionSelector):
    """Wrapper that outputs random actions for defined number of inittial steps"""

    def __init__(
        self, action_selector: ActionSelector, max_exploratory_steps: int = 1000
    ):
        ActionSelector.__init__(self, action_selector.use_cuda)
        self.action_selector = action_selector
        self.exploration = self.action_selector.exploration
        self.max_exploratory_steps = max_exploratory_steps

    def __call__(
        self, policy: nn.Module, state: np.ndarray, episode_num: int = 99999
    ) -> Tuple[np.ndarray, ...]:
        if episode_num < self.max_exploratory_steps:
            random_action = np.random.uniform(
                low=self.action_selector.action_min[0],
                high=self.action_selector.action_max[0],
                size=self.action_selector.action_dim,
            )
            return random_action
        else:
            return self.action_selector(policy, state)

    def rescale_action(self, action: np.ndarray):
        return self.action_selector.rescale_action(action)


class DDPGActionSelector(ActionSelector):
    """Action selector for (vanilla) DDPG policy

    Attributes:
        use_cuda (bool): true if using gpu

        action_min (np.ndarray): lower bound for continuous actions
        action_max (np.ndarray): upper bound for continuous actions

    """

    def __init__(self, action_dim: int, action_range: list, use_cuda: bool):
        ActionSelector.__init__(self, use_cuda)
        self.action_dim = action_dim
        self.action_min = np.array(action_range[0])
        self.action_max = np.array(action_range[1])

    def __call__(self, policy: nn.Module, state: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Generate action via policy"""
        if state.ndim == 1:
            state = state.reshape(1, -1)
        action = policy.forward(np2tensor(state, self.use_cuda))
        action_np = action.cpu().detach().view(-1).numpy()
        return action_np

    def rescale_action(self, action: np.ndarray) -> np.ndarray:
        """Rescale actions to fit continuous action spaces"""
        action_rescaled = (
            action * (self.action_max - self.action_min) / 2.0
            + (self.action_max + self.action_min) / 2.0
        )
        return action_rescaled


class GaussianNoise(ActionSelector):
    """Gaussian noise wrapper to wrap deterministic continuous policies"""

    def __init__(self, action_selector: ActionSelector, mu: float, sigma: float):
        ActionSelector.__init__(self, action_selector.use_cuda)
        self.action_selector = action_selector
        self.action_min = self.action_selector.action_min
        self.action_max = self.action_selector.action_max
        self.action_dim = self.action_selector.action_dim

        self.mu = mu
        self.sigma = sigma
        self.exploration = True

    def __call__(self, policy: nn.Module, state: np.ndarray) -> Tuple[np.ndarray, ...]:
        action = self.action_selector(policy, state)
        if self.exploration:
            action = action + np.random.normal(self.mu, self.sigma)
        return action

    def rescale_action(self, action: np.ndarray):
        return self.action_selector.rescale_action(action)


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
        ActionSelector.__init__(self, action_selector.use_cuda)
        self.action_selector = action_selector
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_min = self.action_selector.action_min
        self.action_max = self.action_selector.action_max
        self.action_dim = self.action_selector.action_dim

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
        return np.clip(action + ou_state, self.action_min, self.action_max)
