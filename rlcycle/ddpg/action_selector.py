import numpy as np
import torch
from gym import spaces

from rlcycle.common.abstract.action_selector import ActionSelector
from rlcycle.common.utils.common_utils import np2tensor


class DDPGActionSelector(ActionSelector):
    def __call__(
        self, policy: nn.Module, state: np.ndarray, device: torch.device
    ) -> Tuple[np.ndarray, ...]:
        action = self.policy.forward(np2tensor(state, device))
        action = action.cpu().detach().numpy()
        return action


# Ornstein-Ulhenbeck Noise
# Adpated from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(ActionSelector):
    def __init__(
        self,
        action_selector: ActionSelector,
        action_space: space.Box,
        mu: float = 0.0,
        theta: float = 0.15,
        max_sigma: float = 0.3,
        min_sigma: float = 0.3,
        decay_period: int = 100000,
    ):
        self.action_selector: action_selector
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self._reset()

    def _reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx

    def __call__(
        self, policy: nn.Module, state: np.ndarray, device: torch.device
    ) -> Tuple[np.ndarray, ...]:
        action = self.action_selector(policy, state, device)
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, t / self.decay_period
        )
        return np.clip(action + ou_state, self.low, self.high)
