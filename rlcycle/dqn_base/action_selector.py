from typing import Tuple

from gym import spaces
import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn as nn

from rlcycle.common.abstract.action_selector import ActionSelector
from rlcycle.common.utils.common_utils import np2tensor


class DQNActionSelector(ActionSelector):
    """DQN arg-max action selector"""

    def __init__(self, use_cuda: bool):
        ActionSelector.__init__(self, use_cuda)

    def __call__(self, policy: nn.Module, state: np.ndarray) -> Tuple[np.ndarray, ...]:
        if state.ndim == 1:
            state = state.reshape(1, -1)
        state = np2tensor(state, self.use_cuda).unsqueeze(0)
        with torch.no_grad():
            qvals = policy.forward(state)
            qvals = qvals.cpu().detach().numpy()
        action = np.argmax(qvals)
        return action


class QRActionSelector(ActionSelector):
    """Action selector for Quantile Q-value representations"""

    def __init__(self, use_cuda: bool):
        ActionSelector.__init__(self, use_cuda)

    def __call__(self, policy: nn.Module, state: np.ndarray) -> Tuple[np.ndarray, ...]:
        if state.ndim == 1:
            state = state.reshape(1, -1)
        state = np2tensor(state, self.use_cuda).unsqueeze(0)
        with torch.no_grad():
            qvals = policy.forward(state).mean(dim=2)
            qvals = qvals.cpu().numpy()
        action = np.argmax(qvals)
        return action


class CategoricalActionSelector(ActionSelector):
    """Action selector for categorical Q-value presentations"""

    def __init__(self, use_cuda: bool):
        ActionSelector.__init__(self, use_cuda)

    def __call__(self, policy: nn.Module, state: np.ndarray) -> Tuple[np.ndarray, ...]:
        state = np2tensor(state, self.use_cuda).unsqueeze(0)
        with torch.no_grad():
            dist = policy.forward(state)
            weights = dist * policy.support
            qvals = weights.sum(dim=2).cpu().numpy()
        action = np.argmax(qvals)

        return action


class EpsGreedy(ActionSelector):
    """ActionSelector wrapper for epsilon greedy policy

    Attributes:
        action_selector (ActionSelector): action selector to wrap
        action_space (???): gym environment action space
        eps (float): epsilon value for epsilon greedy
        eps_final (float): minimum epsilon value to reach
        eps_decay (float): decay rate for epsilon

    """

    def __init__(
        self,
        action_selector: ActionSelector,
        action_space: spaces.Discrete,
        hyper_params: DictConfig,
    ):
        ActionSelector.__init__(self, action_selector.use_cuda)
        self.action_selector = action_selector
        self.action_space = action_space
        self.eps = hyper_params.eps
        self.eps_final = hyper_params.eps_final
        self.eps_decay = (
            self.eps - self.eps_final
        ) / hyper_params.max_exploration_frame

    def __call__(self, policy: nn.Module, state: np.ndarray) -> np.ndarray:
        """Return exploration action if eps > random.uniform(0,1)"""
        if self.eps > np.random.random() and self.exploration:
            return self.action_space.sample()
        return self.action_selector(policy, state)

    def decay_epsilon(self):
        """Decay epsilon as learning progresses"""
        eps = self.eps - self.eps_decay
        self.eps = max(eps, self.eps_final)
