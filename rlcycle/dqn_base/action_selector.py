from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from gym import spaces
from omegaconf import DictConfig

from rlcycle.common.abstract.action_selector import ActionSelector
from rlcycle.common.utils.common_utils import np2tensor


class DQNActionSelector(ActionSelector):
    def __init__(self, device: str):
        ActionSelector.__init__(self, device)

    def __call__(self, policy: nn.Module, state: np.ndarray) -> Tuple[np.ndarray, ...]:
        state = np2tensor(state, self.device).unsqueeze(0)
        qvals = policy.forward(state)
        qvals = qvals.cpu().detach().numpy()
        action = np.argmax(qvals)
        return action


class QRActionSelector(ActionSelector):
    def __init__(self, device: str):
        ActionSelector.__init__(self, device)

    def __call__(self, policy: nn.Module, state: np.ndarray) -> Tuple[np.ndarray, ...]:
        state = np2tensor(state, self.device).unsqueeze(0)
        qvals = policy.forward(state).mean(2)  # fix dim
        qvals = qvals.cpu().detach().numpy()
        action = np.argmax(qvals)
        return action


class EpsGreedy(ActionSelector):
    def __init__(
        self,
        action_selector: ActionSelector,
        action_space: spaces.Discrete,
        hyper_params: DictConfig,
    ):
        ActionSelector.__init__(self, action_selector.device)
        self.action_selector = action_selector
        self.action_space = action_space
        self.eps = hyper_params.eps
        self.eps_final = hyper_params.eps_final
        self.eps_decay = (
            self.eps - self.eps_final
        ) / hyper_params.max_exploration_frame

    def __call__(
        self, policy: nn.Module, state: np.ndarray) -> np.ndarray:
        if self.eps > np.random.random() and self.exploration:
            return self.action_space.sample()
        return self.action_selector(policy, state)

    def decay_epsilon(self):
        eps = self.eps - self.eps_decay
        self.eps = max(eps, self.eps_final)
