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
        device = torch.device(device)
        ActionSelector.__init__(self, device)

    def __call__(self, policy: nn.Module, state: np.ndarray) -> Tuple[np.ndarray, ...]:
        state = np2tensor(state, self.device).unsqueeze(0)
        qvals = policy.forward(np2tensor(state))
        qvals = qvals.cpu().detach().numpy()
        action = np.argmax(qvals)
        return action


class QRActionSelector(ActionSelector):
    def __init__(self, device: str):
        device = torch.device(device)
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
        self.action_selector = action_selector
        self.action_space = action_space
        self.eps = hyper_params.eps
        self.eps_decay = hyper_params.eps_decay
        self.eps_final = hyper_params.eps_final

    def __call__(
        self, policy: nn.Module, state: np.ndarray, explore: bool = True
    ) -> np.ndarray:
        if np.random.randn(0, 1) < self.eps and explore:
            return self.action_space.sample()
        return self.action_selector(policy, state, device)

    def decay_epsilon(self):
        pass
