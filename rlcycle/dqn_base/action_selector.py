import numpy as np
import torch
import torch.nn as nn
from gym import spaces
from rlcycle.common.abstract.action_selector import ActionSelector
from rlcycle.common.utils.common_utils import np2tensor


class DQNActionSelector(ActionSelector):
    def __init__(self, device: torch.device):
        ActionSelector.__init__(self, device)

    def __call__(self, policy: nn.Module, state: np.ndarray) -> Tuple[np.ndarray, ...]:
        qvals = policy.forward(np2tensor(state, device))
        qvals = qvals.cpu().detach().numpy()
        action = np.argmax(qvals)
        return action


class QRActionSelector(ActionSelector):
    def __init__(self, device: torch.device):
        ActionSelector.__init__(self, device)

    def __call__(self, policy: nn.Module, state: np.ndarray) -> Tuple[np.ndarray, ...]:
        qvals = policy.forward(np2tensor(state, device)).mean(2)  # fix dim
        qvals = qvals.cpu().detach().numpy()
        action = np.argmax(qvals)
        return action


class EpsGreedy(ActionSelector):
    def __init__(
        self,
        action_selector: ActionSelector,
        action_space: spaces.Discrete,
        hyper_params: dict,
    ):
        self.action_selector = action_selector
        self.action_space = action_space
        self.eps = self.hyper_params["eps"]
        self.eps_decay = self.hyper_params["eps_decay"]

    def __call__(
        self, policy: nn.Module, state: np.ndarray, explore: bool = True
    ) -> np.ndarray:
        if np.random.randn(0, 1) < self.eps and explore:
            return self.action_space.sample()
        return self.action_selector(policy, state, device)

    def decay_epsilon(self):
        pass
