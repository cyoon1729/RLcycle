import numpy as np
from torch.distributions import Categorical

from rlcycle.common.abstract.action_selector import ActionSelector
from rlcycle.common.models.base import BaseModel
from rlcycle.common.utils.common_utils import np2tensor


class A2CDiscreteActionSelector(ActionSelector):
    """Action selector for A2C discrete action space"""

    def __init__(self, use_cuda: bool):
        ActionSelector.__init__(self, use_cuda)

    def __call__(self, policy: BaseModel, state: np.ndarray) -> np.ndarray:
        if state.ndim == 1:
            state = state.reshape(1, -1)
        state = np2tensor(state, self.use_cuda)
        dist = policy.forward(state)
        categorical_dist = Categorical(dist)
        if self.exploration:
            action = categorical_dist.sample().cpu().detach().numpy()
        else:
            action = categorical_dist.sample().cpu().argmax().numpy()
        return action.item()


class A2CContinuousActionSelector(ActionSelector):
    """Action selector for A2C continuous action space."""

    def __init__(self, use_cuda: bool):
        ActionSelector.__init__(self, use_cuda)

    def __call__(self, policy: BaseModel, state: np.ndarray) -> np.ndarray:
        pass
