from abc import ABC, abstractmethod
from typing import Type

from rlcycle.common.abstract.learner import Learner
from rlcycle.common.utils.logger import Logger


class Agent(ABC):
    """Abstract base class for RL agents"""

    def __init__(self, args: dict, hyper_params: dict, model_cfg: dict, log_cfg: dict):
        self.args = args
        self.hyper_params = hyper_params
        self.model_cfg = model_cfg
        self.log_cfg = log_cfg
        self.device = self.args["device"]

    @abstractmethod
    def _initialize(self):
        pass

    @abstractmethod
    def _init_logger(self, logger: Logger):
        pass

    @abstractmethod
    def step(
        self, state: np.ndarray, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.float64, bool]:
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass
