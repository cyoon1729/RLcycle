from abc import ABC, abstractmethod
from typing import Type

from rlcycle.common.abstract.learner import Learner
from rlcycle.common.utils.logger import Logger


class Agent(ABC):
    """Abstract base class for RL agents"""

    def __init__(self, hyper_params: dict, model_cfg: dict, log_cfg: dict):
        pass

    @abstractmethod
    def _init_learner(self, learner_cls: Type[Learner]):
        pass

    @abstractmethod
    def _init_logger(self, logger: Logger):
        pass

    @abstractmethod
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.float64, bool]:
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass
