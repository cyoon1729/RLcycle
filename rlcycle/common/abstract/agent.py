from abc import ABC, abstractmethod
from typing import Tuple, Type

import numpy as np
from rlcycle.common.abstract.learner import Learner
from omegaconf import DictConfig
# from rlcycle.common.utils.logger import Logger


class Agent(ABC):
    """Abstract base class for RL agents
    
    Attributes:
        experiment_info (DictConfig): configurations for running main loop (like args) 
        env_info (DictConfig): env info for initialization gym environment
        hyper_params (DictConfig): algorithm hyperparameters
        model_cfg (DictConfig): configurations for building neural networks
        log_cfg (DictConfig): configurations for logging algorithm run

    """

    def __init__(
        self,
        experiment_info: DictConfig,
        env_info: DictConfig,
        hyper_params: DictConfig,
        model_cfg: DictConfig,
        log_cfg: DictConfig
    ):
        self.experiment_info = experiment_info
        self.hyper_params = hyper_params
        self.model_cfg = model_cfg
        self.log_cfg = log_cfg
        self.device = self.args["device"]

    @abstractmethod
    def _initialize(self):
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
