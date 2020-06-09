from abc import ABC, abstractmethod
from typing import Tuple

import torch


class LearnerBase(ABC):
    """Abstract class for Learner"""

    @abstractmethod
    def update_model(
        self, experience: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        pass


class Learner(LearnerBase):
    """Abstract class for all """

    def __init__(self, args: dict, hyper_params: dict, model_cfg: dict):
        self.args = args
        self.hyper_params = hyper_params
        self.model_cfg = model_cfg

    @abstractmethod
    def _initialize(self):
        pass

    @abstractmethod
    def update_model(
        self, experience: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        pass


class LearnerWrapper(LearnerBase):
    """AbstractClass for Learner Wrappers"""

    def __init__(self, learner: Learner):
        self.learner = learner

    @abstractmethod
    def update_model(
        self, experience: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        pass    
