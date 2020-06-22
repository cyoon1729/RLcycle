from abc import ABC, abstractmethod
from typing import Tuple

import torch
from omegaconf import DictConfig

from rlcycle.common.models.base import BaseModel


class LearnerBase(ABC):
    """Abstract class for Learner"""

    @abstractmethod
    def update_model(
        self, experience: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        pass

    @abstractmethod
    def get_policy(self, target_device: torch.device) -> BaseModel:
        pass


class Learner(LearnerBase):
    """Abstract class for all """

    def __init__(
        self,
        experiment_info: DictConfig,
        hyper_params: DictConfig,
        model_cfg: DictConfig,
    ):
        self.experiment_info = experiment_info
        self.hyper_params = hyper_params
        self.model_cfg = model_cfg
        self.device = torch.device(experiment_info.device)

    @abstractmethod
    def _initialize(self):
        pass

    @abstractmethod
    def update_model(
        self, experience: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        pass

    @abstractmethod
    def get_policy(self, target_device: torch.device) -> BaseModel:
        pass


class LearnerWrapper(LearnerBase):
    """AbstractClass for Learner Wrappers"""

    def __init__(self, learner: Learner):
        self.learner = learner

    def update_model(
        self, experience: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        return self.learner.update_model(experience)

    def get_policy(self, taraget_location: torch.device):
        return self.learner.get_policy(target_device)
