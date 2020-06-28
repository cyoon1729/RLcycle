import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Tuple

import torch
from omegaconf import DictConfig
from rlcycle.common.models.base import BaseModel


class LearnerBase(ABC):
    """Abstract base class for Learner"""

    @abstractmethod
    def update_model(
        self, experience: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        pass

    @abstractmethod
    def get_policy(self, target_device: torch.device) -> BaseModel:
        pass


class Learner(LearnerBase):
    """Abstract class for all learners

    Attributes:
        experiment_info (DictConfig): experiment info
        hyper_params (DictConfig): algorithm hyperparameters
        model_cfg (DictConfig): model configurations
        device (torch.device): map location for tensor computation

    """

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

        time_info = datetime.now()
        timestamp = f"{time_info.year}-{time_info.month}-{time_info.day}"
        self.ckpt_path = (
            f"../../../../checkpoints/{self.experiment_info.env.name}"
            f"/{self.experiment_info.experiment_name}/{timestamp}/"
        )
        os.makedirs(self.ckpt_path, exist_ok=True)

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
    """Abstract base class for Learner Wrappers

    Attributes:
        learner (Learner): learner to be wrapped

    """

    def __init__(self, learner: Learner):
        self.learner = learner

    def update_model(
        self, experience: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        """Call wrapped learner update_model()"""
        return self.learner.update_model(experience)

    def get_policy(self, target_device: torch.device):
        """Call wrapped learner get_policy()"""
        return self.learner.get_policy(target_device)
