from abc import ABC, abstractmethod
from typing import Tuple

import torch


class Learner:
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
