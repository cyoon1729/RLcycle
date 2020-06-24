from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from omegaconf import DictConfig


class ReplayBufferBase(ABC):
    """Abstract base class for replay buffers"""

    @abstractmethod
    def add(
        self,
        obs_t: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        obs_tp1: np.ndarray,
        done: bool,
    ):
        pass

    @abstractmethod
    def sample(self, batch_size=int) -> Tuple[np.ndarray]:
        pass


class ReplayBufferWrapper(ReplayBufferBase):
    """Base class for replay buffer wrappers

    Attributes:
        replay_buffer (ReplayBufferBase): replay buffer to be wrapped
        hyper_params (DictConfig): algorithm hyperparameters

    """

    def __init__(self, replay_buffer: ReplayBufferBase, hyper_params: DictConfig):
        self.replay_buffer = replay_buffer
        self.hyper_params = hyper_params

    def add(
        self,
        obs_t: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        obs_tp1: np.ndarray,
        done: bool,
    ):
        self.replay_buffer.add(obs_t, action, reward, obs_tp1, done)

    def sample(self, batch_size=int) -> Tuple[np.ndarray]:
        return self.replay_buffer.sample(batch_size)
