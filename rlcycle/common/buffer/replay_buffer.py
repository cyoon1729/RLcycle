"""
Adapted from OpenAI Baselines:
https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
"""
import random
from typing import Tuple

import numpy as np
from omegaconf import DictConfig

from rlcycle.common.abstract.buffer import ReplayBufferBase


class ReplayBuffer(ReplayBufferBase):
    """Replay Buffer

    Attributes:
        hyper_params (DictConfig): algorithm hyperparameters
        _storage (list): internal storage
        _maxsize (int): maximum buffer size
        _next_idx (int): tracker for index of last appended experience

    """

    def __init__(self, hyper_params: DictConfig):
        self.hyper_params = hyper_params
        self._storage = []
        self._maxsize = self.hyper_params.replay_buffer_size
        self._next_idx = 0

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        """Add experience to storage"""
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        """Return encoded sample"""
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return (
            np.array(obses_t),
            np.array(actions),
            np.array(rewards),
            np.array(obses_tp1),
            np.array(dones),
        )

    def sample(self) -> Tuple[np.ndarray, ...]:
        """Sample and return experience from storage"""
        idxes = [
            random.randint(0, len(self._storage) - 1)
            for _ in range(self.hyper_params.batch_size)
        ]
        return self._encode_sample(idxes)
