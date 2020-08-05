"""
Adapted from OpenAI Baselines
https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
"""

import random
from typing import List, Tuple

import numpy as np
from omegaconf import DictConfig

from rlcycle.common.abstract.buffer import ReplayBufferWrapper
from rlcycle.common.buffer.replay_buffer import ReplayBuffer
from rlcycle.common.buffer.segment_tree import MinSegmentTree, SumSegmentTree


class PrioritizedReplayBuffer(ReplayBufferWrapper):
    """Buffer wrapper for prioritized sampling

    Attributes:
        hyper_params (DictConfig): algorithm hyper_params
        _alpha (float): PER alpha
        beta (float): PER beta
        beta_increment (float): beta increment rate
        _it_sum (SumSegmentTree):
        _it_min (MinSegmentTree):
        _max_priority (float): maximum possible priority value

    """

    def __init__(
        self, replay_buffer: ReplayBuffer, hyper_params: DictConfig,
    ):
        ReplayBufferWrapper.__init__(self, replay_buffer, hyper_params)
        self.hyper_params = hyper_params

        self._alpha = self.hyper_params.per_alpha
        assert self._alpha >= 0
        self.beta = self.hyper_params.per_beta
        self.beta_increment = (
            self.hyper_params.per_beta_max - self.beta
        ) / self.hyper_params.per_beta_total_steps

        it_capacity = 1
        while it_capacity < self.replay_buffer._maxsize:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        idx = self.replay_buffer._next_idx
        self.replay_buffer.add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size) -> List[int]:
        res = []
        p_total = self._it_sum.sum(0, len(self.replay_buffer._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self) -> Tuple[np.ndarray, ...]:
        idxes = self._sample_proportional(self.hyper_params.batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self.replay_buffer._storage)) ** (-self.beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self.replay_buffer._storage)) ** (-self.beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self.replay_buffer._encode_sample(idxes)

        self._update_beta()

        return tuple(list(encoded_sample) + [idxes, weights])

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.replay_buffer._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

    def _update_beta(self):
        self.beta = self.beta + self.beta_increment
        assert self.beta > 0

    def __len__(self) -> int:
        return len(self.replay_buffer)
