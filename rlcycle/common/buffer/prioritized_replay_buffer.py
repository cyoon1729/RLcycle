"""
Adapted from OpenAI Baselines
https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
"""

import numpy as np
from rlcycle.common.buffer.replay_buffer import ReplayBuffer
from rlcycle.common.buffer.segment_tree import MinSegmentTree, SumSegmentTree


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, replay_buffer: ReplayBuffer, args: dict, hyper_params: dict):
        self.replay_buffer = replay_buffer

        self._alpha = hyper_params["per_alpha"]
        assert self._alpha >= 0

        self.beta = hyper_params["per_beta"]
        self.beta_increment = (hyper_params["per_beta_max"] - self.beta) / args[
            "max_update_steps"
        ]

        it_capacity = 1
        while it_capacity < self.replay_buffer._maxsize:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        idx = self._next_idx
        self.replay_buffer.add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self):
        idxes = self._sample_proportional(batch_size)

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

        return tuple(list(encoded_sample) + [weights, idxes])

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