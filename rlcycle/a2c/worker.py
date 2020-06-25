from typing import Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from rlcycle.build import build_action_selector, build_env, build_model
from rlcycle.common.utils.common_utils import np2tensor


class TrajectoryRolloutWorker:
    """Worker that stores policy and runs environment trajectory

    Attributes:
        rank (int): rank of process
        env (gym.Env): gym environment to train on
        action_selector (ActionSelector): action selector given policy and state
        policy (BaseModel): policy for action selection

    """

    def __init__(self, rank: int, experiment_info: DictConfig, policy_cfg: DictConfig):
        self.experiment_info = experiment_info

        self.rank = rank
        self.env = build_env(experiment_info)
        self.action_selector = build_action_selector(experiment_info)
        self.policy = build_model(policy_cfg)
        self.device = torch.device(self.experiment_info.device)

    def run_trajectory(self) -> Tuple[np.ndarray]:
        """Finish one env episode and return trajectory experience"""
        trajectory = dict(states=[], actions=[], rewards=[], next_states=[], dones=[])
        done = False
        state = self.env.reset()
        while not done:
            # Carry out environment step
            action = self.action_selector(self.policy, state)
            next_state, reward, done, _ = self.env.step(action)

            # Store to trajectory
            trajectory["states"].append(state)
            trajectory["actions"].append(action)
            trajectory["rewards"].append(reward)
            trajectory["next_states"].append(next_state)
            trajectory["dones"].append(done)

            state = next_state

        trajectory_np = []
        for key in list(trajectory.keys):
            trajectory_np.append(trajectory[key])

        return tuple(trajectory_np)

    def _preprocess_trajectory(
        self, trajectory: Tuple[np.ndarray, ...], target_location: torch.device
    ) -> Tuple[torch.Tensor]:
        states, actions, rewards, next_states, dones = trajectory

        states = np2tensor(states, self.device)
        actions = np2tensor(actions.reshape(-1, 1), self.device)
        rewards = np2tensor(rewards.reshape(-1, 1), self.device)
        next_states = np2tensor(next_states, self.device)
        dones = np2tensor(dones.reshape(-1, 1), self.device)

        if self.experiment_info.is_discrete:
            actions = actions.long()

        trajectory = (states, actions, rewards, next_states, dones)

        return trajectory

    def synchronize_policy(self, new_state_dict):
        self.policy.load_state_dict(new_state_dict)
