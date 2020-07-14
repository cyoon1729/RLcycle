from typing import Dict

import numpy as np
from omegaconf import DictConfig

from rlcycle.build import build_action_selector, build_env, build_model


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
        self.use_cuda = self.experiment_info.worker_device == "cuda"
        self.action_selector = build_action_selector(
            self.experiment_info, self.use_cuda
        )
        self.actor = build_model(policy_cfg, self.use_cuda)

    def run_trajectory(self) -> Dict[str, np.ndarray]:
        """Finish one env episode and return trajectory experience"""
        trajectory = dict(states=[], actions=[], rewards=[])
        done = False
        state = self.env.reset()
        episode_reward = 0
        while not done:
            # Carry out environment step
            action = self.action_selector(self.actor, state)
            next_state, reward, done, _ = self.env.step(action)

            # Store to trajectory
            trajectory["states"].append(state)
            trajectory["actions"].append(action)
            trajectory["rewards"].append(reward)

            state = next_state
            episode_reward = episode_reward + reward

        print(f"[TRAIN] [Worker-{self.rank}] Score: {episode_reward}")

        trajectory_np = []
        for key in list(trajectory.keys()):
            trajectory_np.append(np.array(trajectory[key]))

        trajectory_info = dict(
            rank=self.rank, trajectory=trajectory_np, score=episode_reward
        )

        return trajectory_info

    def synchronize_policy(self, new_state_dict):
        """Synchronize policy with received new parameters"""
        self.actor.load_state_dict(new_state_dict)
