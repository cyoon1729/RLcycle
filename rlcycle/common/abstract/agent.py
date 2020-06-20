from abc import ABC, abstractmethod
from typing import Tuple, Type

import numpy as np
import torch
from omegaconf import DictConfig
from rlcycle.build import build_env
from rlcycle.common.abstract.action_selector import ActionSelector
from rlcycle.common.abstract.learner import Learner
from rlcycle.common.models.base import BaseModel
from rlcycle.common.utils.common_utils import np2tensor

# from rlcycle.common.utils.logger import Logger


class Agent(ABC):
    """Abstract base class for RL agents
    
    Attributes:
        experiment_info (DictConfig): configurations for running main loop (like args) 
        env_info (DictConfig): env info for initialization gym environment
        hyper_params (DictConfig): algorithm hyperparameters
        model_cfg (DictConfig): configurations for building neural networks
        log_cfg (DictConfig): configurations for logging algorithm run

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
        self.device = torch.device(self.experiment_info.device)

        self.env = build_env(self.experiment_info)

    @abstractmethod
    def _initialize(self):
        pass

    @abstractmethod
    def step(
        self, state: np.ndarray, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.float64, bool]:
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def get_policy(self) -> BaseModel:
        pass

    def test(self, action_selector: ActionSelector, episode_i: int, update_step: int):
        """Test policy without random exploration a number of times
        
        Params:
            step (int): step information, by episode number of model update step

        """
        print("====TEST START====")
        policy = self.get_policy()
        policy.eval()
        action_selector.exploration = False
        episode_rewards = []
        for test_i in range(self.experiment_info.test_num):
            state = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                self.env.render()
                action = action_selector(policy, state)
                state, action, reward, next_state, done = self.step(state, action)
                episode_reward = episode_reward + reward
                state = next_state

            print(
                f"episode num: {episode_i} | test: {test_i} episode reward: {episode_reward}"
            )
            episode_rewards.append(episode_reward)

        action_selector.exploration = True
        print(
            f"EPISODE NUM: {episode_i} | UPDATE STEP: {update_step} |"
            f"MEAN REWARD: {np.mean(episode_rewards)}"
        )
        print("====TEST END====")

    def _preprocess_experience(self, experience: Tuple[np.ndarray]):
        states, actions, rewards, next_states, dones = experience[:5]
        if self.hyper_params.use_per:
            indices, weights = experience[-2:]

        states = np2tensor(states, self.device)
        actions = np2tensor(actions.reshape(-1, 1), self.device)
        rewards = np2tensor(rewards.reshape(-1, 1), self.device)
        next_states = np2tensor(next_states, self.device)
        dones = np2tensor(dones.reshape(-1, 1), self.device)

        experience = (states, actions.long(), rewards, next_states, dones)

        if self.hyper_params.use_per:
            weights = np2tensor(weights.reshape(-1, 1), self.device)
            experience = experience + (indices, weights,)

        return experience
