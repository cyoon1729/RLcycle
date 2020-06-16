from typing import Tuple

import gym
import hydra
from omegaconf import DictConfig

from rlcycle.common.abstract.loss import Loss
from rlcycle.common.utils.env_wrappers import generate_atari_env, generate_env


def build_agent(
    experiment_info: DictConfig, hyper_params: DictConfig, model: DictConfig
):
    agent_cfg = DictConfig(dict())
    agent_cfg["class"] = experiment_info.agent
    agent_cfg["params"] = dict(
        experiment_info=experiment_info, hyper_params=hyper_params, model_cfg=model
    )
    agent = hydra.utils.instantiate(agent_cfg)
    return agent


def build_env(experiment_info: DictConfig):
    if experiment_info.env.is_atari:
        env = generate_atari_env(experiment_info.env)
    else:
        env = generate_env(experiment_info.env)
    return env


def build_learner(
    experiment_info: DictConfig, hyper_params: DictConfig, model: DictConfig
):
    learner_cfg = DictConfig(dict())
    learner_cfg["class"] = experiment_info.learner
    learner_cfg["params"] = dict(
        experiment_info=experiment_info, hyper_params=hyper_params, model_cfg=model
    )
    learner = hydra.utils.instantiate(learner_cfg)
    return learner


def build_model(model_cfg: DictConfig):
    model = hydra.utils.instantiate(model_cfg)
    return model


def build_action_selector(experiment_info: DictConfig):
    action_selector_cfg = DictConfig(dict())
    action_selector_cfg["class"] = experiment_info.action_selector
    action_selector_cfg["params"] = dict(device=experiment_info.device)
    action_selector = hydra.utils.instantiate(action_selector_cfg)
    return action_selector


def build_loss(experiment_info: DictConfig) -> Loss:
    loss_cfg = DictConfig(dict())
    loss_cfg["class"] = experiment_info.loss
    loss_fn = hydra.utils.instantiate(loss_cfg)
    return loss_fn
