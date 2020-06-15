from typing import Tuple

import gym
import hydra
from omegaconf import DictConfig

from rlcycle.common.abstract.loss import Loss
from rlcycle.common.utils.env_wrappers import generate_atari_env, generate_env


def build_agent(cfgs: DictConfig):
    agent_cfg = DictConfig()
    agent_cfg.cls = cfgs.experiment_info.agent
    agent_cfg.params.experiment_info = cfgs.experiment_info
    agent_cfg.params.hyper_params = cfgs.hyper_params
    agent_cfg.params.model_cfg = cfgs.model
    agent = hydra.utils.instantiate(agent_cfg)
    return agent


def build_env(env_info: DictConfig):
    if env_info.is_atari:
        env = generate_atari_env(env_info)
    else:
        env = generate_env(env_info)

    return env


def build_learner(
    experiment_info: DictConfig, hyper_params: DictConfig, model_cfg: DictConfig
):
    learner_cfg = DictConfig()
    learner_cfg.cls = experiment_info.learner
    learner_cfg.params.hyper_params = hyper_params
    learner_cfg.params.model_cfg = model_cfg
    learner = hydra.utils.instantiate(learner_cfg)
    return learner, learner_cfg


def build_model(model_cfg: DictConfig):
    model = hydra.utils.instantiate(model_cfg)
    return model


def build_action_selector(experiment_info: DictConfig):
    action_selector = hydra.utils.instantiate(experiment_info.action_selector)
    return action_selector


def build_loss(loss: DictConfig) -> Loss:
    loss_fn = hydra.utils.instantiate(loss)
    return loss_fn
