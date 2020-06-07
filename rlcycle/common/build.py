from typing import Tuple

from rlcycle.common.abstract.agent import Agent
from rlcycle.registry import AGENTS, LOSSES


def build_cfg(path: str) -> Tuple[str, dict, dict, dict]:
    """Organize and return the different configs from config.yml"""
    # TODO: implement....

    return algo, hyper_params, model_cfg, log_cfg


def build_agent_from_yml(path: str) -> Agent:
    """build and return agent from config.yml file"""
    algo, hyper_params, model_cfg, log_cfg = build_cfg(path)
    agent_cls = AGENTS[algo]
    agent = agent_cls(hyper_params, model_cfg, log_cfg)
    return agent


def build_loss(args):
    loss_type = args["loss_type"]
    loss_cls = LOSSES[loss_type]
    loss_fn = loss_cls()
    return loss_fn


def build_model(args, model_cfg):
    pass


def build_env(args):
    "build RL environment"
    env = None
    return env
