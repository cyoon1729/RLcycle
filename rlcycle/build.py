from typing import Tuple

from rlcycle.common.abstract.agent import Agent
from rlcycle.registry import AGENTS


def build_cfg(path: str) -> Tuple[str, dict, dict, dict]:
    """Organize and return the different configs from config.yml"""
    # TODO: implement....

    return algo, hyper_params, model_cfg, log_cfg


def build_agent_from_yml(path: str) -> Agent:
    """build and return agent from config.yml file"""
    algo, hyper_params, model_cfg, log_cfg = build_cfg(path)
    agent_cls = AGENTS["algo"]
    agent = agent_cls(hyper_params, model_cfg, log_cfg)
    return agent
