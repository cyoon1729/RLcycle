import hydra
from omegaconf import DictConfig

from rlcycle.common.abstract.loss import Loss
from rlcycle.common.utils.env_generator import generate_atari_env, generate_env


def build_agent(
    experiment_info: DictConfig, hyper_params: DictConfig, model: DictConfig
):
    """Build agent from DictConfigs via hydra.utils.instantiate()"""
    agent_cfg = DictConfig(dict())
    agent_cfg["class"] = experiment_info.agent
    agent_cfg["params"] = dict(
        experiment_info=experiment_info, hyper_params=hyper_params, model_cfg=model
    )
    agent = hydra.utils.instantiate(agent_cfg)
    return agent


def build_env(experiment_info: DictConfig):
    """Build gym environment from DictConfigs via hydra.utils.instantiate()"""
    if experiment_info.env.is_atari:
        env = generate_atari_env(experiment_info.env)
    else:
        env = generate_env(experiment_info.env)
    return env


def build_learner(
    experiment_info: DictConfig, hyper_params: DictConfig, model: DictConfig
):
    """Build learner from DictConfigs via hydra.utils.instantiate()"""
    learner_cfg = DictConfig(dict())
    learner_cfg["class"] = experiment_info.learner
    learner_cfg["params"] = dict(
        experiment_info=experiment_info, hyper_params=hyper_params, model_cfg=model
    )
    learner = hydra.utils.instantiate(learner_cfg)
    return learner


def build_model(model_cfg: DictConfig, use_cuda: bool):
    """Build model from DictConfigs via hydra.utils.instantiate()"""
    model_cfg.use_cuda = use_cuda
    model = hydra.utils.instantiate(model_cfg)
    if use_cuda:
        return model.cuda()
    else:
        return model.cpu()


def build_action_selector(experiment_info: DictConfig, use_cuda: bool):
    """Build action selector from DictConfig via hydra.utils.instantiate()"""
    action_selector_cfg = DictConfig(dict())
    action_selector_cfg["class"] = experiment_info.action_selector
    action_selector_cfg["params"] = dict(use_cuda=use_cuda)
    if not experiment_info.env.is_discrete:
        action_selector_cfg.params.action_dim = experiment_info.env.action_dim
        action_selector_cfg.params.action_range = experiment_info.env.action_range
    action_selector = hydra.utils.instantiate(action_selector_cfg)
    return action_selector


def build_loss(loss_type: str, hyper_params: DictConfig, use_cuda: bool) -> Loss:
    """Build loss from DictConfigs via hydra.utils.instantiate()"""
    loss_cfg = DictConfig(dict())
    loss_cfg["class"] = loss_type
    loss_cfg["params"] = dict(hyper_params=hyper_params, use_cuda=use_cuda)
    loss_fn = hydra.utils.instantiate(loss_cfg)
    return loss_fn
