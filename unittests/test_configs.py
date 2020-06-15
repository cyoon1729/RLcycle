import hydra
import torch
from omegaconf import DictConfig

from rlcycle.build import (build_action_selector, build_agent, build_env,
                           build_loss, build_model)


@hydra.main(config_path="../configs/meta_config.yaml", strict=False)
def main(cfg: DictConfig):
    # print all configs
    print(cfg.pretty())

    # build env
    print("===TESTING ENV===")
    env = build_env(cfg.experiment_info.env)
    print(env.reset())
    print("=================")

    # build model
    print("===TESTING MODEL===")
    cfg.model.params.model_cfg.state_dim = env.observation_space.shape
    cfg.model.params.model_cfg.action_dim = env.action_space.n
    cfg.model.params.model_cfg.fc.output.params.output_size = env.action_space.n
    model = hydra.utils.instantiate(cfg.model)
    print(model)
    test_input = torch.FloatTensor(env.reset()).unsqueeze(0)
    print(model.forward(test_input))
    print("===================")

    # build action_selector
    print("===Testing Action Selector===")
    action_selector_cfg = DictConfig(dict())
    action_selector_cfg["class"] = cfg.experiment_info.action_selector
    action_selector_cfg["params"] = dict(device="cpu")
    action_selector = hydra.utils.instantiate(action_selector_cfg)
    print(action_selector)
    print("==============================")

    # build loss
    print("===Testing Loss===")
    loss_cfg = DictConfig(dict())
    loss_cfg["class"] = cfg.experiment_info.loss
    loss = hydra.utils.instantiate(loss_cfg)
    print(loss)
    print("==================")

    # # build learner (just testing if it correctly instantiates)
    # print("===Testing Learner===")
    # learner_cfg = DictConfig(dict())
    # learner_cfg.cls = cfg.experiment_info.learner
    # learner_cfg.params = cfg.dqn_hp
    # learner_cfg.model_cfg = cfg.model
    # learner = hydra.utils.instantiate(learner_cfg)
    # print("=====================")


if __name__ == "__main__":
    main()
