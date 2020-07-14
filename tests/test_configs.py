from omegaconf import DictConfig
import torch

from rlcycle.build import (
    build_action_selector,
    build_agent,
    build_env,
    build_learner,
    build_loss,
    build_model,
)


def main(cfg: DictConfig):
    # print all configs
    print(cfg.pretty())

    # build env
    print("===INITIALIZING ENV===")
    env = build_env(cfg.experiment_info)
    print(env.reset())
    print("=================")

    # build model
    print("===INITIALIZING MODEL===")
    cfg.model.params.model_cfg.state_dim = env.observation_space.shape
    cfg.model.params.model_cfg.action_dim = env.action_space.n
    cfg.model.params.model_cfg.fc.output.params.output_size = env.action_space.n
    model = build_model(cfg.model)
    test_input = torch.FloatTensor(env.reset()).unsqueeze(0)
    print(model)
    print(model.forward(test_input))
    print("===================")

    # build action_selector
    print("===INITIALIZING ACTION SELECTOR===")
    action_selector = build_action_selector(cfg.experiment_info)
    print(action_selector)
    print("==============================")

    # build loss
    print("===INITIALIZING LOSS===")
    loss = build_loss(cfg.experiment_info)
    print(loss)
    print("==================")

    # build learner
    print("===INITIALIZING LEARNER===")
    learner = build_learner(**cfg)
    print(learner)
    print("=====================")

    # build agent
    print("===INITIALIZING AGENT===")
    agent = build_agent(**cfg)
    print(agent)
    print("=====================")


if __name__ == "__main__":
    main()
