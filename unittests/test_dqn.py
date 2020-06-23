import hydra
import torch
from omegaconf import DictConfig

from rlcycle.build import (build_action_selector, build_agent, build_env,
                           build_learner, build_loss, build_model)


@hydra.main(config_path="../configs/dqn.yaml", strict=False)
def main(cfg: DictConfig):
    agent = build_agent(**cfg)
    agent.train()


if __name__ == "__main__":
    main()
