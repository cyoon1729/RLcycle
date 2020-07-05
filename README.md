# RLcycle

[![Total alerts](https://img.shields.io/lgtm/alerts/g/cyoon1729/RLcycle.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/cyoon1729/RLcycle/alerts/)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/cyoon1729/RLcycle.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/cyoon1729/RLcycle/alerts/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

RLcycle (pronounced as "recycle") is a reinforcement learning (RL) agents framework. RLcycle provides ready-made RL agents, as well as reusable components for easy prototyping. 

Currently, RLcycle provides:
- DQN + enhancements, Distributional: C51, Quantile Regression
- A2C (data parallel) and A3C (gradient parallel).
- DDPG, both Lillicrap et al. (2015) and Fujimoto et al., (2018) versions.
- Soft Actor Critic with automatic entropy coefficient tuning.
- Prioritized Experience Replay for all off-policy algorithhms

RLcycle uses:
- [PyTorch](https://github.com/pytorch/pytorch) for computations and building and optimizing models.
- [Hydra](https://github.com/facebookresearch/hydra) for configuring and building agents.
- [Ray](https://github.com/ray-project/ray) for parallelization. 
- [WandB](https://www.wandb.com/) for logging training and testing. 

See below for an introduction and guide to using RLcycle, performance benchmarks, and future plans.

#### Contributing

If you have any questions, or would like to contribute to RLcycle or offer any suggestions, feel free to raise an issue or reach out at `cjy2129 [at] columbia [dot] edu`!

## Getting Started

### 1. Configuring experiments with hydra.

### 2. Initializing and running agents.

## Benchmarks
To be added shortly.


## Future Plans

Below are some things I hope to incorporate to RLcycle:
- TRPO and PPO  *(medium priority)*
- Rainbow-DQN and IQN *(medium priority)*
- Compatibility with my distributed RL framework [distributedRL](https://github.com/cyoon1729/distributedRL). (i.e. Ape-X for all off-policy algorithms). *(high priority)*

## References
