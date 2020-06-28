# RLcycle


RLcycle (to be pronounced as "recycle") is a reinforcement learning (RL) agents framework. RLcycle provides ready-made RL agents, as well as reusable components for easy prototyping. 

Currently, RLcycle provides:
- DQN + enhancements.
- A2C (data parallel) and A3C (gradient parallel).
- DDPG, both Lillicrap et al. (2015) and Fujimoto et al., (2018) versions.
- Soft Actor Critic with automatic entropy coefficient tuning.
- Prioritized Experience Replay for all off-policy algorithhms

RLcycle uses 
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