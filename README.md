# RLcycle

[![Total alerts](https://img.shields.io/lgtm/alerts/g/cyoon1729/RLcycle.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/cyoon1729/RLcycle/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/cyoon1729/RLcycle.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/cyoon1729/RLcycle/context:python)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

RLcycle (pronounced as "recycle") is a reinforcement learning (RL) agents framework. RLcycle provides ready-made RL agents, as well as reusable components for easy prototyping. 

Currently, RLcycle provides:
- DQN + enhancements, Distributional: C51, Quantile Regression, Rainbow-DQN
- A2C (data parallel) and A3C (gradient parallel).
- DDPG, both Lillicrap et al. (2015) and Fujimoto et al., (2018) versions (Technically TD3).
- Soft Actor Critic with automatic entropy coefficient tuning.
- Prioritized Experience Replay and n-step updates for all off-policy algorithms

RLcycle uses:
- [PyTorch](https://github.com/pytorch/pytorch) for computations and building and optimizing models.
- [Hydra](https://github.com/facebookresearch/hydra) for configuring and building agents.
- [Ray](https://github.com/ray-project/ray) for parallelization. 
- [WandB](https://www.wandb.com/) for logging training and testing. 

See below for an introduction and guide to using RLcycle, performance benchmarks, and future plans.

#### Contributing

If you have any questions, or would like to contribute to RLcycle or offer any suggestions, feel free to raise an issue or reach out at `cjy2129 [at] columbia [dot] edu`!

## Getting Started
To install:
```
conda create --name myenv python=3.6.9 pip
conda activate myenv
git clone https://github.com/cyoon1729/RLcycle.git
cd RLcycle
pip install -U -r requirements.txt
pip install -e .
```

<details>
<summary> <b> 1. A quick look at Hydra for managing configurations </b></summary>
Let's first take a  look at one of the many useful things [hydra](https://hydra.cc/) can do:

```yaml
"""Instantiating classes from yaml files"""

# in ./examples/rectangle.yaml
shape:
    class: examples.shapes.Rectangle
        params:
            height: 5
            width: 4
```

Initializing `shapes.Rectangle` with `hydra` as in the above yaml file:

```python
"""Instantiating classes from yaml files"""

# in ./examples/shapes.py
class Rectangle:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
       
    def get_area(self):
        return width * height

# in ./examples/main.py
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="./examples/rectangle.yaml")
def main(cfg: DictConfig):
    shape = hydra.utils.instantiate(layer_info)
    print(shape.__class__.__name__)  # >>> 'Rectangle'
	print(shape.get_area()) # >>> 20
    
if __main__ == "__main__":
    main()
```
If you would like to know more about `hydra`,  [check it out](https://hydra.cc/)!
</details>

<details>
<summary> <b> 2. Initializing Components in RLcycle (models, learners, agents, etc) </b></summary>

Most of the components in `RLcycle` are instantiated via `hydra.utils.instantiate`, as illustrated in the section above. 
Examples:
1. models:
```yaml
"""in ./configs/lunarlander/models/dqn.yaml"""
model:
  class: rlcycle.common.models.value.DQNModel
  params:
    model_cfg:
      state_dim: undefined
      action_dim: undefined
      fc:
        input:
          class: rlcycle.common.models.layers.LinearLayer
          params: 
            input_size: undefined
            output_size: 128
            post_activation_fn: relu           
        hidden:
          hidden1:
            class: rlcycle.common.models.layers.LinearLayer
            params: 
              input_size: 128
              output_size: 128
              post_activation_fn: relu
          hidden2:
            class: rlcycle.common.models.layers.LinearLayer
            params: 
              input_size: 128
              output_size: 128
              post_activation_fn: relu
        output:
          class: rlcycle.common.models.layers.LinearLayer
          params:
            input_size: 128
            output_size: undefined
            post_activation_fn: identity
```
Building the above model:
```python
"""in ./rlcycle/build.py"""
def build_model(model_cfg: DictConfig, device: torch.device):
    """Build model from DictConfigs via hydra.utils.instantiate()"""
    model = hydra.utils.instantiate(model_cfg)
    return model.to(device)
```

2. learners:
```python
# in ./rlcycle/dqn_base/agent.py
#....
self.learner = build_learner(
    self.experiment_info, self.hyper_params, self.model_cfg
)
# ...

# in ./rlcycle/build.py
def build_learner(
    experiment_info: DictConfig, hyper_params: DictConfig, model: DictConfig
):
    """Build learner from DictConfigs via hydra.utils.instantiate()"""
    learner_cfg = DictConfig(dict())
    learner_cfg["class"] = experiment_info.learner
    learner_cfg["params"] = dict(
        experiment_info=experiment_info,
        hyper_params=hyper_params,
        model_cfg=model
    )
    learner = hydra.utils.instantiate(learner_cfg)
    return learner
```

3. agents: 
```python
# in ./tests/test_dqn.py
@hydra.main(config_path="../configs/lunarlander/dqn.yaml", strict=False)
def main(cfg: DictConfig):
    agent = build_agent(**cfg)
    agent.train()

# in ./rlcycle.build.py
def build_agent(
    experiment_info: DictConfig, hyper_params: DictConfig, model: DictConfig
):
    """Build agent from DictConfigs via hydra.utils.instantiate()"""
    agent_cfg = DictConfig(dict())
    agent_cfg["class"] = experiment_info.agent
    agent_cfg["params"] = dict(
        experiment_info=experiment_info,
        hyper_params=hyper_params,
        model_cfg=model
    )
    agent = hydra.utils.instantiate(agent_cfg)
    return agent
```
This should help you get started with building & running agents in RLcycle!
</details>

## Benchmarks
To be added shortly.


## Future Plans

Below are some things I hope to incorporate to RLcycle:
- TRPO and PPO  *(medium priority)*
- IQN *(low priority)*
- Compatibility with my distributed RL framework [distributedRL](https://github.com/cyoon1729/distributedRL). (i.e. Ape-X for all off-policy algorithms). *(high priority)*

## References
