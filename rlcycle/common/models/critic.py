import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from rlcycle.common.models.base import BaseModel


class Critic(BaseModel):
    """Critic network"""

    def __init__(self, model_cfg: DictConfig):
        BaseModel.__init__(self, model_cfg)

        # set input size of fc input layer, first hidden later
        self.model_cfg.fc.input.params.input_size = self.get_feature_size()
        self.model_cfg.fc.hidden.hidden1.params.input_size = (
            self.model_cfg.action_dim + self.model_cfg.fc.input.params.output_size
        )

        # set output size of fc output layer
        self.model_cfg.fc.output.params.output_size = self.model_cfg.action_dim

        # initialize input layer
        self.fc_input = hydra.utils.instantiate(self.model_cfg.fc.input)

        # initialize hidden layers
        hidden_layers = []
        for layer in self.model_cfg.fc.hidden:
            layer_info = self.model_cfg.fc.hidden[layer]
            hidden_layers.append(hydra.utils.instantiate(layer_info))
        self.fc_hidden = nn.Sequential(*hidden_layers)

        # initialize output layer
        self.fc_output = hydra.utils.instantiate(self.model_cfg.fc.output)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state = self.features.forward(state)
        x = self.fc_input.forward(state)
        x_action = torch.cat([x, action], 1)
        x = self.fc_hidden.forward(x_action)
        x = self.fc_output.forward(x)
        return x


class FujimotoCritic(BaseModel):
    """Critic network based on Fujimoto et al. 2018 
    'Addressing Function Approximation Error in Actor-Critic Methods'
    """

    def __init__(self, model_cfg: DictConfig):
        BaseModel.__init__(self, model_cfg)

        # set input size of fc input layer, first hidden later
        self.model_cfg.fc.input.params.input_size = (
            self.get_feature_size() + self.model_cfg.action_dim
        )
        self.model_cfg.fc.hidden.hidden1.params.input_size = (
            self.model_cfg.action_dim + self.model_cfg.fc.input.params.output_size
        )

        # set output size of fc output layer
        self.model_cfg.fc.output.params.output_size = self.model_cfg.action_dim

        # initialize input layer
        self.fc_input = hydra.utils.instantiate(self.model_cfg.fc.input)

        # initialize hidden layers
        hidden_layers = []
        for layer in self.model_cfg.fc.hidden:
            layer_info = self.model_cfg.fc.hidden[layer]
            hidden_layers.append(hydra.utils.instantiate(layer_info))
        self.fc_hidden = nn.Sequential(*hidden_layers)

        # initialize output layer
        self.fc_output = hydra.utils.instantiate(self.model_cfg.fc.output)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state = self.features.forward(state)
        state_action = torch.cat([state, action], 1)
        x = self.fc_input.forward(state_action)
        x_action = torch.cat([x, action], 1)
        x = self.fc_hidden.forward(x_action)
        x = self.fc_output.forward(x)
        return x
