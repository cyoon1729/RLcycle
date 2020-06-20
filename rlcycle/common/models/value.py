import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig

from rlcycle.common.models.base import BaseModel


class DQNModel(BaseModel):
    """Vanilla (Nature) DQN model initializable with hydra config"""

    def __init__(self, model_cfg: DictConfig):
        BaseModel.__init__(self, model_cfg)

        # set input size of fc input layer
        self.model_cfg.fc.input.params.input_size = self.get_feature_size()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features.forward(x)
        x = x.view(x.size(0), -1)
        x = self.fc_input.forward(x)
        x = self.fc_hidden.forward(x)
        x = self.fc_output.forward(x)
        return x


class DuelingDQNModel(BaseModel):
    """Dueling DQN model initializable with hydra configs"""

    def __init__(self, model_cfg: DictConfig):
        print(model_cfg.pretty())
        BaseModel.__init__(self, model_cfg)

        # initialize feature layer and fc inputs if not using cnn
        if not self.model_cfg.use_conv:
            self.model_cfg.linear_features.params.input_size = self.get_feature_size()
            self.features = hydra.utils.instantiate(self.model_cfg.linear_features)
            self.model_cfg.advantage.fc1.params.input_size = (
                self.model_cfg.value.fc1.params.input_size
            ) = self.model_cfg.linear_features.params.output_size

        # set input sizes of fc input layer if using cnn
        if self.model_cfg.use_conv:
            self.model_cfg.advantage.fc1.params.input_size = (
                self.model_cfg.value.fc1.params.input_size
            ) = self.get_feature_size()

        # set output size of advantage fc output layer:
        output_layer_key = list(self.model_cfg.advantage.keys())[-1]
        self.model_cfg.advantage[
            output_layer_key
        ].params.output_size = self.model_cfg.action_dim

        # initialize advantage head
        advantage_stream = []
        for layer in self.model_cfg.advantage:
            layer_info = self.model_cfg.advantage[layer]
            advantage_stream.append(hydra.utils.instantiate(layer_info))
        self.advantage_stream = nn.Sequential(*advantage_stream)

        # initialize value head
        value_stream = []
        for layer in self.model_cfg.value:
            layer_info = self.model_cfg.value[layer]
            value_stream.append(hydra.utils.instantiate(layer_info))
        self.value_stream = nn.Sequential(*value_stream)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features.forward(x)
        x = x.view(x.size(0), -1)
        advantage = self.advantage_stream.forward(x)
        value = self.value_stream.forward(x)
        return value + advantage - advantage.mean()


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
