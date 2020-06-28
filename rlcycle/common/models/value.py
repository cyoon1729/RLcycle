import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from rlcycle.common.models.base import BaseModel


class DQNModel(BaseModel):
    """Vanilla (Nature) DQN model initializable with hydra config

    Attributes:
        fc_input (LinearLayer): fully connected input layer
        fc_hidden (nn.Sequential): hidden layers
        fc_output (LinearLayer): fully connected output layer

    """

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
    """Dueling DQN model initializable with hydra configs

    Attributes:
        fc_input (LinearLayer): fully connected input layer
        advantage_stream (nn.Sequential): advantage stream dueling dqn
        value_stream (nn.Sequential): value stream of dueling dqn

    """

    def __init__(self, model_cfg: DictConfig):
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


class QRDQN(BaseModel):
    """Quantile Regression DQN Model initializable with hydra configs

    Attributes:
        fc_input (LinearLayer): fully connected input layer
        fc_hidden (nn.Sequential): hidden layers
        fc_output (LinearLayer): fully connected output layer
        tau (torch.Tensor): quantile weights
        num_quantiles (int): number of quantiles for distributional representation

    """

    def __init__(self, model_cfg):
        BaseModel.__init__(self, model_cfg)
        self.action_dim = self.model_cfg.action_dim
        self.num_quantiles = self.model_cfg.num_quantiles

        # set input size of fc input layer
        self.model_cfg.fc.input.params.input_size = self.get_feature_size()

        # set output size of fc output layer
        self.model_cfg.fc.output.params.output_size = (
            self.num_quantiles * self.action_dim
        )

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

        self.tau = torch.FloatTensor(
            (2.0 * np.arange(self.num_quantiles) + 1) / (2.0 * self.num_quantiles)
        ).view(1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagate through network"""
        x = self.features.forward(x)
        x = x.view(x.size(0), -1)
        x = self.fc_input.forward(x)
        x = self.fc_hidden.forward(x)
        x = self.fc_output.forward(x)

        return x.view(-1, self.num_actions, self.num_quantiles)
