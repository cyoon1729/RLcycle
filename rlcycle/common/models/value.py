import hydra
import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlcycle.common.models.base import BaseModel


class DQN(BaseModel):
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
        if self.model_cfg.fc.output.params.output_size == "undefined":
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

    def reset_noise(self):
        """Reset noise for noisy linear"""
        self.fc_input.reset_noise()
        for hidden in self.fc_hidden:
            hidden.reset_noise()
        self.fc_output.reset_noise()


class DuelingDQN(BaseModel):
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
        if self.model_cfg.advantage[output_layer_key].params.output_size == "undefined":
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

    def reset_noise(self):
        """Reset noise for noisy linear"""
        for layer in self.advantage_stream:
            layer.reset_noise()
        for layer in self.value_stream:
            layer.reset_noise()


class CategoricalDQN(DQN):
    """Categorical DQN (a.k.a C51) Model

    Attributes:
        v_min (float): lower bound for support
        v_max (float): upper bound for support
        delta_z (float): distance between discrete points in support
        support (torch.Tensor): canonical returns
        fc_input (LinearLayer): fully connected input layer
        fc_hidden (nn.Sequential): hidden layers
        fc_output (LinearLayer): fully connected output layer

    """

    def __init__(self, model_cfg):
        self.action_dim = model_cfg.action_dim

        self.num_atoms = model_cfg.num_atoms
        self.v_min = model_cfg.v_min
        self.v_max = model_cfg.v_max
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms)
        if model_cfg.use_cuda:
            self.support = self.support.cuda()

        # set output size of fc output layer
        model_cfg.fc.output.params.output_size = self.num_atoms * self.action_dim

        DQN.__init__(self, model_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_x = x.size(0)
        x = self.features.forward(x)
        x = x.view(x.size(0), -1)
        x = self.fc_input.forward(x)
        x = self.fc_hidden.forward(x)
        x = self.fc_output.forward(x)
        dist = x.view(num_x, -1, self.num_atoms)
        dist = F.softmax(dist, dim=2)

        return dist


class DuelingCategoricalDQN(DuelingDQN):
    """Dueling Categorical DQN as in Rainbow-DQN

    Attributes:
        v_min (float): lower bound for support
        v_max (float): upper bound for support
        delta_z (float): distance between discrete points in support
        support (torch.Tensor): canonical returns
        advantage_stream (nn.Sequential): distributional advantage stream
        value_stream (nn.Sequential): distributional value stream

    """

    def __init__(self, model_cfg):
        self.action_dim = model_cfg.action_dim
        self.num_atoms = model_cfg.num_atoms
        self.v_min = model_cfg.v_min
        self.v_max = model_cfg.v_max
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms)
        if model_cfg.use_cuda:
            self.support = self.support.cuda()

        # set output size of advantage stream to represent distribution:
        output_layer_key = list(model_cfg.advantage.keys())[-1]
        model_cfg.advantage[output_layer_key].params.output_size = (
            self.num_atoms * self.action_dim
        )

        # set output size of value stream to represent distribution:
        output_layer_key = list(model_cfg.value.keys())[-1]
        model_cfg.value[output_layer_key].params.output_size = self.num_atoms

        DuelingDQN.__init__(self, model_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_x = x.size(0)
        x = self.features.forward(x)
        x = x.view(x.size(0), -1)

        advantage_dist = self.advantage_stream.forward(x)
        advantage_dist = advantage_dist.view(num_x, -1, self.num_atoms)
        advantage_mean = torch.mean(advantage_dist, dim=1, keepdim=True)

        value_dist = self.value_stream.forward(x)
        value_dist = value_dist.view(num_x, 1, self.num_atoms)

        q_dist = advantage_dist + value_dist - advantage_mean
        q_dist = F.softmax(q_dist, dim=2)

        return q_dist


class QRDQN(DQN):
    """Quantile Regression DQN Model initializable with hydra configs

    Attributes:
        tau (torch.Tensor): quantile weights
        num_quantiles (int): number of quantiles for distributional representation
        fc_input (LinearLayer): fully connected input layer
        fc_hidden (nn.Sequential): hidden layers
        fc_output (LinearLayer): fully connected output layer

    """

    def __init__(self, model_cfg):
        self.action_dim = model_cfg.action_dim
        self.num_quantiles = model_cfg.num_quantiles
        self.tau = (
            torch.tensor(
                (2.0 * np.arange(self.num_quantiles) + 1) / (2.0 * self.num_quantiles),
                requires_grad=False,
            )
            .float()
            .view(1, -1)
        )
        if model_cfg.use_cuda:
            self.tau = self.tau.cuda()

        # set output size of fc output layer
        model_cfg.fc.output.params.output_size = self.num_quantiles * self.action_dim

        DQN.__init__(self, model_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagate through network"""
        num_x = x.size(0)
        x = self.features.forward(x)
        x = x.view(num_x, -1)
        x = self.fc_input.forward(x)
        x = self.fc_hidden.forward(x)
        dist = self.fc_output.forward(x)
        dist = dist.view(num_x, self.action_dim, self.num_quantiles)

        return dist


class DuelingQRDQN(DuelingDQN):
    """Dueling Categorical DQN as in Rainbow-DQN

    Attributes:
        tau (torch.Tensor): quantile weights
        num_quantiles (int): number of quantiles for distributional representation
        advantage_stream (nn.Sequential): distributional advantage stream
        value_stream (nn.Sequential): distributional value stream

    """

    def __init__(self, model_cfg):
        self.action_dim = model_cfg.action_dim
        self.num_quantiles = model_cfg.num_quantiles
        self.tau = torch.FloatTensor(
            (2.0 * np.arange(self.num_quantiles) + 1) / (2.0 * self.num_quantiles),
        ).view(1, -1)
        if model_cfg.use_cuda:
            self.tau = self.tau.cuda()

        # set output size of advantage stream to represent distribution:
        output_layer_key = list(model_cfg.advantage.keys())[-1]
        model_cfg.advantage[output_layer_key].params.output_size = (
            self.num_quantiles * self.action_dim
        )

        # set output size of value stream to represent distribution:
        output_layer_key = list(model_cfg.value.keys())[-1]
        model_cfg.value[output_layer_key].params.output_size = self.num_quantiles

        DuelingDQN.__init__(self, model_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_x = x.size(0)
        x = self.features.forward(x)
        x = x.view(x.size(0), -1)

        advantage_dist = self.advantage_stream.forward(x)
        advantage_dist = advantage_dist.view(num_x, self.action_dim, self.num_quantiles)
        advantage_mean = torch.mean(advantage_dist, dim=1, keepdim=True)

        value_dist = self.value_stream.forward(x)
        value_dist = value_dist.view(num_x, 1, self.num_quantiles)

        q_dist = advantage_dist + value_dist - advantage_mean

        return q_dist
