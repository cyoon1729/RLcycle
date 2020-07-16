from typing import Tuple

import hydra
from omegaconf import DictConfig
import torch
from torch.distributions import Normal
import torch.nn as nn

from rlcycle.common.models.base import BaseModel


class MLPPolicy(BaseModel):
    """Configurable MLP based policy (e.g. Softmax policy, deterministic policy)

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

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward propogate through policy network"""
        x = self.features.forward(state)
        x = x.view(x.size(0), -1)
        x = self.fc_input.forward(x)
        x = self.fc_hidden.forward(x)
        x = self.fc_output.forward(x)
        return x


class GaussianPolicy(BaseModel):
    """Gaussian Policy

    Attributes:
        fc_input (LinearLayer): fully connected input layer
        fc_hidden (nn.Sequential): hidden layers
        mu_stream (nn.Sequential): layers that output mean of gaussian distribution
        log_sigma_stream (nn.Sequential): layers that output log_std of gaussian distribution
        log_std_min (float): lower bound for log_sigma_stream output
        log_std_max (float): upper bound for log_sigma_stream output

    """

    def __init__(self, model_cfg: DictConfig):
        BaseModel.__init__(self, model_cfg)
        # Define requisite attributes
        self.log_std_min = self.model_cfg.log_std_min
        self.log_std_max = self.model_cfg.log_std_max
        self.model_cfg.fc.input.params.input_size = self.get_feature_size()
        self.model_cfg.fc.mu_stream.output.params.output_size = (
            self.model_cfg.fc.log_sigma_stream.output.params.output_size
        ) = self.model_cfg.action_dim

        # Initialize input layer
        self.fc_input = hydra.utils.instantiate(self.model_cfg.fc.input)

        # Initialize hidden layers
        hidden_layers = []
        for layer in self.model_cfg.fc.hidden:
            layer_info = self.model_cfg.fc.hidden[layer]
            hidden_layers.append(hydra.utils.instantiate(layer_info))
        self.fc_hidden = nn.Sequential(*hidden_layers)

        # Initialize mu stream
        mu_stream = []
        for layer in self.model_cfg.fc.mu_stream:
            layer_info = self.model_cfg.fc.mu_stream[layer]
            mu_stream.append(hydra.utils.instantiate(layer_info))
        self.mu_stream = nn.Sequential(*mu_stream)

        # Initialize log_sigma stream
        log_sigma_stream = []
        for layer in self.model_cfg.fc.log_sigma_stream:
            layer_info = self.model_cfg.fc.log_sigma_stream[layer]
            log_sigma_stream.append(hydra.utils.instantiate(layer_info))
        self.log_sigma_stream = nn.Sequential(*log_sigma_stream)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor]:
        """Forward propogate through policy network"""
        x = self.features.forward(state)
        x = x.view(x.size(0), -1)
        x = self.fc_input.forward(x)
        x = self.fc_hidden.forward(x)

        mu = self.mu_stream.forward(x)
        log_sigma = self.log_sigma_stream(x)
        log_sigma = torch.clamp(log_sigma, self.log_std_min, self.log_std_max)

        return mu, log_sigma

    def sample(self, state: torch.Tensor, epsilon=1e-6) -> Tuple[torch.Tensor]:
        """Sample gaussian distribution"""
        if state.dim() == 1:
            state = state.view(1, -1)

        mu, log_sigma = self.forward(state)
        sigma = log_sigma.exp()

        normal = Normal(mu, sigma)
        z = normal.rsample()

        log_pi = (
            normal.log_prob(z) - torch.log(1 - (torch.tanh(z)).pow(2) + epsilon)
        ).sum(1, keepdim=True)

        return mu, sigma, z, log_pi
