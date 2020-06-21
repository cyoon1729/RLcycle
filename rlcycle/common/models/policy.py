import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig

from rlcycle.common.models.base import BaseModel


class DeterministicPolicy(BaseModel):
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


class GaussianPolicy(BaseModel):
    """Gaussian Policy (for Soft Actor Critic)"""

    def __init__(self, model_cfg: DictConfig):
        BaseModel.__init__(self, model_cfg)
        self.log_std_min = self.model_cfg.log_std_min
        self.log_std_max = self.model_cfg.log_std_max

        self.model_cfg.fc.input.params.input_size = self.get_feature_size()
        self.fc_input = hydra.utils.instantiate(self.model_cfg.fc.input)

        hidden_layers = []
        for layer in self.model_cfg.fc.hidden:
            layer_info = self.model_cfg.fc.hidden[layer]
            hidden_layers.append(hydra.utils.instantiate(layer_info))
        self.fc_hidden = nn.Sequential(*hidden_layers)

        mu_stream = []
        for layer in self.model_cfg.fc.mu_stream:
            layer_info = self.model_cfg.fc.mu_stream[layer]
            mu_stream.append(hydra.utils.instantiate(layer_info))
        self.mu_stream = nn.Sequential(*mu_stream)

        log_sigma_stream = []
        for layer in self.model_cfg.fc.log_sigma_stream:
            layer_info = self.model_cfg.fc.log_sigma_stream[layer]
            mu_stream.append(hydra.utils.instantiate(layer_info))
        self.log_sigma_stream = nn.Sequential(*log_sigma_stream)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor]:
        x = self.features.forward(state)
        x = x.view(x.size(0), -1)
        x = self.fc_input.forward(state)
        x = self.fc_hidden.forward(state)

        mu = self.mu_stream.forward(x)
        log_sigma = self.log_sigma_stream(x)
        log_sigma = torch.clamp(log_sigma, self.log_std_min, self.log_std_max)

        return mean, log_sigma

    def sample(self, state: torch.Tensor, epsilon=1e-6) -> Tuple[torch.Tensor]:
        mu, log_sigma = self.forward(state)
        sigma = log_sigma.exp()
        normal = Normal(mu, sigma)
        z = normal.rsample()
        log_pi = (
            normal.log_prob(z) - torch.log(1 - (torch.tanh(z)).pow(2) + epsilon)
        ).sum(1, keepdim=True)

        return mu, sigma, z, log_pi
