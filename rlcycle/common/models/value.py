import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig


class BaseModel(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        nn.Module.__init__(self)
        self.model_cfg = model_cfg

        if self.model_cfg.use_conv:
            feature_modules = []
            for layer in self.model_cfg.conv_features:
                layer_info = self.model_cfg.conv_features[layer]
                feature_modules.append(hydra.utils.instantiate(layer_info))
            self.features = nn.Sequential(*feature_modules)

        else:
            print("Not using CNN backbone; Using identity layer.")
            self.features = nn.Identity(0)  # args aren't used in nn.Identity

    def get_feature_size(self):
        feature_size = (
            self.conv(torch.zeros(1, *self.model_cfg.state_dim)).view(-1).size(0)
        )
        return feature_size


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
        x = self.fc_input.forward(x)
        x = self.fc_hidden.forward(x)
        x = self.fc_output.forward(x)
        return x


class DuelingDQNModel(BaseModel):
    """Dueling DQN model initializable with hydra configs"""

    def __init__(self, model_cfg: DictConfig):
        BaseModel.__init__(self, model_cfg)

        # initialize feature layer and fc inputs if not using cnn
        if not self.use_conv:
            self.model_cfg.linear_features.params.input_size = self.get_feature_size()
            self.features = hydra.utils.instantiate(self.model_cfg.linear_features)
            self.model_cfg.advantage.fc1.params.input_size = (
                self.model_cfg.value.fc1.params.input_size
            ) = self.model_cfg.linear_features.params.output_size

        # set input sizes of fc input layer if using cnn
        if self.use_conv:
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
        advantage = self.advantage_stream.forward(x)
        value = self.value_stream.forward(x)
        return value + advantage - advantage.mean()