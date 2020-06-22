import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig


class BaseModel(nn.Module):
    """Base Class for neural network model"""

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
        try:  # if state_dim is a tuple, like atari images
            feature_size = (
                self.features(torch.zeros(1, *self.model_cfg.state_dim))
                .view(-1)
                .size(0)
            )
        except TypeError:
            assert (
                type(self.model_cfg.state_dim) is int
            ), "state_dim must be int or iterable"
            feature_size = (
                self.features(torch.zeros(1, self.model_cfg.state_dim)).view(-1).size(0)
            )
        return feature_size
