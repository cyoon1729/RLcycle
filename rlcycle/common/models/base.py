import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """Base Class for neural network model

    Attributes:
        model_cfg (DictConfig): configurations for building the model
        features (nn.Sequential): feature layer (linear, convolutional)

    """

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

    def get_feature_size(self) -> int:
        """Get output size of feature layers; input size for fc input layer"""
        try:  # if state_dim is a tuple, like atari images
            dummy_feature = self.features(torch.zeros(1, *self.model_cfg.state_dim))
        except TypeError:
            dummy_feature = self.features(torch.zeros(1, self.model_cfg.state_dim))

        feature_size = dummy_feature.view(-1).size(0)
        return feature_size
