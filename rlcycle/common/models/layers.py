import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    """Identity mapping"""

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


activation_fn_registry = {
    "identity": Identity(),
    "relu": F.relu,
    "tanh": torch.tanh,
    "softmax": F.softmax,
}


class Conv2DLayer(nn.Module):
    """2D convolutional layer for network composition

    Attributes:
        conv2d (nn.Conv2d): single convolutional layer
        activation_fn (nn.functional): (post) activation function

    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        kernel_size: int,
        stride: int,
        activation_fn: str,
    ):
        nn.Module.__init__(self)
        assert (
            activation_fn in activation_fn_registry.keys()
        ), f"{activation_fn} is not registered in layer.py. Register."
        self.conv2d = nn.Conv2d(input_size, output_size, kernel_size, stride)
        self.activation_fn = activation_fn_registry[activation_fn]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2d(x)
        x = self.activation_fn(x)
        return x


class LinearLayer(nn.Module):
    """LinearLayer for network composition

    Attributes:
        linear (nn.Linear): single linear activation layer
        post_activation_fn (nn.functional): post activation function

    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        post_activation_fn: str,
        init_w: float = None,
    ):
        nn.Module.__init__(self)
        assert (
            post_activation_fn in activation_fn_registry.keys()
        ), f"{post_activation_fn} is not registered in layer.py. Register."
        self.linear = nn.Linear(input_size, output_size)
        self.post_activation_fn = activation_fn_registry[post_activation_fn]

        if init_w is not None:
            self.linear.weight.data.uniform_(-init_w, init_w)
            self.linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.post_activation_fn(x)
        return x


class ActivationLayer(nn.Module):
    """A configurable activation layer for network composition

    Attributes:
        input size (int): layer input size
        output size (int): layer outpt size
        activation_fn (nn.functional): activation function

    """

    def __init__(self, input_size: int, output_size: int, activation_fn: str):
        nn.Module.__init__(self)
        assert (
            activation_fn in activation_fn_registry.keys()
        ), f"{activation_fn} is not registered in layer.py. Register."
        assert input_size == output_size, "Input dim must equal output dim"
        self.input_size = input_size
        self.output_size = output_size
        self.activation_fn = activation_fn_registry[activation_fn]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(0) == self.input_size, "Input dimension mismatch"
        return self.activation_fn(x)
