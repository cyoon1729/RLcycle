import math

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
        activation_args (dict): function parameters for activation_function (e.g. dim)

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

        self.activation_args = dict()
        if post_activation_fn == "softmax":
            self.activation_args["dim"] = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.post_activation_fn(x, **self.activation_args)
        return x


class FactorizedNoisyLinearLayer(nn.Module):
    """Noisy linear layer with factorized gaussian noise.

    Attributes:
        input_size (int): layer input size
        output_size (int): layer output size
        mu_weight (nn.Parameter): trainiable weights
        sigma_weight (nn.Parameter): trainable gaussian distribution noise std for weights
        mu_bias (nn.Parameter): trainiable bias
        sigma_bias (nn.Parameter): trainable gaussian distribution noise std for bias
        eps_weight (torch.FloatTensor): Factorized Gaussian noise
        eps_bias (torch.FloatTensor): Factorized Gaussian noise
        post_activation_fn (nn.functional): post activation function
        activation_args (dict): function parameters for activation_function (e.g. dim)

    """

    def __init__(
        self, input_size: int, output_size: int, post_activation_fn: str,
    ):
        nn.Module.__init__(self)
        assert (
            post_activation_fn in activation_fn_registry.keys()
        ), f"{post_activation_fn} is not registered in layer.py. Register."

        self.input_size = input_size
        self.output_size = output_size

        self.activation_args = dict()
        if post_activation_fn == "softmax":
            self.activation_args["dim"] = 1
        self.post_activation_fn = activation_fn_registry[post_activation_fn]

        # Define layer parameters
        self.mu_weight = nn.Parameter(torch.zeros(self.output_size, self.input_size))
        self.sigma_weight = nn.Parameter(torch.zeros(self.output_size, self.input_size))
        self.register_buffer(
            "eps_weight", torch.zeros(self.output_size, self.input_size)
        )

        self.mu_bias = nn.Parameter(torch.zeros(self.output_size))
        self.sigma_bias = nn.Parameter(torch.zeros(self.output_size))
        self.register_buffer("eps_bias", torch.zeros(self.output_size))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x) -> torch.Tensor:
        """Forward propagate through layer."""
        if self.train:
            linear_output = F.linear(
                x,
                self.mu_weight + self.sigma_weight * self.eps_weight,
                self.mu_bias + self.sigma_bias * self.eps_bias,
            )
        else:
            linear_output = F.linear(x, self.mu_weight, self.mu_bias)
        output = self.post_activation_fn(linear_output, **self.activation_args)
        return output

    def reset_parameters(self):
        """Reset trainable parameters."""
        std = 1 / math.sqrt(self.input_size)
        self.mu_weight.data.uniform_(-std, std)
        self.sigma_weight.data.fill_(0.5 / math.sqrt(self.input_size))
        self.mu_bias.data.uniform_(-std, std)
        self.sigma_bias.data.fill_(0.5 / math.sqrt(self.output_size))

    def reset_noise(self):
        """Reset noise."""
        eps_in = self.scale_noise(self.input_size)
        eps_out = self.scale_noise(self.output_size)

        self.eps_weight.copy_(eps_out.ger(eps_in))
        self.eps_bias.copy_(eps_out)

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        # x = torch.from_numpy(np.random.normal(loc=0.0, scale=1.0, size=size)).float()
        x = torch.normal(mean=0.0, std=1.0, size=tuple([size]))
        return x.sign().mul(x.abs().sqrt())


class NoisyLinearLayer(nn.Module):
    """Noisy linear layer with gaussian noise

    Attributes:
        input_size (int): layer input size
        output_size (int): layer output size
        sigma_weight (nn.Parameter): trainable gaussian distribution noise std for weights
        mu_bias (nn.Parameter): trainiable bias
        sigma_bias (nn.Parameter): trainable gaussian distribution noise std for bias
        post_activation_fn (nn.functional): post activation function
        activation_args (dict): function parameters for activation_function (e.g. dim)

    """

    def __init__(
        self, input_size: int, output_size: int, post_activation_fn: str,
    ):
        nn.Module.__init__(self)
        assert (
            post_activation_fn in activation_fn_registry.keys()
        ), f"{post_activation_fn} is not registered in layer.py. Register."

        self.input_size = input_size
        self.output_size = output_size

        self.activation_args = dict()
        if post_activation_fn == "softmax":
            self.activation_args["dim"] = 1
        self.post_activation_fn = activation_fn_registry[post_activation_fn]

        # Define layer parameters
        self.mu_weight = nn.Parameter(
            torch.zeros(self.output_size, self.input_size).float()
        )
        self.sigma_weight = nn.Parameter(
            torch.zeros(self.output_size, self.input_size).float()
        )
        self.register_buffer(
            "eps_weight", torch.zeros(self.output_size, self.input_size).float()
        )

        self.mu_bias = nn.Parameter(torch.zeros(self.output_size).float())
        self.sigma_bias = nn.Parameter(torch.zeros(self.output_size).float())
        self.register_buffer("eps_bias", torch.zeros(self.output_size).float())

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        """Forward propagate through layer."""
        if self.train:
            linear_output = F.linear(
                x,
                self.mu_weight + self.sigma_weight * self.eps_weight,
                self.mu_bias + self.sigma_bias * self.eps_bias,
            )
        else:
            linear_output = F.linear(x, self.mu_weight, self.mu_bias)
        output = self.post_activation_fn(linear_output, **self.activation_args)
        return output

    def reset_parameters(self):
        """Reset trainable parameters."""
        std = 1 / math.sqrt(self.input_size)
        self.mu_weight.data.uniform_(-std, std)
        self.sigma_weight.data.fill_(0.5 / math.sqrt(self.output_size))
        self.mu_bias.data.uniform_(-std, std)
        self.sigma_bias.data.fill_(0.5 / math.sqrt(self.output_size))

    def reset_noise(self):
        """Reset noise."""
        self.eps_weight.data.normal_()
        self.eps_bias.data.normal_()
