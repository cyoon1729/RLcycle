import numpy as np
import torch
import torch.nn as nn


def soft_update(network: nn.Module, target_network: nn.Module, tau: float):
    for param, target_param in zip(network.parameters(), target_network.parameters()):
        target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))


def hard_update(network: nn.Module, target_network: nn.Module):
    for param, target_param in zip(network.parameters(), target_network.parameters()):
        target_param.data.copy_(param.data)
