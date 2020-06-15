from typing import List

import numpy as np
import torch
import torch.nn as nn


def np2tensor(np_arr: np.ndarray, device: torch.device):
    tensor_output = torch.FloatTensor(np_arr).to(device)
    if device.type is "cuda":
        tensor_output.cuda(non_blocking=True)
    return tensor_output


def soft_update(network: nn.Module, target_network: nn.Module, tau: float):
    for param, target_param in zip(network.parameters(), target_network.parameters()):
        target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))


def hard_update(network: nn.Module, target_network: nn.Module):
    for param, target_param in zip(network.parameters(), target_network.parameters()):
        target_param.data.copy_(param.data)
