from typing import Deque

import numpy as np
import torch
import torch.nn as nn


def np2tensor(np_arr: np.ndarray, use_cuda: bool):
    """Convert numpy array to tensor"""
    if use_cuda:
        return torch.from_numpy(np_arr).cuda(non_blocking=True).float()
    return torch.from_numpy(np_arr).cpu().float()


def preprocess_nstep(n_step_queue: Deque, gamma: float) -> tuple:
    """Return n-step transition data with discounted n-step rewards"""
    discounted_reward = 0
    _, _, _, last_state, done = n_step_queue[-1]
    for transition in list(reversed(n_step_queue)):
        state, action, reward, _, _ = transition
        discounted_reward = reward + gamma * discounted_reward

    return state, action, discounted_reward, last_state, done


def soft_update(network: nn.Module, target_network: nn.Module, tau: float):
    """Update target network weights with polyak averaging"""
    for param, target_param in zip(network.parameters(), target_network.parameters()):
        target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))


def hard_update(network: nn.Module, target_network: nn.Module):
    """Copy target network weights from network"""
    for param, target_param in zip(network.parameters(), target_network.parameters()):
        target_param.data.copy_(param.data)
