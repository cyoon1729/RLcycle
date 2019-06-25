import torch
import numpy as np

# KL divergence of two univariate Gaussian distributions
def KL_divergence(p_mean, p_std, q_mean, q_std):
    kld = torch.log(q_std/p_std) + (torch.pow(p_std) + torch.pow(p_mean - q_mean, 2))/(2 * torch.pow(q_std)) - 0.5
    return kld

# project value distribution onto atoms as in Categorical Algorithm
def dist_projection(optimal_dist, rewards, dones, gamma, n_atoms, Vmin, Vmax, support):
    batch_size = rewards.size(0)
    m = torch.zeros(batch_size, n_atoms)
    delta_z = (Vmax - Vmin) / (n_atoms - 1)

    for sample_idx in range(batch_size):
        reward = rewards[sample_idx]
        done = dones[sample_idx]

        for atom in range(n_atoms):
            # compute projection of Tz_j
            Tz_j = reward + (1 - done) * gamma * support[atom]
            Tz_j = torch.clamp(Tz_j, Vmin, Vmax)
            b_j = (Tz_j - Vmin) / delta_z
            l = torch.floor(b_j).long().item()
            u = torch.ceil(b_j).long().item()

            # distribute probability of Tz_j
            m[sample_idx][l] = m[sample_idx][l] + optimal_dist[sample_idx][atom] * (u - b_j)
            m[sample_idx][u] = m[sample_idx][u] + optimal_dist[sample_idx][atom] * (b_j - l)

    return m
