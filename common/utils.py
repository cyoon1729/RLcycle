import torch
import numpy as np

# KL divergence of two univariate Gaussian distributions
def KL_divergence(p_mean, p_std, q_mean, q_std):
    kld = torch.log(q_std/p_std) + (torch.pow(p_std) + torch.pow(p_mean - q_mean, 2))/(2 * torch.pow(q_std)) - 0.5
    return kld

# project value distribution onto atoms as in Categorical Algorithm
def dist_projection(n_atoms, Vmin, Vmax, batch, next_best_dist, gamma):
    """
    Params:
        n_atoms (int)
        Vmin (float)
        Vmax (float)
        batch (array of experience tuples)
        next_dist (tensor)
        gamma (float)
    """

    batch_size = batch.size(0)
    states, actions, rewards, next_states, dones = batch
    # next_best_dist = next_dist[range(batch_size), next_actions]

    delta_z = (Vmax - Vmin) / (n_atoms - 1)
    m = torch.zeros_like(next_dist)

    for sample_idx in range(batch_size):
        action = actions[sample_idx].long.item()
        reward = rewards[sample_idx]
        done = dones[sample_idx]


        for atom_dix in range(n_atoms):
            # Compute projection of Tz_j onto {z_i}
            Tz_j = reward + (1 - done) * gamma * (Vmin + i * delta_z)
            Tz_j = torch.clamp(Tz_j, Vmin, Vmax)
            b_j = (Tz_j - Vmin) / delta_z

            l = torch.floor(b_j).long().item()
            u = torch.ceil(b_j).long().item()

            p_j = next_dist[sample_idx, action]
            # Distribute probability of Tz_j
            if done:
                m[sample_idx, action, l] = m[sample_idx, action, l] + (u - b_j)
                m[sample_idx, action, u] = m[sample_idx, action, u] + (b_j - l)
            else:
                m[sample_idx, action, l] = m[sample_idx, action, l] + next_best_dist[sample_idx] * (u - b_j)
                m[sample_idx, action, u] = m[sample_idx, action, u] + next_best_dist[sample_idx]* (b_j - l)


