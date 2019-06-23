import torch
import numpy as np

# KL divergence of two univariate Gaussian distributions
def KL_divergence(p_mean, p_std, q_mean, q_std):
    kld = torch.log(q_std/p_std) + (torch.pow(p_std) + torch.pow(p_mean - q_mean, 2))/(2 * torch.pow(q_std)) - 0.5
    return kld

# project value distribution onto atoms as in Categorical Algorithm
def dist_projection():
    """
    Params:
        next_dist (softmax-ed)
        n_atoms
        Vmin
        Vmax
        batch_size
        rewards
        gamma
    """

    proj_dist = np.zeros(n_atoms, dtype=np.float32)
    delta_z = (Vmax - Vmin) / (n_atoms - 1)

    for i in range(n_atoms):
        # Compute projection of Tz_j onto support {z_i}
        Tz_j = rewards + gamma * (Vmin + i * delta_z)
        b_j = (Tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ciel(b_j).astype(np.int64)

        # Distribute probability of Tz_j
        proj_dist[l] = proj_dist[l] + next_dist * (u - b_j)
        proj_dist[u] = proj_dist[u] +  next_dist * (b_j - l)

    return proj_dist




