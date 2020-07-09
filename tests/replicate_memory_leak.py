import numpy as np
import torch

from rlcycle.common.utils.debug.memory import MemProfiler
from rlcycle.common.utils.common_utils import np2tensor, np2tensor2


def leak():
    mp = MemProfiler(stopper=False)
    device = torch.device("cuda")


    for _ in range(100):
        nparr = np.random.randn(32, 3, 84, 84)
        mp.start()
        tens = np2tensor(nparr, device)
        mp.stop()


if __name__ == "__main__":
    leak()
