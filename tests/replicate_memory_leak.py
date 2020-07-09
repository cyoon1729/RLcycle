import numpy as np
import torch

from rlcycle.common.utils.debug.memory import MemProfiler


def leak():
    mp = MemProfiler(stopper=False)
    device = torch.device("cuda")
    nparr = np.zeros((32, 84, 84, 3)).astype(np.float64)

    for _ in range(100):
        mp.start()
        tens = torch.from_numpy(nparr).float().to(device)
        tens.cuda(non_blocking=True)
        mp.stop()


if __name__ == "__main__":
    leak()
