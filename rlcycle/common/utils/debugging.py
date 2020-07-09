import collections
import gc
import resource

import torch


def gc_mem_profile():
    memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"max mem usage: {memory_usage}")

    # Tensors caught in gc
    tensors = []
    total_mem = 0
    for garbage in gc.get_objects():
        if torch.is_tensor(garbage):
            info = (str(garbage.device), garbage.dtype, tuple(garbage.shape))
            tensors.append(info)

    for info in tensors:
        print(f"device: {info[0]}\ttensor type: {info[1]}\ttensor shape: {info[2]}")
    print(f"tensors in gc: {len(tensors)} \tmax mem usage: {memory_usage}")
