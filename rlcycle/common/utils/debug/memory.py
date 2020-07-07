import gc
import resource

from guppy import hpy
import torch


class MemProfiler:
    """A simple memory profiler to detect leaks line-by-line"""

    def __init__(self, stopper: bool = True):
        self.rss_base = 0
        self.gc_base = 0
        self.heap = hpy()
        self.stopper = stopper

    def start(self):
        """Set base memory size"""
        self.rss_base = self._get_rss()
        self.gc_base = self._get_gc_size()

    def end(self):
        """Compute memory consumption relative to starting point"""
        rss_size = self._get_rss() - self.rss_base
        gc_size = self._get_gc_size() - self.gc_base
        print(f"rss size = {rss_size} \tgc size = {gc_size}")
        if self.stopper:
            input("------------")

    def view_gc_tensors(self):
        """view garbage collected tensors"""
        tensor_info = []
        gc_objects = gc.get_objects()
        for gc_object in gc_objects():
            if torch.is_tensor(gc_object):
                tensor_info.append(
                    (str(gc_object.device), gc_object.dtype, tuple(gc_object.size()))
                )

        for info in tensor_info:
            print(f"{info[0]}, {info[1]}, {info[2]}")

    def _get_rss(self):
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    def _get_gc_size(self):
        return self.heap.heap().size
