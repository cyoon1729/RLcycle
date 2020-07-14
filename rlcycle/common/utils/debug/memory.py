import os

from guppy import hpy
import psutil


class MemProfiler:
    """A simple memory profiler to detect leaks line-by-line"""

    def __init__(self, stopper: bool = True):
        self.rss_base = 0
        self.gc_base = 0
        self.heap = hpy()
        self.stopper = stopper
        self.total_leaked = 0

    def start(self):
        """Set base memory size"""
        self.rss_base = self._get_rss()
        self.gc_base = self._get_gc_size()

    def stop(self):
        """Compute memory consumption relative to starting point"""
        leak = self._get_rss() - self.rss_base
        gc_size = self._get_gc_size() - self.gc_base
        self.total_leaked += leak
        print("--------------------------------------------")
        print(
            f"rss base = {self.rss_base} \tgc size = {self.gc_base}\n"
            f"leak = {leak} \tgc size = {gc_size}, \ttotal leaked = {self.total_leaked}"
        )
        print("--------------------------------------------")

        if self.stopper:
            input()

    def set_rss_ckpt(self):
        self.rss_ckpt = self._get_rss()

    def _get_rss(self):
        return psutil.Process(os.getpid()).memory_info().rss // 1024

    def _get_gc_size(self):
        return self.heap.heap().size
