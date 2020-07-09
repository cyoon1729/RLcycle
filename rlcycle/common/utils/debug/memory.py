import gc
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

    def start(self):
        """Set base memory size"""
        self.rss_base = self._get_rss()
        self.gc_base = self._get_gc_size()
        print("--------------------------------------------")
        print(f"rss base = {self.rss_base} \tgc size = {self.gc_base}")

    def stop(self):
        """Compute memory consumption relative to starting point"""
        rss_size = self._get_rss() - self.rss_base
        gc_size = self._get_gc_size() - self.gc_base
        print(f"leak = {rss_size} \tgc size = {gc_size}")
        print("--------------------------------------------")

        if self.stopper:
            input()

    def _get_rss(self):
        return psutil.Process(os.getpid()).memory_info().rss // 1024

    def _get_gc_size(self):
        return self.heap.heap().size
