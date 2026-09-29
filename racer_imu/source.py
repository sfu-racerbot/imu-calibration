"""Base class for threaded sample sources (serial drivers, replay)."""
from __future__ import annotations

import queue
import threading
import time


class Source(threading.Thread):
    def __init__(self, name: str):
        super().__init__(name=f"src-{name}", daemon=True)
        self.src_name = name
        self.out: queue.Queue = queue.Queue(maxsize=20000)
        self._stop_evt = threading.Event()
        self.count = 0
        self.errors = 0
        self.last_error = ""
        self._rate_t0 = time.monotonic()
        self._rate_n = 0
        self.rate_hz = 0.0

    def emit(self, item):
        try:
            self.out.put_nowait(item)
        except queue.Full:
            self.errors += 1
            self.last_error = "output queue full (consumer too slow)"

    def tick_rate(self):
        self.count += 1
        self._rate_n += 1
        now = time.monotonic()
        if now - self._rate_t0 >= 1.0:
            self.rate_hz = self._rate_n / (now - self._rate_t0)
            self._rate_t0, self._rate_n = now, 0

    def drain(self) -> list:
        items = []
        while True:
            try:
                items.append(self.out.get_nowait())
            except queue.Empty:
                return items

    def stop(self):
        self._stop_evt.set()

    @property
    def stopped(self):
        return self._stop_evt.is_set()
