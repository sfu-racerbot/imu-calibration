"""Put several IMU streams (and wheel speed) on one common time grid.

Samples arrive from different threads at different rates. Each stream's timestamps first get the
constant offset measured by tools/sync_calibrate.py (calib/sync.json). Then, for every grid time
t_k = k / rate that lies at least `delay_s` in the past, each stream is interpolated at t_k: linear for
acc/gyro, slerp for the quaternion. A frame is only produced when every IMU stream has samples on
both sides of t_k and within max_gap_s; otherwise that grid time is skipped and counted.
"""
from __future__ import annotations

import json
import math
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .frames import slerp
from .types import ImuSample, WheelSample


def interp_imu(s0: ImuSample, s1: ImuSample, t: float) -> ImuSample:
    u = 0.0 if s1.t == s0.t else (t - s0.t) / (s1.t - s0.t)
    q = s0.quat
    if np.all(np.isfinite(s0.quat)) and np.all(np.isfinite(s1.quat)):
        q = slerp(s0.quat, s1.quat, u)
    return ImuSample(t, s0.acc + u * (s1.acc - s0.acc), s0.gyro + u * (s1.gyro - s0.gyro), q, s0.src)


def resample(samples: list[ImuSample], grid) -> list[ImuSample | None]:
    """Offline helper: interpolate a time-sorted list of samples at each grid time (None outside)."""
    ts = [s.t for s in samples]
    out = []
    for t in grid:
        i = bisect_right(ts, t)
        if i == 0 or i == len(ts):
            out.append(None)
        else:
            out.append(interp_imu(samples[i - 1], samples[i], t))
    return out


@dataclass
class SyncedFrame:
    t: float
    imus: dict                       # name -> ImuSample interpolated at t
    wheel_erpm: float = float("nan")
    gap: dict = field(default_factory=dict)   # name -> distance to nearest real sample [s]


def load_offsets(path) -> dict:
    if path and Path(path).exists():
        return json.loads(Path(path).read_text()).get("offsets", {})
    return {}


class Synchronizer:
    def __init__(self, names, rate_hz: float = 200.0, delay_s: float = 0.04, offsets: dict | None = None,
                 max_gap_s: float = 0.05, stall_s: float = 0.3):
        self.names = list(names)
        self.stall_s = stall_s
        self.dt = 1.0 / rate_hz
        self.delay_s = delay_s
        self.offsets = {n: float(v) for n, v in (offsets or {}).items()}
        self.max_gap_s = max_gap_s
        self.buf: dict[str, deque[ImuSample]] = {n: deque() for n in self.names}
        self.wheel: deque[WheelSample] = deque(maxlen=400)
        self.k: int | None = None
        self.skipped = 0
        self.produced = 0

    def push(self, s):
        if isinstance(s, WheelSample):
            s = replace(s, t=s.t + self.offsets.get("wheel", 0.0))
            self.wheel.append(s)
            return
        if s.src not in self.buf:
            return
        s = replace(s, t=s.t + self.offsets.get(s.src, 0.0))
        b = self.buf[s.src]
        if b and s.t <= b[-1].t:            # out of order / duplicate: drop
            return
        b.append(s)

    def _wheel_at(self, t):
        if not self.wheel:
            return float("nan")
        w = self.wheel
        if t <= w[0].t:
            return w[0].erpm
        if t >= w[-1].t:
            return w[-1].erpm if t - w[-1].t < 0.2 else float("nan")
        for a, b in zip(reversed(list(w)[:-1]), reversed(list(w)[1:])):
            if a.t <= t <= b.t:
                u = (t - a.t) / (b.t - a.t) if b.t > a.t else 0.0
                return a.erpm + u * (b.erpm - a.erpm)
        return float("nan")

    def pop_ready(self, now: float) -> list[SyncedFrame]:
        if any(not self.buf[n] for n in self.names):
            return []
        if self.k is None:
            t_start = max(self.buf[n][0].t for n in self.names)
            self.k = math.ceil(t_start / self.dt)
        out = []
        newest = max(self.buf[n][-1].t for n in self.names)
        while True:
            t = self.k * self.dt
            if t > now - self.delay_s or t > newest:
                break
            # a stream that is merely late (not stalled) holds the grid back instead of losing frames
            if any(self.buf[n][-1].t < t and newest - self.buf[n][-1].t < self.stall_s for n in self.names):
                break
            frame_imus, gaps, ok = {}, {}, True
            for n in self.names:
                b = self.buf[n]
                while len(b) >= 2 and b[1].t <= t:
                    b.popleft()
                if len(b) < 2 or not (b[0].t <= t <= b[1].t):
                    ok = False
                    break
                s0, s1 = b[0], b[1]
                if s1.t - s0.t > self.max_gap_s:
                    ok = False
                    break
                frame_imus[n] = interp_imu(s0, s1, t)
                gaps[n] = min(t - s0.t, s1.t - t)
            if ok:
                out.append(SyncedFrame(t, frame_imus, self._wheel_at(t), gaps))
                self.produced += 1
            else:
                self.skipped += 1
            self.k += 1
        return out
