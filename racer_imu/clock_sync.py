"""Clock tools.

ClockMapper: map a device clock (e.g. BNO085/MCU micros) onto the host monotonic clock.
    Each packet gives (t_dev, t_recv). t_recv - t_dev = clock offset + transport delay, where the delay
    is always >= its minimum. So the lower envelope of (t_recv - t_dev) over t_dev is the offset line
    (plus the minimum transport delay, which is absorbed by the sync offset later).
    We keep per-bin minima over a sliding window and fit a line through them, which also tracks the
    device crystal drifting (tens of ppm).

estimate_time_offset: cross-correlate two signals to find their relative delay (used on |gyro|,
    which is identical for every IMU on a rigid body and independent of how each is mounted).
"""
from __future__ import annotations

from collections import deque

import numpy as np


class ClockMapper:
    def __init__(self, window_s: float = 30.0, bin_s: float = 0.5, refit_every_s: float = 0.5):
        self.window_s = window_s
        self.bin_s = bin_s
        self.refit_every_s = refit_every_s
        self.bins: deque[list[float]] = deque()  # [bin_start_dev, t_dev_at_min, min_offset]
        self.slope = 0.0      # drift: offset changes by slope * t_dev
        self.intercept = None
        self._last_fit = -np.inf

    def add(self, t_dev: float, t_recv: float) -> float:
        off = t_recv - t_dev
        if not self.bins or t_dev >= self.bins[-1][0] + self.bin_s:
            self.bins.append([t_dev, t_dev, off])
        elif off < self.bins[-1][2]:
            self.bins[-1][1:] = [t_dev, off]
        while self.bins and self.bins[0][0] < t_dev - self.window_s:
            self.bins.popleft()
        if self.intercept is None or t_dev - self._last_fit >= self.refit_every_s:
            self._fit()
            self._last_fit = t_dev
        return self.to_host(t_dev)

    def _fit(self):
        # the newest bin is still filling and may not contain a low-latency packet yet
        pts = list(self.bins)[:-1] if len(self.bins) > 3 else list(self.bins)
        x = np.array([p[1] for p in pts])
        y = np.array([p[2] for p in pts])
        if len(pts) >= 4 and np.ptp(x) > 2.0:
            self.slope, self.intercept = np.polyfit(x - x[0], y, 1)
            self.intercept -= self.slope * x[0]
        else:
            self.slope, self.intercept = 0.0, float(y.min())

    def to_host(self, t_dev: float) -> float:
        return t_dev + self.intercept + self.slope * t_dev


class Unwrapper:
    """Unwrap a free-running unsigned counter (e.g. uint32 micros that wraps every ~71.6 min)."""

    def __init__(self, bits: int = 32):
        self.mod = 1 << bits
        self.last = None
        self.offset = 0

    def __call__(self, raw: int) -> int:
        if self.last is not None and raw < self.last and self.last - raw > self.mod // 2:
            self.offset += self.mod
        self.last = raw
        return raw + self.offset


def estimate_time_offset(t_a, x_a, t_b, x_b, max_lag: float = 0.25, fs: float = 1000.0):
    """Return (dt, peak_corr): ADD dt to stream b's timestamps to align it with stream a.

    Signals are resampled to fs on their overlap, detrended, cross-correlated, and the peak is refined
    with a parabola fit, which gives sub-sample (sub-millisecond) resolution.
    """
    t_a, x_a, t_b, x_b = map(np.asarray, (t_a, x_a, t_b, x_b))
    t0 = max(t_a[0], t_b[0]) + max_lag
    t1 = min(t_a[-1], t_b[-1]) - max_lag
    if t1 - t0 < 4 * max_lag:
        raise ValueError("not enough overlapping data to estimate the time offset")
    grid = np.arange(t0, t1, 1.0 / fs)
    a = np.interp(grid, t_a, x_a)
    a = a - a.mean()
    nlag = int(round(max_lag * fs))
    # b evaluated on shifted grids: corr[k] = sum a(t) * b(t + lag_k)
    lags = np.arange(-nlag, nlag + 1)
    tb = np.asarray(t_b)
    corr = np.empty(len(lags))
    for i, k in enumerate(lags):
        b = np.interp(grid + k / fs, tb, x_b)
        b = b - b.mean()
        corr[i] = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    i = int(np.argmax(corr))
    frac = 0.0
    if 0 < i < len(corr) - 1:
        y0, y1, y2 = corr[i - 1], corr[i], corr[i + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            frac = 0.5 * (y0 - y2) / den
    lag = (lags[i] + frac) / fs
    # x_b(t + lag) ~ x_a(t): an event shows up in b `lag` seconds later, so b's stamps are late by lag
    return -lag, float(corr[i])
