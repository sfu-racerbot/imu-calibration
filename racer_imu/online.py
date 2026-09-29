"""Online (while-driving) calibration helpers: standstill detection, delayed lidar measurements,
health monitoring, and saving the learned gyro bias between sessions.

Nothing here runs on a timer. Corrections happen when there is evidence:
  - every time the car is parked      -> EKF re-zeroes the gyro (StandstillDetector)
  - every lidar pose/heading/velocity -> EKF corrects position/heading and keeps learning the gyro bias
A timer only drives the health checks, which *report* problems (they don't change the estimate).
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from .estimator import wrap
from .types import G


class StandstillDetector:
    """True while the car has been parked for at least window_s.

    Parked = wheels ~0 (if wheel speed is available) AND the IMU readings are steady (low spread).
    Spread, not value, is tested for the gyro, so an uncorrected gyro bias doesn't hide a standstill.
    Without wheel data, the gyro must also read near zero, so a steady turn isn't taken as parked.
    """

    def __init__(self, window_s: float = 0.5, wheel_max: float = 0.02, gyro_std_max: float = 0.01,
                 acc_std_max: float = 0.08, gyro_abs_max_no_wheel: float = 0.03):
        self.window_s = window_s
        self.wheel_max = wheel_max
        self.gyro_std_max = gyro_std_max
        self.acc_std_max = acc_std_max
        self.gyro_abs_max_no_wheel = gyro_abs_max_no_wheel
        self.buf: deque = deque()
        self.still = False
        self.since: float | None = None

    def update(self, t: float, gyro: np.ndarray, acc: np.ndarray, v_wheel: float) -> bool:
        wheel_ok = bool(abs(v_wheel) < self.wheel_max) if np.isfinite(v_wheel) else None   # plain bool: np.False_ is not False
        if wheel_ok is False:
            self.buf.clear()
            self.still, self.since = False, None
            return False
        self.buf.append((t, np.asarray(gyro, float), np.asarray(acc, float), wheel_ok))
        while self.buf and self.buf[0][0] < t - self.window_s:
            self.buf.popleft()
        if len(self.buf) < 5 or t - self.buf[0][0] < 0.9 * self.window_s:
            self.still = False
            return False
        g = np.array([b[1] for b in self.buf])
        a = np.array([b[2] for b in self.buf])
        steady = g.std(axis=0).max() < self.gyro_std_max and a.std(axis=0).max() < self.acc_std_max
        if steady and any(b[3] is None for b in self.buf):      # no wheel data: also require ~no rotation
            steady = np.abs(g.mean(axis=0)).max() < self.gyro_abs_max_no_wheel
        if steady and not self.still:
            self.since = t
        self.still = bool(steady)
        return self.still


class PoseHistory:
    """Recent EKF poses, to move a late lidar measurement forward to the current time."""

    def __init__(self, keep_s: float = 2.0):
        self.keep_s = keep_s
        self.buf: deque = deque()

    def add(self, t, x, y, psi):
        self.buf.append((t, x, y, psi))
        while self.buf and self.buf[0][0] < t - self.keep_s:
            self.buf.popleft()

    def at(self, t):
        """Interpolated (x, y, psi) at time t, or None if t is outside the stored history."""
        b = self.buf
        if not b or t < b[0][0] or t > b[-1][0]:
            return None
        ts = [p[0] for p in b]
        i = int(np.searchsorted(ts, t))
        if i == 0:
            return b[0][1:]
        (t0, x0, y0, p0), (t1, x1, y1, p1) = b[i - 1], b[i]
        u = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        return x0 + u * (x1 - x0), y0 + u * (y1 - y0), p0 + u * wrap(p1 - p0)

    def motion_since(self, t):
        """(dx, dy, dpsi) the car moved from time t to now, expressed in its frame at time t."""
        then = self.at(t)
        if then is None or not self.buf:
            return None
        _, xn, yn, pn = self.buf[-1]
        x0, y0, p0 = then
        c, s = np.cos(p0), np.sin(p0)
        dxw, dyw = xn - x0, yn - y0
        return c * dxw + s * dyw, -s * dxw + c * dyw, wrap(pn - p0)


class HealthMonitor:
    """Watches the online estimates and says when something needs attention. Reports only."""

    def __init__(self, bias_warn_rad_s: float = np.radians(0.3), accel_warn: float = 0.1,
                 no_reference_warn_s: float = 120.0, check_every_s: float = 1.0):
        self.bias_warn = bias_warn_rad_s
        self.accel_warn = accel_warn
        self.no_ref_warn_s = no_reference_warn_s
        self.check_every_s = check_every_s
        self.last_reference_t: float | None = None
        self._still_acc: deque = deque(maxlen=400)
        self._last_check = -np.inf
        self.messages: list[str] = []

    def reference(self, t: float):
        """Call whenever heading got an absolute correction (standstill or lidar)."""
        self.last_reference_t = t

    def update(self, t: float, est: dict, standstill: bool, f_center: np.ndarray):
        if standstill:
            self._still_acc.append(float(np.linalg.norm(f_center)))
            self.reference(t)
        if self.last_reference_t is None:
            self.last_reference_t = t
        if t - self._last_check < self.check_every_s:
            return
        self._last_check = t
        msgs = []
        bg = est.get("gyro_bias", 0.0)
        if abs(bg) > self.bias_warn:
            msgs.append(f"gyro bias moved {np.degrees(bg):+.2f} deg/s since calibration "
                        "(online-corrected; re-run --step gyro if it keeps growing)")
        if len(self._still_acc) > 100:
            err = float(np.mean(self._still_acc)) - G
            if abs(err) > self.accel_warn:
                msgs.append(f"parked |a| - g = {err:+.3f} m/s^2: accel calibration looks stale, recalibrate")
        gap = t - self.last_reference_t
        if gap > self.no_ref_warn_s:
            msgs.append(f"no heading reference for {gap:.0f} s (no stop, no lidar): heading drifting uncorrected")
        self.messages = msgs


# ---------------------------------------------------------------- persistence of the learned bias
def online_bias_path(cfg) -> Path:
    sync_file = cfg.resolve(cfg.section("sync").get("file", "calib/sync.json"))
    return sync_file.parent / "online_gyro_bias.json"


def calib_stamps(calibs: dict) -> dict:
    return {n: c.meta.get("updated", "") for n, c in calibs.items()}


def load_online_bias(cfg, calibs: dict) -> float:
    """Last session's learned residual bias, if it was learned on top of the same calibration files."""
    p = online_bias_path(cfg)
    if not p.exists():
        return 0.0
    try:
        d = json.loads(p.read_text())
    except (ValueError, OSError):
        return 0.0
    if d.get("calib_stamps") != calib_stamps(calibs):
        return 0.0            # recalibrated since: the old residual no longer applies
    return float(d.get("gyro_bias_rad_s", 0.0))


def save_online_bias(cfg, calibs: dict, est: dict, max_std: float = np.radians(0.1)) -> Path | None:
    if not est or est.get("std_bias", 1.0) > max_std:
        return None           # not learned well enough to be worth keeping
    p = online_bias_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "gyro_bias_rad_s": est["gyro_bias"], "gyro_bias_deg_s": float(np.degrees(est["gyro_bias"])),
        "std_deg_s": float(np.degrees(est["std_bias"])), "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
        "calib_stamps": calib_stamps(calibs),
        "note": "residual gyro z bias learned while driving, on top of calib/*.json; loaded at next start"},
        indent=2))
    return p
