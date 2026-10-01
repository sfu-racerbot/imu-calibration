"""Independent checks of an IMU calibration, each against a physical truth the fit never used.

  still  a parked car does not rotate            -> gyro bias left over, heading drift per minute
  poses  gravity is g in EVERY orientation       -> accel bias/scale on NEW poses (out-of-sample)
  flip   turning the car 180 deg about vertical on the same spot flips the table's tilt but not what
         the sensor gets wrong, so (A + B) / 2 = how far from level the calibrated accel reads relative to
         the gyro's vertical: leftover bias + accel/gyro axis tilt (reversal test; table needn't be level)
  turn   N full turns by hand, back to the same line -> gyro SCALE error (not corrected by the calibration)

Samples fed in must already be calibrated (ImuCalibration.apply) except where noted.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .calibration import PoseCollector
from .types import G, ImuSample


class StillWindow:
    """Mean accel once the IMU has been still for hold_s (sliding window)."""

    def __init__(self, hold_s: float = 2.0, gyro_thr: float = 0.05, acc_thr: float = 0.1):
        self.hold_s, self.gyro_thr, self.acc_thr = hold_s, gyro_thr, acc_thr
        self.buf: deque = deque()
        self.moving = False

    def add(self, s: ImuSample):
        if np.linalg.norm(s.gyro) > self.gyro_thr:
            self.buf.clear()
            self.moving = True
            return None
        self.moving = False
        self.buf.append(s)
        while self.buf and self.buf[0].t < s.t - self.hold_s:
            self.buf.popleft()
        if len(self.buf) < 10 or s.t - self.buf[0].t < 0.95 * self.hold_s:
            return None
        acc = np.array([b.acc for b in self.buf])
        if acc.std(axis=0).max() > self.acc_thr:
            return None
        return acc.mean(axis=0)


def grade(value: float, good: float, ok: float) -> str:
    v = abs(value)
    return "PASS" if v <= good else ("OK" if v <= ok else "FAIL")


# ---------------------------------------------------------------- still
def check_still(samples: list[ImuSample]) -> dict:
    t = np.array([s.t for s in samples])
    acc = np.array([s.acc for s in samples])
    gyr = np.array([s.gyro for s in samples])
    up = acc.mean(axis=0) / np.linalg.norm(acc.mean(axis=0))
    w_up = gyr @ up                                   # rotation about the vertical = heading change
    span = t[-1] - t[0]
    heading = float(np.degrees(np.sum(w_up[1:] * np.diff(t))))
    moved = float(np.degrees(np.abs(gyr).max()))
    a_err = float(np.linalg.norm(acc.mean(axis=0)) - G)
    return {"seconds": span, "gyro_mean_deg_s": np.degrees(gyr.mean(axis=0)),
            "gyro_noise_deg_s": np.degrees(gyr.std(axis=0)), "heading_drift_deg_min": heading / span * 60,
            "acc_err": a_err, "max_rate_deg_s": moved,
            "grade_gyro": grade(heading / span * 60, 1.0, 3.0), "grade_acc": grade(a_err, 0.03, 0.06)}


# ---------------------------------------------------------------- new poses
class PoseCheck:
    """Auto-captures new static poses; compares |a| with g before and after calibration."""

    def __init__(self, cal, calib_poses=None, hold_s: float = 1.5):
        self.cal = cal
        self.pc = PoseCollector(hold_s=hold_s)
        self.calib_dirs = ([p / np.linalg.norm(p) for p in np.asarray(calib_poses)]
                           if calib_poses is not None and len(calib_poses) else [])
        self.rows = []

    def add(self, raw: ImuSample):
        """raw = as received from the device. Returns the new row when a pose is captured."""
        s = ImuSample(raw.t, raw.acc, raw.gyro - self.cal.eff_gyro_bias(), raw.quat, raw.src)
        m = self.pc.add(s)
        if m is None:
            return None
        c = self.cal.A_inv @ (m - self.cal.eff_acc_bias())
        u = m / np.linalg.norm(m)
        near = min((np.degrees(np.arccos(np.clip(np.dot(u, d), -1, 1))) for d in self.calib_dirs), default=np.nan)
        row = {"dir": u, "raw_err": float(np.linalg.norm(m) - G), "cal_err": float(np.linalg.norm(c) - G),
               "deg_from_calib_pose": float(near)}
        self.rows.append(row)
        return row

    def result(self) -> dict:
        raw = np.array([r["raw_err"] for r in self.rows])
        cal = np.array([r["cal_err"] for r in self.rows])
        rms_c = float(np.sqrt(np.mean(cal ** 2)))
        return {"n": len(self.rows), "raw_rms": float(np.sqrt(np.mean(raw ** 2))), "cal_rms": rms_c,
                "cal_max": float(np.abs(cal).max()), "grade": grade(rms_c, 0.04, 0.07)}


# ---------------------------------------------------------------- flip / turn
class TurnCheck:
    """Still (A) -> rotate about vertical -> still (B). Integrates the rotation in between."""

    def __init__(self, min_turn_deg: float = 90.0, hold_s: float = 3.0):
        self.win = StillWindow(hold_s=hold_s)
        self.min_turn = np.radians(min_turn_deg)
        self.phase = "A"            # A -> move -> B -> done
        self.A = self.B = self.up = None
        self.A_raw = self.B_raw = None
        self.yaw = 0.0
        self.rot = np.zeros(3)      # integrated rotation vector: for a turn about a fixed axis = angle * axis
        self._last_t = None
        self._raw_buf: deque = deque(maxlen=2000)

    def add(self, s: ImuSample, raw: ImuSample | None = None) -> str:
        """s calibrated; raw (optional) uncalibrated, for a before/after comparison. Returns the phase."""
        if raw is not None:
            self._raw_buf.append(raw.acc)
        if self.phase in ("move", "B") and self._last_t is not None:
            dt = s.t - self._last_t
            self.rot += s.gyro * dt
            self.yaw = float(np.linalg.norm(self.rot)) * np.sign(np.dot(self.rot, self.up))
        self._last_t = s.t
        m = self.win.add(s)
        if self.phase == "A" and m is not None:
            self.A, self.up = m, m / np.linalg.norm(m)
            self.A_raw = np.mean(list(self._raw_buf)[-int(len(self.win.buf)):], axis=0) if raw is not None else None
            self.phase = "move"
        elif self.phase == "move" and abs(self.yaw) > self.min_turn:
            self.phase = "B"
        elif self.phase == "B" and m is not None:
            self.B = m
            self.B_raw = np.mean(list(self._raw_buf)[-int(len(self.win.buf)):], axis=0) if raw is not None else None
            self.phase = "done"
        return self.phase


def _horizontal_basis(up):
    a = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(up, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(up, e1)


def flip_result(tc: TurnCheck, cal=None) -> dict:
    """Reversal test: how level the CALIBRATED accel reads relative to the axis the gyro saw the car turn about.

    The result mixes two things one flip cannot separate: leftover accel bias in the two horizontal axes,
    and a small tilt between the calibrated accel axes and the gyro axes (the ellipsoid fit fixes the
    accel's shape but not its rotation; 0.1 deg of tilt reads as g * 0.0017 = 0.017 m/s^2 here).
    Both matter in the same way when removing gravity while driving, so they are graded together.
    For information, the bias the reversal sees in the RAW data is shown next to the calibration's bias.
    """
    # Axes perpendicular to the axis the car actually turned about (measured by the gyro), NOT to gravity:
    # on a tilted table they differ, and gravity's tilt would leak into the result.
    axis = tc.rot / np.linalg.norm(tc.rot)
    e1, e2 = _horizontal_basis(axis)
    half = (tc.A - tc.B) / 2
    mid = (tc.A + tc.B) / 2
    res = np.array([mid @ e1, mid @ e2])
    out = {"turned_deg": float(np.degrees(tc.yaw)), "residual_bias_ms2": res,
           "level_error_deg": float(np.degrees(np.linalg.norm(res) / G)),
           "table_tilt_deg": float(np.degrees(np.arcsin(min(1.0, np.hypot(half @ e1, half @ e2) / G)))),
           "grade": grade(float(np.linalg.norm(res)), 0.06, 0.12)}
    if tc.A_raw is not None and tc.B_raw is not None:
        mr = (tc.A_raw + tc.B_raw) / 2
        out["raw_residual_ms2"] = np.array([mr @ e1, mr @ e2])
        if cal is not None:
            b = cal.eff_acc_bias()
            out["calib_bias_ms2"] = np.array([b @ e1, b @ e2])
    warn = []
    if abs(abs(np.degrees(tc.yaw)) - 180) > 25:
        warn.append("turned far from 180 deg")
    tA, tB = (np.degrees(np.arccos(np.clip(v @ axis / np.linalg.norm(v), -1, 1))) for v in (tc.A, tc.B))
    if abs(tA - tB) > 0.3:      # same table, same car: the tilt from the turn axis must not change
        warn.append(f"the car sat differently after the turn (tilt {tA:.2f} -> {tB:.2f} deg): it shifted/rocked")
    if warn:
        out["warning"] = "; ".join(warn) + ". Result unreliable, repeat slowly."
    return out


def turn_result(tc: TurnCheck, turns: float) -> dict:
    measured = abs(np.degrees(tc.yaw))
    expected = 360.0 * turns
    err_pct = (measured / expected - 1) * 100
    return {"measured_deg": measured, "expected_deg": expected, "scale_error_pct": err_pct,
            "grade": grade(err_pct, 1.0, 2.5)}
