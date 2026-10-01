"""IMU calibration.

Accelerometer: ellipsoid fit, the same model Magneto uses (michaelwro/accelerometer-calibration):
    calibrated = A_inv @ (raw - bias),   |calibrated| = g in every static pose.
A_inv is symmetric (scale + cross-axis). With < 12 poses only a diagonal A_inv is fitted
(6 axis-aligned poses cannot pin down cross-axis terms).
Gyro: bias = mean while perfectly still.
Mount (extrinsic) rotation: roll/pitch from gravity while level, yaw from a straight forward push.

Device offsets: the VESC can subtract accel/gyro offsets itself (App Settings -> IMU). acc_bias/gyro_bias
here are always the TOTAL bias of the sensor; dev_*_offset records what the device already subtracts,
and apply() removes only the remainder, so nothing is corrected twice (tools/vesc_offsets.py).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .frames import R_to_rpy, rot_z
from .types import G, ImuSample


@dataclass
class ImuCalibration:
    A_inv: np.ndarray = field(default_factory=lambda: np.eye(3))
    acc_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    meta: dict = field(default_factory=dict)
    dev_acc_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))    # already subtracted by the device [m/s^2]
    dev_gyro_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))   # already subtracted by the device [rad/s]

    def eff_acc_bias(self) -> np.ndarray:
        """Bias still present in the data the device sends."""
        return self.acc_bias - self.dev_acc_offset

    def eff_gyro_bias(self) -> np.ndarray:
        return self.gyro_bias - self.dev_gyro_offset

    def apply(self, s: ImuSample) -> ImuSample:
        return ImuSample(s.t, self.A_inv @ (s.acc - self.eff_acc_bias()), s.gyro - self.eff_gyro_bias(), s.quat,
                         s.src, s.t_dev, s.rtt, s.label)

    def to_dict(self):
        return {"A_inv": self.A_inv.tolist(), "acc_bias": self.acc_bias.tolist(),
                "gyro_bias": self.gyro_bias.tolist(), "units": "SI (m/s^2, rad/s)",
                "device_offsets": {"accel_g": (self.dev_acc_offset / G).tolist(),
                                   "gyro_deg_s": np.degrees(self.dev_gyro_offset).tolist(),
                                   "note": "offsets the device itself subtracts (VESC App Settings -> IMU); "
                                           "only the remainder of acc_bias/gyro_bias is subtracted in racer_imu"},
                "meta": self.meta}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        dev = d.get("device_offsets", {})
        return cls(np.array(d["A_inv"]), np.array(d["acc_bias"]), np.array(d["gyro_bias"]), d.get("meta", {}),
                   np.array(dev.get("accel_g", [0.0, 0.0, 0.0])) * G,
                   np.radians(dev.get("gyro_deg_s", [0.0, 0.0, 0.0])))

    @classmethod
    def load_or_identity(cls, path):
        if path and Path(path).exists():
            return cls.load(path)
        return cls(meta={"note": "identity (no calibration file)"})


# ---------------------------------------------------------------- accelerometer
def _sym(p):
    xx, yy, zz, xy, xz, yz = p
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def fit_accel(points, g: float = G, model: str = "auto"):
    """Fit A_inv, bias to static-pose mean readings (N x 3). Returns (A_inv, bias, info)."""
    P = np.asarray(points, float)
    n = len(P)
    if n < 6:
        raise ValueError(f"need at least 6 poses, got {n}")
    if model == "auto":
        model = "full" if n >= 12 else "diag"
    s0 = g / np.median(np.linalg.norm(P, axis=1))

    def unpack(x):
        b = x[:3]
        A = _sym(x[3:]) if model == "full" else np.diag(x[3:])
        return A, b

    def res(x):
        A, b = unpack(x)
        return np.linalg.norm((P - b) @ A.T, axis=1) - g

    x0 = np.r_[np.zeros(3), [s0, s0, s0, 0, 0, 0] if model == "full" else [s0, s0, s0]]
    sol = least_squares(res, x0)
    A, b = unpack(sol.x)
    info = {"model": model, "n_poses": n,
            "resid_before_ms2": float(np.std(np.linalg.norm(P, axis=1) - g)),
            "rms_before_ms2": float(np.sqrt(np.mean((np.linalg.norm(P, axis=1) - g) ** 2))),
            "rms_after_ms2": float(np.sqrt(np.mean(sol.fun ** 2)))}
    return A, b, info


def gyro_bias(gyros) -> np.ndarray:
    return np.mean(np.asarray(gyros, float), axis=0)


def detect_static_segments(t, acc, gyro, window_s=1.0, gyro_thr=0.05, acc_thr=0.15):
    """Return [(i0, i1)] index ranges where the IMU is still for at least window_s."""
    t, acc, gyro = map(np.asarray, (t, acc, gyro))
    still = np.zeros(len(t), bool)
    j = 0
    for i in range(len(t)):
        while t[i] - t[j] > window_s:
            j += 1
        if t[i] - t[0] < window_s:
            continue
        seg = slice(j, i + 1)
        still[i] = (np.max(np.linalg.norm(gyro[seg], axis=1)) < gyro_thr and
                    np.max(np.std(acc[seg], axis=0)) < acc_thr)
    segs, start = [], None
    for i, s in enumerate(still):
        if s and start is None:
            start = i
        if not s and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, len(t)))
    return segs


class PoseCollector:
    """Online auto-capture of distinct static poses for the ellipsoid fit.

    A pose is captured once the IMU has been still for hold_s and gravity points at least
    min_angle_deg away from every pose captured so far. No key presses needed: just turn the car/board
    to a new orientation and hold it.
    """

    def __init__(self, hold_s=1.5, gyro_thr=0.06, acc_thr=0.12, min_angle_deg=25.0):
        self.hold_s, self.gyro_thr, self.acc_thr = hold_s, gyro_thr, acc_thr
        self.min_cos = np.cos(np.radians(min_angle_deg))
        self.buf: list[ImuSample] = []
        self.poses: list[np.ndarray] = []
        self.pose_samples: list[list[ImuSample]] = []
        # live status for the user: state in {moving, holding, settling, same, captured}
        self.state, self.held_s, self.nearest_deg, self.nearest_idx = "moving", 0.0, 180.0, -1
        self.direction = np.array([0.0, 0.0, 1.0])

    def nearest_pose(self, u):
        """(angle_deg, index) of the captured pose closest to direction u."""
        if not self.poses:
            return 180.0, -1
        c = [np.dot(u, p / np.linalg.norm(p)) for p in self.poses]
        i = int(np.argmax(c))
        return float(np.degrees(np.arccos(np.clip(c[i], -1, 1)))), i

    def add(self, s: ImuSample):
        """Returns the new pose mean if this sample completed a capture, else None."""
        n = np.linalg.norm(s.acc)
        if n > 0:
            self.direction = s.acc / n
        self.nearest_deg, self.nearest_idx = self.nearest_pose(self.direction)
        g = np.linalg.norm(s.gyro)
        if g > self.gyro_thr:
            self.buf.clear()
            self.state, self.held_s = "moving", 0.0
            return None
        self.buf.append(s)
        self.held_s = self.buf[-1].t - self.buf[0].t
        if self.held_s < self.hold_s:
            self.state = "holding"
            return None
        while self.buf[-1].t - self.buf[1].t >= self.hold_s:   # sliding window of hold_s
            self.buf.pop(0)
        acc = np.array([b.acc for b in self.buf])
        if np.max(np.std(acc, axis=0)) > self.acc_thr:
            self.state = "settling"
            return None                   # still settling: slide on
        m = acc.mean(axis=0)
        u = m / np.linalg.norm(m)
        if any(np.dot(u, p / np.linalg.norm(p)) > self.min_cos for p in self.poses):
            self.state = "same"
            return None                   # same pose as one we already have: keep waiting
        self.state = "captured"

        self.poses.append(m)
        self.pose_samples.append(list(self.buf))
        self.buf.clear()
        return m


# ---------------------------------------------------------------- mounting (extrinsic rotation)
def level_rotation(acc_static) -> np.ndarray:
    """Smallest rotation R (sensor -> car) that makes gravity read straight +z: R @ a = |a| z."""
    u = np.asarray(acc_static, float)
    u = u / np.linalg.norm(u)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(u, z)
    s, c = np.linalg.norm(v), np.dot(u, z)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s
    th = np.arctan2(s, c)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def mount_rotation(acc_static, acc_push=None, push_thr=0.8):
    """Sensor -> car rotation from a level static average and (optionally) samples of a forward push.

    acc_push: (N x 3) raw readings from the moment the car is pushed straight forward from rest.
    The first strong horizontal acceleration defines car +x.
    """
    R = level_rotation(acc_static)
    yaw = 0.0
    if acc_push is not None and len(acc_push):
        h = (np.asarray(acc_push) @ R.T)[:, :2]   # after leveling, gravity is only on z
        mag = np.linalg.norm(h, axis=1)
        idx = np.flatnonzero(mag > push_thr)
        if len(idx) == 0:
            raise ValueError("no push detected: push harder or lower push_thr")
        first = idx[0]
        # take the samples of the first acceleration pulse (until magnitude drops again)
        end = first
        while end < len(mag) and mag[end] > push_thr:
            end += 1
        hv = h[first:end].mean(axis=0)
        yaw = np.arctan2(hv[1], hv[0])
    return rot_z(-yaw) @ R


def kabsch(src, dst) -> np.ndarray:
    """Rotation R minimising |dst - R src| over rows (dst_i ~ R @ src_i)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    H = src.T @ dst
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def rpy_deg(R):
    return np.degrees(R_to_rpy(R))


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")
