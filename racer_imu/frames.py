"""Rotations, car geometry and moving IMU readings to the car center frame.

Car frame (ROS REP-103): x forward, y left, z up, origin at the car center set in config/car.yaml.
Extrinsic R_i maps sensor-frame vectors into the car frame:  v_car = R_i @ v_sensor.
Quaternions are (w, x, y, z) and map sensor -> world.

Rigid-body relations for an IMU at lever arm r from the center:
    omega_car          = R_i @ omega_sensor                           (same everywhere on the car)
    f_imu (car frame)  = f_center + alpha x r + omega x (omega x r)    (specific force)
so  f_center = R_i @ f_sensor - alpha x r - omega x (omega x r).
With two accelerometers, f_1 - f_2 = alpha x d + omega x (omega x d), d = r_1 - r_2, which gives the
components of alpha perpendicular to d without differentiating the gyro.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .types import G, ImuSample

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "car.yaml"


# ---------------------------------------------------------------- rotations
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rpy_to_R(roll, pitch, yaw):
    """Z-Y-X (yaw, then pitch, then roll) as in ROS/tf."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def R_to_rpy(R):
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def quat_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def R_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q if q[0] >= 0 else -q


def slerp(q0, q1, u):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + u * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - u) * th) * q0 + np.sin(u * th) * q1) / np.sin(th)


def angle_between(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))


# ---------------------------------------------------------------- config
@dataclass
class CarGeometry:
    length: float = 0.55
    width: float = 0.30
    wheelbase: float = 0.33
    track: float = 0.26
    wheel_radius: float = 0.05
    wheel_width: float = 0.045
    speed_to_erpm_gain: float = 4614.0   # erpm = gain * v + offset (same meaning as the f1tenth vesc.yaml)
    speed_to_erpm_offset: float = 0.0

    def erpm_to_speed(self, erpm: float) -> float:
        return (erpm - self.speed_to_erpm_offset) / self.speed_to_erpm_gain

    def speed_to_erpm(self, v: float) -> float:
        return v * self.speed_to_erpm_gain + self.speed_to_erpm_offset


@dataclass
class ImuExtrinsic:
    name: str
    r: np.ndarray                       # position in the car frame [m]
    R: np.ndarray                       # sensor -> car rotation
    weight: float = 1.0
    extra: dict = field(default_factory=dict)   # port, baud, calib path, ...

    @property
    def rpy(self):
        return R_to_rpy(self.R)


@dataclass
class Config:
    car: CarGeometry
    imus: dict[str, ImuExtrinsic]
    raw: dict
    path: Path

    def section(self, name: str) -> dict:
        return self.raw.get(name, {}) or {}

    def resolve(self, p: str | None) -> Path | None:
        if not p:
            return None
        p = Path(p)
        return p if p.is_absolute() else self.path.parent.parent / p


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    raw = yaml.safe_load(path.read_text())
    car = CarGeometry(**(raw.get("car") or {}))
    imus = {}
    for name, d in (raw.get("imus") or {}).items():
        d = dict(d)
        r = np.array(d.pop("position", [0, 0, 0]), float)
        rpy = np.radians(d.pop("rpy_deg", [0, 0, 0]))
        w = float(d.pop("weight", 1.0))
        imus[name] = ImuExtrinsic(name, r, rpy_to_R(*rpy), w, d)
    return Config(car, imus, raw, path)


# ---------------------------------------------------------------- center-frame fusion
def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


@dataclass
class CenterState:
    t: float
    gyro: np.ndarray            # omega at the car, car frame [rad/s]
    alpha: np.ndarray           # angular acceleration [rad/s^2]
    f_center: np.ndarray        # specific force at the center, car frame [m/s^2]
    acc: np.ndarray             # linear acceleration at the center (gravity removed), car frame
    roll: float
    pitch: float
    per_imu: dict               # name -> {"f_car": f at the IMU in car frame, "f_center": its estimate at center}


class CenterFuser:
    """Turn synced per-IMU samples into one set of car-center quantities."""

    def __init__(self, extrinsics: dict[str, ImuExtrinsic], alpha_source: str = "dual_accel",
                 alpha_tau: float = 0.03, attitude_from: list[str] | None = None):
        self.ext = extrinsics
        self.alpha_source = alpha_source
        self.alpha_tau = alpha_tau
        self.attitude_from = attitude_from or list(extrinsics)
        self._last_t = None
        self._last_w = None
        self._alpha = np.zeros(3)

    def update(self, t: float, samples: dict[str, ImuSample]) -> CenterState:
        names = [n for n in samples if n in self.ext]
        ws = np.array([self.ext[n].weight for n in names])
        ws = ws / ws.sum()
        f_car = {n: self.ext[n].R @ samples[n].acc for n in names}
        w_car = {n: self.ext[n].R @ samples[n].gyro for n in names}
        omega = sum(wt * w_car[n] for wt, n in zip(ws, names))

        # angular acceleration: low-passed gyro derivative ...
        alpha_raw = np.zeros(3)
        if self._last_t is not None and t > self._last_t:
            alpha_raw = (omega - self._last_w) / (t - self._last_t)
        # ... with the components perpendicular to the IMU baseline taken from the two accelerometers
        if self.alpha_source == "dual_accel" and len(names) >= 2:
            a, b = names[0], names[1]
            d = self.ext[a].r - self.ext[b].r
            if np.linalg.norm(d) > 0.05:
                rhs = f_car[a] - f_car[b] - np.cross(omega, np.cross(omega, d))
                alpha_perp = np.linalg.lstsq(-skew(d), rhs, rcond=None)[0]   # min-norm: no component along d
                dh = d / np.linalg.norm(d)
                alpha_raw = alpha_perp + np.dot(alpha_raw, dh) * dh
        if self._last_t is not None:
            k = 1.0 - np.exp(-(t - self._last_t) / self.alpha_tau) if t > self._last_t else 0.0
            self._alpha = self._alpha + k * (alpha_raw - self._alpha)
        self._last_t, self._last_w = t, omega
        alpha = self._alpha

        per = {}
        for n in names:
            r = self.ext[n].r
            fc = f_car[n] - np.cross(alpha, r) - np.cross(omega, np.cross(omega, r))
            per[n] = {"f_car": f_car[n], "f_center": fc}
        f_center = sum(wt * per[n]["f_center"] for wt, n in zip(ws, names))

        roll = pitch = 0.0
        for n in self.attitude_from:
            if n in samples and np.all(np.isfinite(samples[n].quat)):
                R_wc = quat_to_R(samples[n].quat) @ self.ext[n].R.T
                roll, pitch, _ = R_to_rpy(R_wc)
                break
        R_level = rot_y(pitch) @ rot_x(roll)             # car -> yaw-free world
        g_car = R_level.T @ np.array([0.0, 0.0, G])      # what a still accelerometer reads, in car frame
        return CenterState(t, omega, alpha.copy(), f_center, f_center - g_car, roll, pitch, per)
