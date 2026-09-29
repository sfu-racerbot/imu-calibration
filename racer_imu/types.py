"""Shared sample types. Everything inside racer_imu is SI: m/s^2, rad/s, s, m."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

G = 9.80665  # standard gravity [m/s^2]

NAN3 = np.full(3, np.nan)
NAN4 = np.full(4, np.nan)


@dataclass
class ImuSample:
    t: float                      # best estimate of measurement time, host monotonic clock [s]
    acc: np.ndarray               # specific force in the sensor frame [m/s^2] (reads +g up when still)
    gyro: np.ndarray              # angular rate in the sensor frame [rad/s]
    quat: np.ndarray = field(default_factory=lambda: NAN4.copy())  # w, x, y, z (sensor -> world), may be NaN
    src: str = ""
    t_dev: float = float("nan")   # device timestamp [s] if the device sends one
    rtt: float = float("nan")     # request round-trip time [s] (VESC only)
    label: str = ""               # free-form tag, e.g. calibration pose name


@dataclass
class WheelSample:
    t: float                      # host monotonic time [s]
    erpm: float                   # electrical RPM from the VESC
    v_in: float = float("nan")    # battery voltage [V]
    tacho: float = float("nan")   # tachometer count
    src: str = "vesc"
