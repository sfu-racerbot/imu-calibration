"""Planar EKF for position, heading, body velocity, yaw rate and the gyro's z bias -> sideslip (drift) angle.

State s = [x, y, psi, vx, vy, r, bg]
    world x/y [m], heading [rad], car-frame velocity [m/s], true yaw rate [rad/s],
    bg = what is left of the gyro z bias after calibration [rad/s] (estimated online, drifts with temperature)
Prediction (car-center linear acceleration a = (ax, ay) from CenterFuser, gravity removed):
    x'  = vx cos(psi) - vy sin(psi)      vx' = ax + r vy
    y'  = vx sin(psi) + vy cos(psi)      vy' = ay - r vx
    psi' = r                             r'  = random walk,   bg' = slow random walk
Updates:
    gyro z            -> r + bg        (the gyro measures the true rate plus its bias)
    wheel speed       -> vx            (VESC ERPM; assumes the driven wheels are not spinning much)
    no-slip (weak)    -> vy = 0        only while lateral acceleration is small, so real drift isn't hidden
    standstill        -> vx = vy = 0, and r = 0 when a standstill detector confirms it; with the gyro update
                         this re-zeroes the gyro every time the car stops (no timer)
    lidar (optional)  -> pose / heading / yaw rate / velocity, each gated against outliers. A lidar heading
                         makes bg observable while driving, so the gyro keeps being calibrated without stops.
Sideslip beta = atan2(vy, vx). IMU-only velocity drifts within seconds; wheel speed is what anchors it.
Limitation: while the wheels spin (power slide), wheel speed over-reads vx and nothing else observes vy,
so beta is biased by a few degrees (tests/test_estimator.py measures it). A lidar velocity fixes it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

X, Y, PSI, VX, VY, R, BG = range(7)
N = 7
CHI2_99 = {1: 6.63, 2: 9.21, 3: 11.34}   # 99 % gates for 1/2/3-dof innovations


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class EkfParams:
    sigma_acc: float = 0.5          # accel process noise [m/s^2]
    sigma_yaw_acc: float = 8.0      # yaw-rate random walk [rad/s^2]
    sigma_pos: float = 0.01         # extra position process noise [m/sqrt(s)]
    sigma_gyro: float = 0.01        # gyro z measurement noise [rad/s]
    sigma_wheel: float = 0.08       # wheel speed noise [m/s]
    sigma_nonholo: float = 0.15     # weak vy = 0 pseudo-measurement [m/s]
    nonholo_max_ay: float = 1.0     # only apply vy=0 when |ay| below this [m/s^2]
    zupt_speed: float = 0.03        # standstill: |wheel speed| below [m/s]
    zupt_gyro: float = 0.03         # ... and |r| below [rad/s]
    beta_min_speed: float = 0.3     # report beta = 0 below this speed [m/s]
    online_gyro_bias: bool = True   # keep estimating the gyro z bias while running
    sigma_bias0: float = 0.0087     # initial bias uncertainty [rad/s] (0.5 deg/s)
    sigma_bias_rw: float = 3e-4     # bias random walk [rad/s/sqrt(s)] (~0.1 deg/s per minute)
    sigma_standstill_r: float = 0.002   # how sure "not rotating" is when parked [rad/s]
    max_consecutive_rejects: int = 3    # after this many gated-out readings in a row, trust the sensor again

    @classmethod
    def from_dict(cls, d: dict | None):
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class PlanarEkf:
    def __init__(self, params: EkfParams | None = None, gyro_bias0: float = 0.0):
        self.p = params or EkfParams()
        self.s = np.zeros(N)
        self.s[BG] = gyro_bias0
        sb = self.p.sigma_bias0 if self.p.online_gyro_bias else 0.0
        self.P = np.diag([1e-4, 1e-4, 1e-4, 0.1, 0.1, 0.1, sb ** 2])
        self.t = None
        self.last_ay = 0.0
        self.rejected = {}           # gated-out measurements per kind
        self._reject_run = {}        # consecutive rejections per kind

    # ---- prediction
    def predict(self, t: float, ax: float, ay: float):
        if self.t is None:
            self.t = t
            return
        dt = t - self.t
        self.t = t
        if dt <= 0 or dt > 0.5:
            return
        x, y, psi, vx, vy, r, bg = self.s
        c, s = np.cos(psi), np.sin(psi)
        self.s = np.array([
            x + (vx * c - vy * s) * dt,
            y + (vx * s + vy * c) * dt,
            psi + r * dt,
            vx + (ax + r * vy) * dt,
            vy + (ay - r * vx) * dt,
            r,
            bg,
        ])
        F = np.eye(N)
        F[X, PSI], F[X, VX], F[X, VY] = (-vx * s - vy * c) * dt, c * dt, -s * dt
        F[Y, PSI], F[Y, VX], F[Y, VY] = (vx * c - vy * s) * dt, s * dt, c * dt
        F[PSI, R] = dt
        F[VX, VY], F[VX, R] = r * dt, vy * dt
        F[VY, VX], F[VY, R] = -r * dt, -vx * dt
        q = self.p
        brw = q.sigma_bias_rw ** 2 * dt if q.online_gyro_bias else 0.0
        Q = np.diag([q.sigma_pos ** 2 * dt, q.sigma_pos ** 2 * dt, 0.0,
                     q.sigma_acc ** 2 * dt, q.sigma_acc ** 2 * dt, q.sigma_yaw_acc ** 2 * dt, brw])
        self.P = F @ self.P @ F.T + Q
        self.last_ay = ay

    # ---- generic update: z = H s (+ noise R); optional chi-square gate; angle rows wrapped
    def _update(self, H, z, Rm, gate: bool = False, angle_rows=(), kind: str = "") -> bool:
        H = np.atleast_2d(np.asarray(H, float))
        z = np.atleast_1d(np.asarray(z, float))
        Rm = np.atleast_2d(np.asarray(Rm, float))
        y = z - H @ self.s
        for i in angle_rows:
            y[i] = wrap(y[i])
        S = H @ self.P @ H.T + Rm
        if gate:
            d2 = float(y @ np.linalg.solve(S, y))
            run = self._reject_run.get(kind, 0)
            # One odd reading is an outlier. A whole run of them means the filter is the one that's wrong
            # (e.g. wheel spin fooled it), so accept the sensor again instead of locking it out.
            if d2 > CHI2_99[len(z)] and run < self.p.max_consecutive_rejects:
                self.rejected[kind] = self.rejected.get(kind, 0) + 1
                self._reject_run[kind] = run + 1
                return False
            self._reject_run[kind] = 0
        K = self.P @ H.T @ np.linalg.inv(S)
        self.s = self.s + K @ y
        I_KH = np.eye(N) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ Rm @ K.T      # Joseph form
        return True

    @staticmethod
    def _row(*idx_val):
        h = np.zeros(N)
        for i, v in idx_val:
            h[i] = v
        return h

    def update_gyro(self, r_meas: float):
        self._update(self._row((R, 1), (BG, 1)), r_meas, self.p.sigma_gyro ** 2)

    def update_wheel(self, v_wheel: float):
        if not np.isfinite(v_wheel):
            return
        self._update(self._row((VX, 1)), v_wheel, self.p.sigma_wheel ** 2)
        if abs(v_wheel) < self.p.zupt_speed and abs(self.s[R]) < self.p.zupt_gyro:
            self._update(self._row((VX, 1)), 0.0, 1e-4)
            self._update(self._row((VY, 1)), 0.0, 1e-4)
        elif abs(self.last_ay) < self.p.nonholo_max_ay and abs(self.s[VX]) > self.p.beta_min_speed:
            self._update(self._row((VY, 1)), 0.0, self.p.sigma_nonholo ** 2)

    def update_standstill(self):
        """Confirmed parked (see StandstillDetector): not moving and not rotating. Re-zeroes the gyro."""
        self._update(self._row((VX, 1)), 0.0, 1e-4)
        self._update(self._row((VY, 1)), 0.0, 1e-4)
        self._update(self._row((R, 1)), 0.0, self.p.sigma_standstill_r ** 2)

    # ---- lidar / external references (all gated)
    def update_velocity(self, vx: float, vy: float, sigma: float = 0.1) -> bool:
        """Car-frame velocity from lidar odometry / scan matching / mocap."""
        H = np.vstack([self._row((VX, 1)), self._row((VY, 1))])
        return self._update(H, [vx, vy], np.eye(2) * sigma ** 2, gate=True, kind="velocity")

    def update_heading(self, psi: float, sigma: float = np.radians(2.0)) -> bool:
        return self._update(self._row((PSI, 1)), psi, sigma ** 2, gate=True, angle_rows=(0,), kind="heading")

    def update_yaw_rate(self, r: float, sigma: float = 0.02) -> bool:
        """True yaw rate from lidar odometry. Differs from the gyro by exactly the bias."""
        return self._update(self._row((R, 1)), r, sigma ** 2, gate=True, kind="yaw_rate")

    def update_pose(self, x: float, y: float, psi: float, sigma_xy: float = 0.05,
                    sigma_psi: float = np.radians(2.0)) -> bool:
        H = np.vstack([self._row((X, 1)), self._row((Y, 1)), self._row((PSI, 1))])
        Rm = np.diag([sigma_xy ** 2, sigma_xy ** 2, sigma_psi ** 2])
        return self._update(H, [x, y, psi], Rm, gate=True, angle_rows=(2,), kind="pose")

    def set_pose(self, x: float, y: float, psi: float):
        """Hard reset, e.g. to put the EKF into the lidar map frame once at start-up."""
        self.s[[X, Y, PSI]] = [x, y, psi]
        self.P[[X, Y, PSI], :] = 0.0
        self.P[:, [X, Y, PSI]] = 0.0
        self.P[X, X] = self.P[Y, Y] = 1e-4
        self.P[PSI, PSI] = 1e-4

    def step(self, t: float, ax: float, ay: float, r_meas: float, v_wheel: float = float("nan"),
             standstill: bool = False):
        self.predict(t, ax, ay)
        self.update_gyro(r_meas)
        self.update_wheel(v_wheel)
        if standstill:
            self.update_standstill()
        return self.output(v_wheel)

    def output(self, v_wheel: float = float("nan")) -> dict:
        x, y, psi, vx, vy, r, bg = self.s
        speed = float(np.hypot(vx, vy))
        beta = float(np.arctan2(vy, vx)) if speed > self.p.beta_min_speed else 0.0
        slip = float((v_wheel - vx) / max(abs(vx), 0.5)) if np.isfinite(v_wheel) else float("nan")
        return {"t": self.t, "x": float(x), "y": float(y), "psi": float(psi), "vx": float(vx), "vy": float(vy),
                "r": float(r), "speed": speed, "beta": beta, "wheel_slip": slip,
                "gyro_bias": float(bg), "std_bias": float(np.sqrt(max(self.P[BG, BG], 0.0))),
                "std_vy": float(np.sqrt(self.P[VY, VY])), "std_psi": float(np.sqrt(self.P[PSI, PSI]))}
