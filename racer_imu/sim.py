"""Car + IMU simulator, so the whole stack can be developed without hardware.

- Trajectory: ground-truth rigid-body motion of the car center at 1 kHz for a scenario
  (still, circle, slalom, drift, shake, poses).
- SensorTruth: what each simulated IMU gets wrong, on purpose and with KNOWN values: scale/cross-axis,
  bias, noise, mounting position/rotation, processing delay, device clock offset and drift.
- SimServer: live mode. Opens pseudo-terminals (/dev/pts/N) that behave like the real devices:
  a fake VESC that answers COMM_GET_IMU_DATA / COMM_GET_VALUES_SELECTIVE, and a fake MCU that streams
  BNO085 lines. The real drivers connect to them unchanged.
- generate_offline(): the same sensor model without threads/ptys, for fast tests.

Run standalone:  python -m racer_imu.sim --scenario drift
"""
from __future__ import annotations

import argparse
import json
import os
import select
import threading
import time
import tty
from dataclasses import dataclass, field

import numpy as np

from . import vesc_protocol as vp
from .bno_driver import format_line
from .clock_sync import ClockMapper
from .frames import Config, R_to_quat, R_to_rpy, load_config, rot_z, slerp, quat_to_R, rpy_to_R
from .types import G, ImuSample, WheelSample

SCENARIOS = ("still", "push", "circle", "slalom", "drift", "stopgo", "shake", "poses")


# ---------------------------------------------------------------- helpers
def _expm_so3(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _log_so3(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-9:
        return 0.5 * v
    return th / (2 * np.sin(th)) * v


def _smooth(x, n):
    if n <= 1:
        return x
    k = np.ones(n) / n
    pad = np.pad(x, (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(pad, k, mode="valid")


# ---------------------------------------------------------------- trajectory
class Trajectory:
    def __init__(self, scenario: str = "drift", fs: float = 1000.0, seed: int = 0):
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario}; choose from {SCENARIOS}")
        self.scenario = scenario
        self.fs = fs
        self.rng = np.random.default_rng(seed)
        getattr(self, f"_build_{scenario}")()
        self.duration = self.t[-1] + 1.0 / fs
        # pose at the end of one loop, to make repeated loops continuous
        self.end_pose = (self.p[-1, 0], self.p[-1, 1], self.psi[-1])

    # --- planar scenarios: body-frame vx, vy, r and wheel slip over time
    def _planar(self, T, kf):
        """kf: list of (t, speed, yaw_rate, beta_deg, wheel_slip) keyframes, linearly blended + smoothed."""
        fs = self.fs
        t = np.arange(0, T, 1 / fs)
        kf = np.array(kf, float)
        n = int(0.4 * fs)
        V = _smooth(np.interp(t, kf[:, 0], kf[:, 1]), n)
        r = _smooth(np.interp(t, kf[:, 0], kf[:, 2]), n)
        beta = _smooth(np.radians(np.interp(t, kf[:, 0], kf[:, 3])), n)
        slip = _smooth(np.interp(t, kf[:, 0], kf[:, 4]), n)
        self._set_planar(t, V * np.cos(beta), V * np.sin(beta), r, slip)

    def _set_planar(self, t, vx, vy, r, slip):
        dt = 1 / self.fs
        psi = np.concatenate([[0.0], np.cumsum(r[:-1]) * dt])
        vwx = vx * np.cos(psi) - vy * np.sin(psi)
        vwy = vx * np.sin(psi) + vy * np.cos(psi)
        px = np.concatenate([[0.0], np.cumsum(vwx[:-1]) * dt])
        py = np.concatenate([[0.0], np.cumsum(vwy[:-1]) * dt])
        N = len(t)
        self.t = t
        self.p = np.stack([px, py, np.zeros(N)], 1)
        self.psi = psi
        c, s = np.cos(psi), np.sin(psi)
        self.R = np.zeros((N, 3, 3))
        self.R[:, 0, 0], self.R[:, 0, 1], self.R[:, 1, 0], self.R[:, 1, 1], self.R[:, 2, 2] = c, -s, s, c, 1
        self.w = np.stack([np.zeros(N), np.zeros(N), r], 1)
        self.al = np.stack([np.zeros(N), np.zeros(N), np.gradient(r, dt)], 1)
        self.a = np.stack([np.gradient(vx, dt) - r * vy, np.gradient(vy, dt) + r * vx, np.zeros(N)], 1)
        self.v = np.stack([vx, vy, np.zeros(N)], 1)
        self.v_wheel = vx * (1 + slip)

    def _set_rotation_only(self, t, R):
        dt = 1 / self.fs
        N = len(t)
        w = np.zeros((N, 3))
        for k in range(N - 1):
            w[k] = _log_so3(R[k].T @ R[k + 1]) / dt
        w[-1] = w[-2]
        self.t, self.R, self.w = t, R, w
        self.al = np.gradient(w, dt, axis=0)
        self.p = np.zeros((N, 3))
        self.psi = np.array([np.arctan2(Rk[1, 0], Rk[0, 0]) for Rk in R])
        self.a = np.zeros((N, 3))
        self.v = np.zeros((N, 3))
        self.v_wheel = np.zeros(N)

    def _build_still(self):
        t = np.arange(0, 20, 1 / self.fs)
        self._set_planar(t, *(np.zeros(len(t)) for _ in range(4)))

    def _build_push(self):
        """4 s standing still, then a straight forward push (for the mount-yaw step)."""
        self._planar(14, [(0, 0, 0, 0, 0), (4, 0, 0, 0, 0), (5, 1.5, 0, 0, 0), (7, 1.5, 0, 0, 0),
                          (9, 0, 0, 0, 0), (14, 0, 0, 0, 0)])

    def _build_stopgo(self):
        """Drive a loop, park 3 s, drive the other way, park... (tests re-zeroing the gyro at stops)."""
        kf, t = [(0, 0, 0, 0, 0)], 0.0
        for k in range(5):
            r = 1.0 if k % 2 == 0 else -1.0
            kf += [(t + 1.5, 2.0, r, -2, 0.02), (t + 8.0, 2.0, r, -2, 0.02), (t + 9.5, 0, 0, 0, 0), (t + 12.5, 0, 0, 0, 0)]
            t += 12.5
        self._planar(t, kf)

    def _build_circle(self):
        self._planar(40, [(0, 0, 0, 0, 0), (3, 2.0, 0, 0, 0.03), (4, 2.0, 1.33, -2, 0.02),
                          (36, 2.0, 1.33, -2, 0.02), (38, 0, 0, 0, -0.05), (40, 0, 0, 0, 0)])

    def _build_slalom(self):
        T = 40
        t = np.arange(0, T, 1 / self.fs)
        ramp = np.clip(t / 3, 0, 1) * np.clip((T - t) / 3, 0, 1)
        V = 2.5 * ramp
        r = 1.6 * np.sin(2 * np.pi * 0.35 * t) * ramp
        beta = -0.06 * r
        self._set_planar(t, V * np.cos(beta), V * np.sin(beta), r, 0.03 * ramp)

    def _build_drift(self):
        self._planar(30, [(0, 0, 0, 0, 0), (3, 3.0, 0, 0, 0.06), (4, 3.0, 0, 0, 0.01),
                          (5, 3.0, 2.0, -3, 0.02), (8, 3.0, 2.0, -3, 0.02),
                          (10, 3.0, 2.6, -25, 0.35), (14, 3.0, 2.6, -25, 0.35),
                          (16, 3.0, 1.0, -2, 0.05), (17, 3.0, 0, 0, 0.01), (22, 3.0, 0, 0, 0.01),
                          (25, 0, 0, 0, -0.1), (30, 0, 0, 0, 0)])

    def _build_shake(self):
        T, fs = 20.0, self.fs
        t = np.arange(0, T, 1 / fs)
        env = np.clip((t - 2) / 1.0, 0, 1) * np.clip((T - 2 - t) / 1.0, 0, 1)
        w = np.zeros((len(t), 3))
        for ax, amp in zip(range(3), (1.5, 1.5, 2.5)):
            for f in self.rng.uniform(0.5, 3.0, 3):
                w[:, ax] += amp / 3 * np.sin(2 * np.pi * f * t + self.rng.uniform(0, 2 * np.pi))
        w *= env[:, None]
        R = np.zeros((len(t), 3, 3))
        R[0] = np.eye(3)
        for k in range(len(t) - 1):
            R[k + 1] = R[k] @ _expm_so3(w[k] / fs)
        self._set_rotation_only(t, R)

    def _build_poses(self):
        hold, move, fs = 2.5, 1.0, self.fs
        axis = [rpy_to_R(0, 0, 0), rpy_to_R(np.pi, 0, 0), rpy_to_R(np.pi / 2, 0, 0),
                rpy_to_R(-np.pi / 2, 0, 0), rpy_to_R(0, np.pi / 2, 0), rpy_to_R(0, -np.pi / 2, 0)]
        rand = [rpy_to_R(*self.rng.uniform(-np.pi, np.pi, 3)) for _ in range(16)]
        poses = axis + rand
        qs = [R_to_quat(R) for R in poses]
        Rs = []
        for i, q in enumerate(qs):
            Rs += [quat_to_R(q)] * int(hold * fs)
            if i + 1 < len(qs):
                n = int(move * fs)
                for k in range(n):
                    u = 0.5 - 0.5 * np.cos(np.pi * k / n)
                    Rs.append(quat_to_R(slerp(q, qs[i + 1], u)))
        self.pose_list = poses
        self._set_rotation_only(np.arange(len(Rs)) / fs, np.array(Rs))

    # --- lookup
    def index(self, t_sim: float):
        k = int(round((t_sim % self.duration) * self.fs))
        return min(k, len(self.t) - 1)

    def truth(self, t_sim: float) -> dict:
        """Planar truth, continuous across loop repeats."""
        n = int(t_sim // self.duration)
        k = self.index(t_sim)
        x, y, psi = self.p[k, 0], self.p[k, 1], self.psi[k]
        ex, ey, epsi = self.end_pose
        for _ in range(n):   # compose n loop-end poses
            c, s = np.cos(epsi), np.sin(epsi)
            x, y, psi = ex + c * x - s * y, ey + s * x + c * y, epsi + psi
        vx, vy = self.v[k, 0], self.v[k, 1]
        speed = np.hypot(vx, vy)
        return {"x": x, "y": y, "psi": psi, "vx": vx, "vy": vy, "r": self.w[k, 2],
                "beta": float(np.arctan2(vy, vx)) if speed > 0.3 else 0.0, "v_wheel": self.v_wheel[k]}


# ---------------------------------------------------------------- sensors
@dataclass
class SensorTruth:
    name: str
    r: np.ndarray                                  # position in the car frame [m]
    R: np.ndarray                                  # sensor -> car
    scale: np.ndarray = field(default_factory=lambda: np.eye(3))   # raw = scale @ f + bias
    acc_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acc_noise: float = 0.02
    gyro_noise: float = 0.002
    delay: float = 0.0                             # measurement is this much older than its timestamp
    world_yaw: float = 0.0                         # the sensor's AHRS heading reference is arbitrary
    clock_offset: float = 0.0                      # device clock at sim t=0 [s]
    clock_ppm: float = 0.0
    usb_latency_min: float = 0.0
    usb_latency_mean_extra: float = 0.0
    gyro_bias_drift: np.ndarray = field(default_factory=lambda: np.zeros(3))   # bias change [rad/s per s]

    def measure(self, traj: Trajectory, t_sim: float, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k = traj.index(t_sim - self.delay)
        Rwb, w, al, a = traj.R[k], traj.w[k], traj.al[k], traj.a[k]
        f_b = a + np.cross(al, self.r) + np.cross(w, np.cross(w, self.r)) + Rwb.T @ np.array([0, 0, G])
        f_s = self.R.T @ f_b
        w_s = self.R.T @ w
        acc = self.scale @ f_s + self.acc_bias + rng.normal(0, self.acc_noise, 3)
        gyro = w_s + self.gyro_bias + self.gyro_bias_drift * t_sim + rng.normal(0, self.gyro_noise, 3)
        quat = R_to_quat(rot_z(self.world_yaw) @ Rwb @ self.R)
        return acc, gyro, quat


def default_truth(cfg: Config) -> dict[str, SensorTruth]:
    """Known errors injected by the simulator. Mounting comes from car.yaml so the GUI matches."""
    ex = cfg.imus
    out = {}
    if "vesc" in ex:
        out["vesc"] = SensorTruth(
            "vesc", ex["vesc"].r, ex["vesc"].R,
            scale=np.array([[1.020, 0.008, -0.004], [0.008, 0.980, 0.006], [-0.004, 0.006, 1.010]]),
            acc_bias=np.array([0.15, -0.10, 0.25]), gyro_bias=np.array([0.010, -0.020, 0.015]),
            acc_noise=0.03, gyro_noise=0.003, delay=0.004)
    if "bno" in ex:
        out["bno"] = SensorTruth(
            "bno", ex["bno"].r, ex["bno"].R,
            scale=np.array([[1.005, 0.002, 0.0], [0.002, 0.995, -0.003], [0.0, -0.003, 1.000]]),
            acc_bias=np.array([0.03, 0.02, -0.05]), gyro_bias=np.array([0.002, -0.001, 0.003]),
            acc_noise=0.02, gyro_noise=0.002, delay=0.012, world_yaw=0.7,
            clock_offset=12.345, clock_ppm=40.0, usb_latency_min=0.001, usb_latency_mean_extra=0.0015)
    return out


def expected_sync_offset(truth: dict[str, SensorTruth], ref: str = "vesc", other: str = "bno") -> float:
    """The dt tools/sync_calibrate.py should find for `other` (VESC stamps at reply time)."""
    return truth[ref].delay - (truth[other].delay + truth[other].usb_latency_min)


def _vesc_values(acc, gyro, quat) -> dict:
    rpy = R_to_rpy(quat_to_R(quat))
    a, g = acc / G, np.degrees(gyro)
    return {"roll": rpy[0], "pitch": rpy[1], "yaw": rpy[2], "acc_x": a[0], "acc_y": a[1], "acc_z": a[2],
            "gyro_x": g[0], "gyro_y": g[1], "gyro_z": g[2], "mag_x": 0.0, "mag_y": 0.0, "mag_z": 0.0,
            "q0": quat[0], "q1": quat[1], "q2": quat[2], "q3": quat[3]}


# ---------------------------------------------------------------- offline generator
def generate_offline(cfg: Config, scenario: str, duration: float | None = None, truth=None, seed: int = 0,
                     vesc_rate: float = 200.0, bno_rate: float = 200.0, wheel_every: int = 4, t0: float = 1000.0):
    """Streams as the drivers would deliver them (host-clock stamps, raw values). Returns (streams, traj, truth)."""
    traj = Trajectory(scenario, seed=seed)
    truth = truth or default_truth(cfg)
    rng = np.random.default_rng(seed + 1)
    duration = duration or traj.duration
    out = {n: [] for n in truth}
    out["wheel"] = []
    if "vesc" in truth:
        st = truth["vesc"]
        for k in range(int(duration * vesc_rate)):
            t_send = k / vesc_rate + rng.uniform(0, 2e-4)
            rtt = 0.0008 + rng.exponential(0.0003)
            t_mid = t_send + rtt / 2
            acc, gyro, quat = st.measure(traj, t_mid, rng)
            out["vesc"].append(ImuSample(t0 + t_mid, acc, gyro, quat, "vesc", rtt=rtt))
            if wheel_every and k % wheel_every == 0:
                out["wheel"].append(WheelSample(t0 + t_mid, cfg.car.speed_to_erpm(traj.v_wheel[traj.index(t_mid)])))
    if "bno" in truth:
        st = truth["bno"]
        cm = ClockMapper()
        for k in range(int(duration * bno_rate)):
            t_meas = k / bno_rate
            acc, gyro, quat = st.measure(traj, t_meas, rng)
            t_dev = st.clock_offset + t_meas * (1 + st.clock_ppm * 1e-6)
            t_dev = int(t_dev * 1e6) * 1e-6
            t_recv = t_meas + st.usb_latency_min + rng.exponential(st.usb_latency_mean_extra)
            # t_recv is on the sim clock (starts at 0); shift everything to the host epoch t0
            out["bno"].append(ImuSample(t0 + cm.add(t_dev, t_recv), acc, gyro, quat, "bno", t_dev=t_dev))
    return out, traj, truth


# ---------------------------------------------------------------- live pty server
class SimServer:
    def __init__(self, cfg: Config, scenario: str = "drift", seed: int = 0, truth=None,
                 bno_rate: float = 200.0):
        self.cfg = cfg
        self.traj = Trajectory(scenario, seed=seed)
        self.truth_sensors = truth or default_truth(cfg)
        self.rng = np.random.default_rng(seed + 1)
        self.bno_rate = bno_rate
        self._stop = threading.Event()
        self.ports: dict[str, str] = {}
        self.last_drive: dict = {}
        self._fds = []
        self._threads = []
        self.t0 = None

    def _pty(self):
        m, s = os.openpty()
        tty.setraw(m)
        tty.setraw(s)
        self._fds += [m, s]
        return m, os.ttyname(s)

    def start(self):
        self.t0 = time.monotonic()
        if "vesc" in self.truth_sensors:
            m, name = self._pty()
            self.ports["vesc"] = name
            self._threads.append(threading.Thread(target=self._vesc_loop, args=(m,), daemon=True))
        if "bno" in self.truth_sensors:
            m, name = self._pty()
            self.ports["bno"] = name
            self._threads.append(threading.Thread(target=self._bno_loop, args=(m,), daemon=True))
        for th in self._threads:
            th.start()
        return self

    def sim_time(self, t_host: float) -> float:
        return t_host - self.t0

    def truth_at(self, t_host: float) -> dict:
        return self.traj.truth(self.sim_time(t_host))

    def _vesc_loop(self, fd):
        st = self.truth_sensors["vesc"]
        parser = vp.FrameParser()
        while not self._stop.is_set():
            r, _, _ = select.select([fd], [], [], 0.1)
            if not r:
                continue
            try:
                data = os.read(fd, 4096)
            except OSError:
                time.sleep(0.05)
                continue
            for p in parser.feed(data):
                ts = time.monotonic() - self.t0
                if p[0] == vp.COMM_GET_IMU_DATA:
                    acc, gyro, quat = st.measure(self.traj, ts, self.rng)
                    reply = vp.encode_imu_reply(_vesc_values(acc, gyro, quat))
                elif p[0] == vp.COMM_GET_VALUES_SELECTIVE:
                    v = self.traj.v_wheel[self.traj.index(ts)]
                    reply = vp.encode_values_selective_reply(
                        {"rpm": self.cfg.car.speed_to_erpm(v), "v_in": 16.4, "tachometer": 0})
                else:
                    self._record_drive(p)            # set commands have no reply
                    continue
                os.write(fd, vp.pack_frame(reply))

    def _record_drive(self, p: bytes):
        import struct
        cmd = p[0]
        if cmd in (vp.COMM_SET_DUTY, vp.COMM_SET_CURRENT, vp.COMM_SET_CURRENT_BRAKE):
            scale = 1e5 if cmd == vp.COMM_SET_DUTY else 1e3
            self.last_drive[cmd] = struct.unpack(">i", p[1:5])[0] / scale
        elif cmd == vp.COMM_SET_SERVO_POS:
            self.last_drive[cmd] = struct.unpack(">h", p[1:3])[0] / 1e3
        self.last_drive["n"] = self.last_drive.get("n", 0) + 1
        self.last_drive["last_cmd"] = cmd

    def _bno_loop(self, fd):
        st = self.truth_sensors["bno"]
        os.write(fd, b"# sim bno085_stream ready\r\n")
        k = 0
        while not self._stop.is_set():
            t_meas = k / self.bno_rate
            t_send = t_meas + st.usb_latency_min + self.rng.exponential(st.usb_latency_mean_extra)
            dt = self.t0 + t_send - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            acc, gyro, quat = st.measure(self.traj, t_meas, self.rng)
            t_us = int((st.clock_offset + t_meas * (1 + st.clock_ppm * 1e-6)) * 1e6)
            try:
                os.write(fd, format_line(t_us, acc, gyro, quat).encode())
            except OSError:
                pass
            k += 1

    def stop(self):
        self._stop.set()
        for th in self._threads:
            th.join(timeout=1)
        for fd in self._fds:
            try:
                os.close(fd)
            except OSError:
                pass

    def describe(self) -> dict:
        return {"scenario": self.traj.scenario, "ports": self.ports,
                "expected_sync_offset_bno_s": expected_sync_offset(self.truth_sensors)
                if {"vesc", "bno"} <= set(self.truth_sensors) else None,
                "sensors": {n: {"acc_bias": s.acc_bias.tolist(), "gyro_bias": s.gyro_bias.tolist(),
                                "scale": s.scale.tolist(), "delay_s": s.delay,
                                "mount_rpy_deg": np.degrees(R_to_rpy(s.R)).tolist(), "position_m": s.r.tolist()}
                            for n, s in self.truth_sensors.items()}}


def main():
    ap = argparse.ArgumentParser(description="Fake VESC + BNO085 on pseudo-terminals")
    ap.add_argument("--scenario", default="drift", choices=SCENARIOS)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    srv = SimServer(load_config(a.config), a.scenario, a.seed).start()
    print(json.dumps(srv.describe(), indent=2))
    print(f"\nConnect with e.g.:  python tools/imu_monitor.py --vesc {srv.ports.get('vesc')} "
          f"--bno {srv.ports.get('bno')}\nCtrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        srv.stop()


if __name__ == "__main__":
    main()
