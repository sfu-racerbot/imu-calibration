"""Glue: raw samples -> calibration -> sync -> car-center fusion -> EKF (+ online calibration).

Live:    pipe = Pipeline(cfg, sources=[VescDriver(...), BnoDriver(...)]); pipe.start(); pipe.poll()
Offline: pipe = Pipeline(cfg, names=["vesc", "bno"]); pipe.feed(item) ...; pipe.process(now)
Lidar:   pipe.lidar_pose(t_meas, x, y, psi) / lidar_heading / lidar_velocity / lidar_yaw_rate
         t_meas = when the scan was taken (host monotonic clock); late arrivals are moved forward to "now".
While running, the gyro bias is re-learned every time the car parks and from every lidar heading.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from .bno_driver import BnoDriver
from .calibration import ImuCalibration
from .estimator import EkfParams, PlanarEkf
from .frames import CenterFuser, Config
from .online import (HealthMonitor, PoseHistory, StandstillDetector, load_online_bias,
                     save_online_bias)
from .source import Source
from .sync import Synchronizer, load_offsets
from .types import ImuSample, WheelSample
from .vesc_driver import VescDriver


def make_sources(cfg: Config, ports: dict[str, str] | None = None, only: list[str] | None = None) -> list[Source]:
    """Build drivers from car.yaml; `ports` overrides the configured port per IMU name."""
    ports = ports or {}
    out = []
    for name, ex in cfg.imus.items():
        if only and name not in only:
            continue
        port = ports.get(name, ex.extra.get("port"))
        if not port:
            continue
        if name.startswith("vesc"):
            out.append(VescDriver(port, name, rate_hz=ex.extra.get("rate_hz", 200), can_id=ex.extra.get("can_id")))
        else:
            out.append(BnoDriver(port, name, baud=ex.extra.get("baud", 921600)))
    return out


class Pipeline:
    def __init__(self, cfg: Config, sources: list[Source] | None = None, names: list[str] | None = None,
                 logger=None, use_calib: bool = True, use_sync: bool = True, history_s: float = 20.0,
                 calibs: dict | None = None, offsets: dict | None = None, persist_bias: bool = True):
        self.cfg = cfg
        self.sources = sources or []
        self.names = names or [s.src_name for s in self.sources]
        self.logger = logger
        if calibs is None:
            calibs = {n: ImuCalibration.load_or_identity(cfg.resolve(cfg.imus[n].extra.get("calib")))
                      if use_calib else ImuCalibration() for n in self.names}
        self.calibs = calibs
        sc = cfg.section("sync")
        if offsets is None:
            offsets = load_offsets(cfg.resolve(sc.get("file"))) if use_sync else {}
        self.sync = Synchronizer(self.names, sc.get("rate_hz", 200), sc.get("delay_s", 0.04), offsets)
        fc = cfg.section("fusion")
        self.fuser = CenterFuser({n: cfg.imus[n] for n in self.names}, fc.get("alpha_source", "dual_accel"),
                                 attitude_from=fc.get("attitude_from"))
        ep = EkfParams.from_dict(cfg.section("estimator"))
        self.persist_bias = persist_bias and ep.online_gyro_bias
        bias0 = load_online_bias(cfg, self.calibs) if self.persist_bias else 0.0
        self.ekf = PlanarEkf(ep, gyro_bias0=bias0)
        self.standstill = StandstillDetector()
        self.health = HealthMonitor()
        self.poses = PoseHistory()
        self.lidar_counts = {"accepted": 0, "rejected": 0}
        self._pose_initialized = False
        n_hist = int(history_s * sc.get("rate_hz", 200))
        self.history: deque[dict] = deque(maxlen=n_hist)
        self.latest_raw: dict[str, ImuSample] = {}
        self.latest_wheel: WheelSample | None = None

    def start(self):
        for s in self.sources:
            s.start()
        return self

    def stop(self):
        for s in self.sources:
            s.stop()
        for s in self.sources:
            s.join(timeout=1.0)
        if self.logger:
            self.logger.close()
        if self.persist_bias and self.history:
            self.saved_bias_path = save_online_bias(self.cfg, self.calibs, self.history[-1]["est"])

    def feed(self, item):
        if self.logger:
            self.logger.log(item)
        if isinstance(item, ImuSample):
            self.latest_raw[item.src] = item
            if item.src in self.calibs:
                item = self.calibs[item.src].apply(item)
        else:
            self.latest_wheel = item
        self.sync.push(item)

    def process(self, now: float) -> list[dict]:
        outs = []
        for fr in self.sync.pop_ready(now):
            c = self.fuser.update(fr.t, fr.imus)
            v_wheel = self.cfg.car.erpm_to_speed(fr.wheel_erpm) if np.isfinite(fr.wheel_erpm) else float("nan")
            still = self.standstill.update(fr.t, c.gyro, c.f_center, v_wheel)
            est = self.ekf.step(fr.t, c.acc[0], c.acc[1], c.gyro[2], v_wheel, standstill=still)
            self.health.update(fr.t, est, still, c.f_center)
            self.poses.add(fr.t, est["x"], est["y"], est["psi"])
            o = {"t": fr.t, "frame": fr, "center": c, "est": est, "v_wheel": v_wheel, "standstill": still}
            self.history.append(o)
            outs.append(o)
            if self.logger:
                self.logger.log_row("est", {**est, "v_wheel": v_wheel, "standstill": int(still),
                                            "ax_c": float(c.acc[0]),
                                            "ay_c": float(c.acc[1]), "roll": float(c.roll),
                                            "pitch": float(c.pitch),
                                            **{f"r_{n}": float((self.cfg.imus[n].R @ s.gyro)[2])
                                               for n, s in fr.imus.items()}})
        return outs

    # ---------------------------------------------------------------- lidar / external references
    def _count(self, ok: bool) -> bool:
        self.lidar_counts["accepted" if ok else "rejected"] += 1
        if ok and self.ekf.t is not None:
            self.health.reference(self.ekf.t)
        return ok

    def lidar_pose(self, t_meas: float, x: float, y: float, psi: float, sigma_xy: float = 0.05,
                   sigma_psi: float = np.radians(2.0)) -> bool:
        """Pose from scan matching / a particle filter, taken at t_meas. The first one sets the map frame."""
        m = self.poses.motion_since(t_meas)
        if m is not None:           # move the late measurement forward by our own motion since t_meas
            dx, dy, dpsi = m
            c, s = np.cos(psi), np.sin(psi)
            x, y, psi = x + c * dx - s * dy, y + s * dx + c * dy, psi + dpsi
        if not self._pose_initialized:
            self.ekf.set_pose(x, y, psi)
            self._pose_initialized = True
            self.poses.buf.clear()
            return self._count(True)
        return self._count(self.ekf.update_pose(x, y, psi, sigma_xy, sigma_psi))

    def lidar_heading(self, t_meas: float, psi: float, sigma: float = np.radians(2.0)) -> bool:
        m = self.poses.motion_since(t_meas)
        if m is not None:
            psi = psi + m[2]
        return self._count(self.ekf.update_heading(psi, sigma))

    def lidar_velocity(self, t_meas: float, vx: float, vy: float, sigma: float = 0.1) -> bool:
        return self._count(self.ekf.update_velocity(vx, vy, sigma))

    def lidar_yaw_rate(self, t_meas: float, r: float, sigma: float = 0.02) -> bool:
        return self._count(self.ekf.update_yaw_rate(r, sigma))

    def poll(self, now: float | None = None) -> list[dict]:
        for s in self.sources:
            for item in s.drain():
                self.feed(item)
        return self.process(time.monotonic() if now is None else now)

    def status(self) -> dict:
        st = {s.src_name: {"rate_hz": round(s.rate_hz, 1), "errors": s.errors, "last_error": s.last_error}
              for s in self.sources}
        for s in self.sources:
            if hasattr(s, "rtt_stats"):
                st[s.src_name].update(s.rtt_stats() or {})
        st["sync"] = {"produced": self.sync.produced, "skipped": self.sync.skipped}
        est = self.history[-1]["est"] if self.history else {}
        st["online"] = {"gyro_bias_deg_s": float(np.degrees(est.get("gyro_bias", 0.0))),
                        "std_deg_s": float(np.degrees(est.get("std_bias", 0.0))),
                        "standstill": self.standstill.still, "lidar": dict(self.lidar_counts),
                        "health": list(self.health.messages)}
        return st
