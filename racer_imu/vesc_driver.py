"""Poll the VESC IMU (and motor ERPM for wheel speed) over USB serial.

The IMU reply carries no device timestamp, so each sample is stamped with the midpoint between sending
the request and receiving the reply. Replies with a round-trip far above the recent median are marked
(rtt_ok=False in the label) and dropped, since their midpoint is unreliable.
Close VESC Tool first: only one program can hold the serial port.

Driving: set_drive() queues a throttle/steering command that is written between IMU polls, so the same
program can drive and log. Commands are a dead-man switch: each one expires after hold_s (0.3 s) unless
refreshed, and then the motor is released (current 0). The VESC's own app timeout (default 1 s) is a
second safety net if this program dies.
"""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np
import serial

from . import vesc_protocol as vp
from .source import Source
from .types import G, ImuSample, WheelSample


def imu_reply_to_sample(d: dict, t: float, rtt: float, name: str) -> ImuSample:
    acc = np.array([d["acc_x"], d["acc_y"], d["acc_z"]]) * G
    gyro = np.radians([d["gyro_x"], d["gyro_y"], d["gyro_z"]])
    quat = np.array([d["q0"], d["q1"], d["q2"], d["q3"]])
    return ImuSample(t, acc, gyro, quat, name, rtt=rtt)


class VescDriver(Source):
    def __init__(self, port: str, name: str = "vesc", rate_hz: float = 200.0, wheel_every: int = 4,
                 can_id: int | None = None, timeout_s: float = 0.05, rtt_reject_factor: float = 3.0):
        super().__init__(name)
        self.port = port
        self.period = 1.0 / rate_hz
        self.wheel_every = wheel_every
        self.can_id = can_id
        self.timeout_s = timeout_s
        self.rtt_reject_factor = rtt_reject_factor
        self.parser = vp.FrameParser()
        self.rtts: deque[float] = deque(maxlen=200)
        self.rejected = 0
        self.ser = None
        self._req_imu = vp.build_get_imu()
        self._req_val = vp.build_get_values_selective()
        if can_id is not None:
            self._req_imu = vp.forward_can(can_id, self._req_imu)
            self._req_val = vp.forward_can(can_id, self._req_val)
        self._drive_lock = threading.Lock()
        self._drive: dict | None = None
        self._drive_active = False
        self.drive_sent = 0

    # ---------------------------------------------------------------- driving
    def set_drive(self, duty: float | None = None, current: float | None = None, brake: float | None = None,
                  servo: float | None = None, hold_s: float = 0.3):
        """Throttle: duty (-1..1) or current [A]; brake [A] overrides both. servo 0..1. Refresh < hold_s."""
        with self._drive_lock:
            self._drive = {"duty": duty, "current": current, "brake": brake, "servo": servo,
                           "expires": time.monotonic() + hold_s}

    def stop_drive(self):
        with self._drive_lock:
            self._drive = {"expires": 0.0}

    def _frame(self, f: bytes) -> bytes:
        return vp.forward_can(self.can_id, f) if self.can_id is not None else f

    def _drive_frames(self) -> list[bytes]:
        with self._drive_lock:
            d = self._drive
        if d is None:
            return []
        if time.monotonic() > d["expires"]:
            if self._drive_active:                 # dead-man: release the motor once
                self._drive_active = False
                return [self._frame(vp.build_set_current(0.0))]
            return []
        self._drive_active = True
        if d.get("brake"):
            out = [vp.build_set_current_brake(d["brake"])]
        elif d.get("duty") is not None:
            out = [vp.build_set_duty(d["duty"])]
        elif d.get("current") is not None:
            out = [vp.build_set_current(d["current"])]
        else:
            out = [vp.build_set_current(0.0)]
        if d.get("servo") is not None:
            out.append(vp.build_set_servo_pos(d["servo"]))
        return [self._frame(f) for f in out]

    def _request(self, req: bytes, cmd: int):
        if self.ser.in_waiting > 512:      # stale replies piled up (e.g. after a timeout)
            self.ser.reset_input_buffer()
            self.parser = vp.FrameParser()
        t_send = time.monotonic()
        self.ser.write(req)
        deadline = t_send + self.timeout_s
        while time.monotonic() < deadline:
            data = self.ser.read(max(1, self.ser.in_waiting))
            if not data:
                continue
            t_recv = time.monotonic()
            for p in self.parser.feed(data):
                if p[0] == cmd:
                    return p, t_send, t_recv
        self.errors += 1
        self.last_error = f"timeout waiting for cmd {cmd}"
        return None, t_send, None

    def run(self):
        try:
            self.ser = serial.Serial(self.port, 115200, timeout=0.005)
        except serial.SerialException as e:
            self.last_error = str(e)
            self.errors += 1
            return
        k = 0
        next_t = time.monotonic()
        while not self.stopped:
            try:
                p, t0, t1 = self._request(self._req_imu, vp.COMM_GET_IMU_DATA)
                if p is not None:
                    rtt = t1 - t0
                    med = float(np.median(self.rtts)) if len(self.rtts) > 20 else rtt
                    self.rtts.append(rtt)
                    if rtt <= self.rtt_reject_factor * med + 0.002:
                        self.emit(imu_reply_to_sample(vp.parse_imu(p), 0.5 * (t0 + t1), rtt, self.src_name))
                        self.tick_rate()
                    else:
                        self.rejected += 1
                if k % 2 == 1:                     # drive commands at half the poll rate (100 Hz)
                    for f in self._drive_frames():
                        self.ser.write(f)
                        self.drive_sent += 1
                if self.wheel_every and k % self.wheel_every == 0:
                    p, t0, t1 = self._request(self._req_val, vp.COMM_GET_VALUES_SELECTIVE)
                    if p is not None:
                        d = vp.parse_values_selective(p)
                        self.emit(WheelSample(0.5 * (t0 + t1), d.get("rpm", np.nan), d.get("v_in", np.nan),
                                              d.get("tachometer", np.nan), self.src_name))
            except (serial.SerialException, OSError) as e:
                self.errors += 1
                self.last_error = str(e)
                time.sleep(0.5)
                continue
            except (ValueError, KeyError) as e:   # malformed payload
                self.errors += 1
                self.last_error = f"parse: {e}"
            k += 1
            next_t += self.period
            dt = next_t - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            else:
                next_t = time.monotonic()   # running behind: don't try to catch up in a burst
        if self._drive is not None:            # we drove at some point: always leave the motor released
            try:
                self.ser.write(self._frame(vp.build_set_current(0.0)))
                self.ser.flush()
            except (serial.SerialException, OSError):
                pass
        self.ser.close()

    def rtt_stats(self):
        if not self.rtts:
            return {}
        a = np.array(self.rtts) * 1e3
        return {"rtt_med_ms": float(np.median(a)), "rtt_p95_ms": float(np.percentile(a, 95)),
                "rejected": self.rejected}
