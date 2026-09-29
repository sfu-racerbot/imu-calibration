"""Read the BNO085 stream produced by firmware/bno085_stream on a microcontroller.

Line format (ASCII, one sample per line, NMEA-style XOR checksum over the text between $ and *):
    $B,<t_us>,<ax>,<ay>,<az>,<gx>,<gy>,<gz>,<qw>,<qx>,<qy>,<qz>,<acc_status>*HH\r\n
    t_us   : sensor timestamp, uint32 microseconds (wraps every ~71.6 min)
    a*     : SH2_ACCELEROMETER, m/s^2 (includes gravity)
    g*     : SH2_GYROSCOPE_CALIBRATED, rad/s
    q*     : SH2_GAME_ROTATION_VECTOR (no magnetometer), w x y z
Lines starting with '#' are comments from the MCU and are ignored.

The device timestamp is mapped onto the host clock with ClockMapper, so USB jitter doesn't show up in
sample times.
"""
from __future__ import annotations

import time

import numpy as np
import serial

from .clock_sync import ClockMapper, Unwrapper
from .source import Source
from .types import ImuSample


def checksum(body: str) -> int:
    c = 0
    for ch in body.encode():
        c ^= ch
    return c


def format_line(t_us: int, acc, gyro, quat, status: int = 3) -> str:
    body = "B,%d,%.4f,%.4f,%.4f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%d" % (
        t_us & 0xFFFFFFFF, *acc, *gyro, *quat, status)
    return "$%s*%02X\r\n" % (body, checksum(body))


def parse_line(line: str) -> dict | None:
    line = line.strip()
    if not line.startswith("$B,"):
        return None
    star = line.rfind("*")
    if star < 0:
        raise ValueError("missing checksum")
    body = line[1:star]
    if int(line[star + 1:star + 3], 16) != checksum(body):
        raise ValueError("bad checksum")
    f = body.split(",")
    if len(f) != 13:
        raise ValueError(f"expected 13 fields, got {len(f)}")
    v = [float(x) for x in f[2:12]]
    return {"t_us": int(f[1]), "acc": np.array(v[0:3]), "gyro": np.array(v[3:6]),
            "quat": np.array(v[6:10]), "status": int(f[12])}


class LineDecoder:
    """Turn (t_recv, line) into ImuSamples with host-clock timestamps."""

    def __init__(self, name: str = "bno"):
        self.name = name
        self.unwrap = Unwrapper(32)
        self.clock = ClockMapper()
        self.status = -1

    def decode(self, line: str, t_recv: float) -> ImuSample | None:
        d = parse_line(line)
        if d is None:
            return None
        if not (np.all(np.isfinite(d["acc"])) and np.all(np.isfinite(d["gyro"]))):
            return None           # first lines after boot: gyro report not received yet
        t_dev = self.unwrap(d["t_us"]) * 1e-6
        t = self.clock.add(t_dev, t_recv)
        self.status = d["status"]
        return ImuSample(t, d["acc"], d["gyro"], d["quat"], self.name, t_dev=t_dev)


class BnoDriver(Source):
    def __init__(self, port: str, name: str = "bno", baud: int = 921600):
        super().__init__(name)
        self.port = port
        self.baud = baud
        self.dec = LineDecoder(name)
        self.comments: list[str] = []

    def run(self):
        try:
            ser = serial.Serial(self.port, self.baud, timeout=0.05)
        except serial.SerialException as e:
            self.last_error = str(e)
            self.errors += 1
            return
        buf = b""
        while not self.stopped:
            try:
                data = ser.read(max(1, ser.in_waiting))
            except (serial.SerialException, OSError) as e:
                self.errors += 1
                self.last_error = str(e)
                time.sleep(0.5)
                continue
            if not data:
                continue
            t_recv = time.monotonic()
            buf += data
            *lines, buf = buf.split(b"\n")
            for raw in lines:
                line = raw.decode(errors="replace")
                if line.startswith("#"):
                    self.comments.append(line.strip())
                    continue
                try:
                    s = self.dec.decode(line, t_recv)
                except ValueError as e:
                    self.errors += 1
                    self.last_error = str(e)
                    continue
                if s is not None:
                    self.emit(s)
                    self.tick_rate()
        ser.close()
