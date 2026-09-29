"""CSV run logs: logs/<YYYYmmdd_HHMMSS>_<name>/{meta.json, <imu>_imu.csv, wheel.csv, est.csv, ...}.

IMU logs are RAW (before calibration, before sync offsets) so any run can be re-calibrated and
re-synced offline. Each CSV starts with '# key: value' metadata lines.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from .types import ImuSample, WheelSample

ROOT = Path(__file__).resolve().parent.parent
IMU_COLS = ["t", "t_dev", "rtt", "ax", "ay", "az", "gx", "gy", "gz", "qw", "qx", "qy", "qz", "label"]
WHEEL_COLS = ["t", "erpm", "v_in", "tacho"]


class RunLogger:
    def __init__(self, name: str = "run", root: str | Path | None = None, meta: dict | None = None):
        root = Path(root) if root else ROOT / "logs"
        self.dir = root / f"{time.strftime('%Y%m%d_%H%M%S')}_{name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "clock": "host time.monotonic() [s]",
                     "units": "acc m/s^2 (raw, before calibration), gyro rad/s, quat w x y z",
                     **(meta or {})}
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str))
        self._files: dict[str, tuple] = {}

    def _writer(self, stream: str, cols: list[str]):
        if stream not in self._files:
            f = open(self.dir / f"{stream}.csv", "w", newline="")
            for k, v in self.meta.items():
                f.write(f"# {k}: {v}\n")
            w = csv.writer(f)
            w.writerow(cols)
            self._files[stream] = (f, w, cols)
        return self._files[stream]

    def log(self, item, label: str | None = None):
        if isinstance(item, ImuSample):
            _, w, _ = self._writer(f"{item.src}_imu", IMU_COLS)
            w.writerow(["%.6f" % item.t, "%.6f" % item.t_dev, "%.6f" % item.rtt,
                        *("%.5f" % v for v in item.acc), *("%.6f" % v for v in item.gyro),
                        *("%.6f" % v for v in item.quat), label if label is not None else item.label])
        elif isinstance(item, WheelSample):
            _, w, _ = self._writer("wheel", WHEEL_COLS)
            w.writerow(["%.6f" % item.t, "%.1f" % item.erpm, "%.2f" % item.v_in, "%.0f" % item.tacho])

    def log_row(self, stream: str, row: dict):
        _, w, cols = self._writer(stream, list(row))
        w.writerow([("%.6g" % row[c]) if isinstance(row.get(c), float) else row.get(c, "") for c in cols])

    def flush(self):
        for f, _, _ in self._files.values():
            f.flush()

    def close(self):
        for f, _, _ in self._files.values():
            f.close()
        self._files.clear()


def _read_csv(path: Path):
    with open(path) as f:
        rows = [r for r in csv.reader(line for line in f if not line.startswith("#"))]
    return rows[0], rows[1:]


def load_imu_csv(path, src: str | None = None) -> list[ImuSample]:
    path = Path(path)
    src = src or path.stem.replace("_imu", "")
    hdr, rows = _read_csv(path)
    ix = {c: i for i, c in enumerate(hdr)}
    out = []
    for r in rows:
        f = lambda c: float(r[ix[c]])  # noqa: E731
        out.append(ImuSample(f("t"), np.array([f("ax"), f("ay"), f("az")]),
                             np.array([f("gx"), f("gy"), f("gz")]),
                             np.array([f("qw"), f("qx"), f("qy"), f("qz")]), src,
                             f("t_dev"), f("rtt"), r[ix["label"]] if "label" in ix else ""))
    return out


def load_wheel_csv(path) -> list[WheelSample]:
    hdr, rows = _read_csv(Path(path))
    ix = {c: i for i, c in enumerate(hdr)}
    return [WheelSample(float(r[ix["t"]]), float(r[ix["erpm"]]), float(r[ix["v_in"]]), float(r[ix["tacho"]]))
            for r in rows]


def load_run(run_dir) -> dict:
    """{'vesc': [ImuSample...], 'bno': [...], 'wheel': [WheelSample...]} from a run directory."""
    run_dir = Path(run_dir)
    out = {}
    for p in sorted(run_dir.glob("*_imu.csv")):
        out[p.stem.replace("_imu", "")] = load_imu_csv(p)
    if (run_dir / "wheel.csv").exists():
        out["wheel"] = load_wheel_csv(run_dir / "wheel.csv")
    return out


def latest_run(root: str | Path | None = None, contains: str = "") -> Path | None:
    root = Path(root) if root else ROOT / "logs"
    runs = sorted(p for p in root.glob(f"*{contains}*") if p.is_dir())
    return runs[-1] if runs else None
