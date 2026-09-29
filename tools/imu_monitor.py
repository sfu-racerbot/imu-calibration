#!/usr/bin/env python3
"""Live text view of the IMU streams. First thing to run at the lab: are both talking, at what rate,
and do the axes make sense? (Tilt nose-down: +x reads ~+g? Roll left: ...)

Values are averaged over each 0.2 s screen refresh. If calib/<imu>_calib.json exists, the calibrated
values are shown under the raw ones: at rest, |a| - g should be much closer to 0 and gyro close to 0.

    python tools/imu_monitor.py                       # ports from config/car.yaml
    python tools/imu_monitor.py --vesc /dev/ttyACM0   # only the VESC
    python tools/imu_monitor.py --sim drift --log     # simulator, and record a run
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from racer_imu.calibration import ImuCalibration  # noqa: E402
from racer_imu.cli import add_common_args, setup  # noqa: E402
from racer_imu.frames import R_to_rpy, quat_to_R  # noqa: E402
from racer_imu.logger import RunLogger  # noqa: E402
from racer_imu.types import G, ImuSample, WheelSample  # noqa: E402


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    ap.add_argument("--log", action="store_true", help="record raw samples to logs/")
    a = ap.parse_args()
    cfg, sources, sim = setup(a)
    if not sources:
        sys.exit("no IMU sources configured")
    cals = {}
    for s in sources:
        path = cfg.resolve(cfg.imus[s.src_name].extra.get("calib")) if s.src_name in cfg.imus else None
        if path and path.exists():
            cals[s.src_name] = (ImuCalibration.load(path), path)
    log = RunLogger("monitor", meta={"ports": {s.src_name: getattr(s, "port", "") for s in sources}}) if a.log else None
    for s in sources:
        s.start()
    latest: dict[str, ImuSample] = {}
    window: dict[str, list[ImuSample]] = {}
    wheel: WheelSample | None = None
    try:
        while True:
            time.sleep(0.2)
            for s in sources:
                batch = []
                for it in s.drain():
                    if log:
                        log.log(it)
                    if isinstance(it, WheelSample):
                        wheel = it
                    else:
                        latest[it.src] = it
                        batch.append(it)
                if batch:
                    window[s.src_name] = batch
            out = ["\x1b[2J\x1b[H" + "IMU monitor  (Ctrl+C to quit)" + ("   [SIM]" if sim else "")
                   + (f"   logging -> {log.dir}" if log else ""), ""]
            for s in sources:
                x = latest.get(s.src_name)
                out.append(f"[{s.src_name}] {getattr(s, 'port', '')}  rate {s.rate_hz:6.1f} Hz   errors {s.errors}"
                           + (f"   last: {s.last_error}" if s.last_error else ""))
                if hasattr(s, "rtt_stats") and s.rtt_stats():
                    r = s.rtt_stats()
                    out.append(f"    round trip  median {r['rtt_med_ms']:.2f} ms  p95 {r['rtt_p95_ms']:.2f} ms"
                               f"  rejected {r['rejected']}")
                if x is None:
                    out.append("    (no data yet)")
                    continue
                rpy = np.degrees(R_to_rpy(quat_to_R(x.quat))) if np.all(np.isfinite(x.quat)) else [np.nan] * 3
                w = window.get(s.src_name, [x])
                acc = np.mean([v.acc for v in w], axis=0)
                gyr = np.mean([v.gyro for v in w], axis=0)
                rows = [("raw", acc, gyr)]
                if s.src_name in cals:
                    c = cals[s.src_name][0]
                    rows.append(("cal", c.A_inv @ (acc - c.acc_bias), gyr - c.gyro_bias))
                for tag, av, gv in rows:
                    n = np.linalg.norm(av)
                    out.append("    %s acc  [m/s^2]  x %+8.3f  y %+8.3f  z %+8.3f   |a| %6.3f  |a|-g %+7.3f"
                               % (tag, *av, n, n - G))
                for tag, av, gv in rows:
                    out.append("    %s gyro [deg/s]  x %+8.2f  y %+8.2f  z %+8.2f" % (tag, *np.degrees(gv)))
                if s.src_name in cals:
                    out.append(f"    calibration: {cals[s.src_name][1]}")
                else:
                    out.append("    calibration: none yet (tools/calibrate_imu.py)")
                out.append("    rpy  [deg]    r %+8.2f  p %+8.2f  y %+8.2f   (sensor's own filter)" % tuple(rpy))
                out.append("")
            if wheel is not None:
                v = cfg.car.erpm_to_speed(wheel.erpm)
                out.append(f"[wheel] erpm {wheel.erpm:8.0f}   speed {v:+6.2f} m/s   battery {wheel.v_in:5.2f} V")
            print("\n".join(out), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        for s in sources:
            s.stop()
        if log:
            log.close()
            print(f"saved {log.dir}")


if __name__ == "__main__":
    main()
