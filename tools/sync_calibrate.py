#!/usr/bin/env python3
"""Measure the time offset (and relative mounting rotation) between the two IMUs.

Both IMUs sit on the same rigid car, so they feel exactly the same rotation rate. Pick the car up and
shake/twist it about all axes for ~15 s. The rotation-rate magnitude |gyro| does not depend on how each
IMU is mounted, so cross-correlating it gives the time offset; aligning the gyro vectors afterwards gives
the relative rotation, which is a check on the rpy_deg in car.yaml.

Result -> calib/sync.json (offsets are ADDED to that stream's timestamps).

    python tools/sync_calibrate.py
    python tools/sync_calibrate.py --sim           # simulator 'shake' scenario, knows the true answer
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from racer_imu.calibration import ImuCalibration, kabsch, stamp  # noqa: E402
from racer_imu.cli import add_common_args, setup  # noqa: E402
from racer_imu.clock_sync import estimate_time_offset  # noqa: E402
from racer_imu.frames import R_to_rpy  # noqa: E402
from racer_imu.logger import RunLogger  # noqa: E402
from racer_imu.sim import expected_sync_offset  # noqa: E402
from racer_imu.sync import resample  # noqa: E402
from racer_imu.types import ImuSample  # noqa: E402


def rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter),
                         sim_default="shake")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--max-lag", type=float, default=0.15, help="search window [s]")
    a = ap.parse_args()
    cfg, sources, sim = setup(a)
    ref = cfg.section("sync").get("reference", "vesc")
    names = [s.src_name for s in sources]
    if ref not in names or len(names) < 2:
        sys.exit(f"need the reference '{ref}' and one more IMU; have {names}")
    other = next(n for n in names if n != ref)
    cal = {n: ImuCalibration.load_or_identity(cfg.resolve(cfg.imus[n].extra.get("calib"))) for n in names}
    log = RunLogger("sync", meta={"reference": ref, "other": other, "sim": bool(a.sim)})

    print(f"Shake and twist the car about all axes for {a.seconds:.0f} s (starting now) ...")
    for s in sources:
        s.start()
    data = {n: [] for n in names}
    t_end = time.monotonic() + a.seconds + 1.0
    while time.monotonic() < t_end:
        time.sleep(0.05)
        for s in sources:
            for it in s.drain():
                log.log(it)
                if isinstance(it, ImuSample):
                    data[it.src].append(cal[it.src].apply(it))
    for s in sources:
        s.stop()
    log.close()
    for n in names:
        print(f"  {n}: {len(data[n])} samples")
        if len(data[n]) < 100:
            sys.exit(f"not enough data from {n}")

    ta = np.array([x.t for x in data[ref]])
    tb = np.array([x.t for x in data[other]])
    wa = np.array([np.linalg.norm(x.gyro) for x in data[ref]])
    wb = np.array([np.linalg.norm(x.gyro) for x in data[other]])
    if wa.std() < 0.3:
        print("  WARNING: very little rotation; shake harder for a reliable result")
    dt, corr = estimate_time_offset(ta, wa, tb, wb, max_lag=a.max_lag)

    # relative rotation: omega_ref = R_rel @ omega_other, on a common grid after the time shift
    shifted = [ImuSample(x.t + dt, x.acc, x.gyro, x.quat, x.src) for x in data[other]]
    grid = np.arange(max(ta[0], tb[0] + dt) + 0.1, min(ta[-1], tb[-1] + dt) - 0.1, 0.005)
    ra, rb = resample(data[ref], grid), resample(shifted, grid)
    pairs = [(p.gyro, q.gyro) for p, q in zip(ra, rb) if p is not None and q is not None
             and np.linalg.norm(p.gyro) > 0.3]
    R_rel = kabsch([q for _, q in pairs], [p for p, _ in pairs])
    R_cfg = cfg.imus[ref].R.T @ cfg.imus[other].R
    mount_err = rot_angle_deg(R_cfg.T @ R_rel)
    R_other_suggest = cfg.imus[ref].R @ R_rel

    print(f"\n  time offset for '{other}': {dt * 1e3:+.2f} ms   (correlation peak {corr:.3f})")
    if sim:
        print(f"  simulator truth:           {expected_sync_offset(sim.truth_sensors, ref, other) * 1e3:+.2f} ms")
    print(f"  relative rotation vs car.yaml: {mount_err:.2f} deg off")
    if mount_err > 3:
        r = np.degrees(R_to_rpy(R_other_suggest))
        print(f"  -> if '{ref}' rpy_deg is right, '{other}' should be about [{r[0]:.1f}, {r[1]:.1f}, {r[2]:.1f}]")

    path = cfg.resolve(cfg.section("sync").get("file", "calib/sync.json"))
    old = json.loads(path.read_text()) if path.exists() else {}
    offsets = {**old.get("offsets", {}), other: float(dt), ref: 0.0}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "offsets": offsets, "reference": ref, "peak_corr": corr, "created": stamp(),
        "R_rel_other_to_ref": R_rel.tolist(), "mount_error_deg": mount_err, "log": str(log.dir),
        "note": "offsets are added to each stream's host timestamps"}, indent=2))
    print(f"\nsaved {path}\nraw log {log.dir}")


if __name__ == "__main__":
    main()
