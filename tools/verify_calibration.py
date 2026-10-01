#!/usr/bin/env python3
"""Verify an IMU calibration with fresh measurements it never saw. Each test prints PASS / OK / FAIL.

    python tools/verify_calibration.py still --vesc /dev/ttyACM0            # 60 s parked: gyro drift
    python tools/verify_calibration.py poses --vesc /dev/ttyACM0            # 6 NEW orientations: |a| = g?
    python tools/verify_calibration.py flip  --vesc /dev/ttyACM0            # 180 deg reversal: x/y bias
    python tools/verify_calibration.py turn  --vesc /dev/ttyACM0 --turns 2  # full turns: gyro scale
    python tools/verify_calibration.py all   --vesc /dev/ttyACM0            # still, poses, flip, turn
    add --sim to try any of them against the simulator

still  Leave the car untouched on the floor.
poses  Rest it in orientations DIFFERENT from the calibration ones (diagonals, odd angles).
flip   Car on a table or floor (level not needed): keep still, turn it 180 deg on the same spot, keep still.
       Turn it SLOWLY by sliding; rocking or the car settling differently spoils the result (it warns).
turn   Line the car up with a tape line or wall edge, keep still, turn it by hand full turns in one
       direction back onto the same line, keep still. More turns = more precise.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from racer_imu.calibration import ImuCalibration  # noqa: E402
from racer_imu.cli import add_common_args, setup, use_sim_paths  # noqa: E402
from racer_imu.frames import load_config  # noqa: E402
from racer_imu.logger import RunLogger  # noqa: E402
from racer_imu.types import ImuSample  # noqa: E402
from racer_imu.verify import PoseCheck, TurnCheck, check_still, flip_result, turn_result  # noqa: E402

SIM = {"still": "still", "poses": "poses", "flip": "flip", "turn": "turn"}


def stream(args, imu, mode, handler, seconds, log):
    """Feed raw samples to handler(raw) until it returns True or `seconds` pass."""
    cfg, sources, sim = setup(args, [imu], SIM[mode] if args.sim else None)
    if not sources:
        sys.exit(f"no port for '{imu}'")
    src = sources[0]
    src.start()
    t_end = time.monotonic() + seconds
    try:
        while time.monotonic() < t_end:
            time.sleep(0.02)
            for it in src.drain():
                if isinstance(it, ImuSample):
                    it.label = mode
                    log.log(it)
                    if handler(it):
                        return True
            if src.count == 0 and src.last_error and time.monotonic() > t_end - seconds + 3:
                sys.exit(f"no data: {src.last_error}")
    finally:
        src.stop()
        src.join(1.0)
        if sim:
            sim.stop()
    return False


def line(label, value, grade=""):
    print(f"  {label:34s} {value:>30s}   {grade}")


def run_still(a, cal, log):
    print(f"\n[still] don't touch the car for {a.seconds:.0f} s ...")
    got = []
    stream(a, a.imu, "still", lambda r: got.append(cal.apply(r)) and False, a.seconds, log)
    r = check_still(got)
    if r["max_rate_deg_s"] > 5:
        print("  WARNING: it moved during the test; repeat")
    line("gyro at rest (cal) [deg/s]", np.array2string(r["gyro_mean_deg_s"], precision=3))
    line("heading drift", f"{r['heading_drift_deg_min']:+.2f} deg/min", r["grade_gyro"])
    line("|a| - g at rest (cal)", f"{r['acc_err']:+.4f} m/s^2", r["grade_acc"])
    line("gyro noise [deg/s]", np.array2string(r["gyro_noise_deg_s"], precision=3))
    return {"still": r}


def run_poses(a, cal, log):
    print(f"\n[poses] rest the car in {a.poses} NEW orientations (not the ones used to calibrate) ...")
    pc = PoseCheck(cal, cal.meta.get("accel", {}).get("pose_means"))
    last = {"t": 0.0}

    def handler(raw):
        row = pc.add(raw)
        col = pc.pc
        if row:
            print(f"\r\x1b[2K  >>> pose {len(pc.rows)}/{a.poses} CAPTURED: |a|-g raw {row['raw_err']:+.4f}  "
                  f"cal {row['cal_err']:+.4f}  ({row['deg_from_calib_pose']:.0f} deg from nearest calibration pose)"
                  + ("   -> move to the next pose" if len(pc.rows) < a.poses else ""), flush=True)
        elif raw.t - last["t"] > 0.25:
            last["t"] = raw.t
            what = {"moving": "moving - rest it somewhere",
                    "holding": f"still... hold {col.held_s:.1f}/{col.hold_s:.1f} s",
                    "settling": "settling (vibration?)",
                    "same": f"same as pose {col.nearest_idx + 1} ({col.nearest_deg:.0f} deg): tilt more",
                    "captured": "captured"}[col.state]
            print(f"\r\x1b[2K  [{len(pc.rows)}/{a.poses}] {what}", end="", flush=True)
        return len(pc.rows) >= a.poses

    stream(a, a.imu, "poses", handler, a.timeout, log)
    print()
    if not pc.rows:
        print("  no poses captured")
        return {}
    r = pc.result()
    line("|a| - g RMS raw -> cal", f"{r['raw_rms']:.4f} -> {r['cal_rms']:.4f} m/s^2", r["grade"])
    line("worst pose (cal)", f"{r['cal_max']:.4f} m/s^2")
    if any(row["deg_from_calib_pose"] < 15 for row in pc.rows):
        print("  note: some poses are close to calibration poses; diagonals test the fit harder")
    return {"poses": r}


def run_turn(a, cal, log, mode):
    turns = 0.5 if mode == "flip" else a.turns
    what = "turn it 180 deg on the same spot" if mode == "flip" else \
        f"turn it by hand {a.turns:g} full turn(s) in one direction, back onto the same line"
    print(f"\n[{mode}] keep still ~2 s, then {what}, then keep still ~2 s ...")
    target = 180.0 if mode == "flip" else 360.0 * turns
    tc = TurnCheck(min_turn_deg=90 if mode == "flip" else 300 * turns)
    last = {"phase": None, "t": 0.0}

    def handler(raw):
        ph = tc.add(cal.apply(raw), raw)
        if ph == last["phase"] and raw.t - last["t"] > 0.25:      # live status line
            last["t"] = raw.t
            held = (tc.win.buf[-1].t - tc.win.buf[0].t) if tc.win.buf else 0.0
            if ph == "A":
                st = "moving - set it down and let go" if tc.win.moving else f"hold still {held:.1f}/{tc.win.hold_s:.0f} s"
            elif ph == "move":
                st = f"turned {abs(np.degrees(tc.yaw)):5.0f} of {target:.0f} deg (one direction only)"
            else:
                st = (f"turned {abs(np.degrees(tc.yaw)):5.0f} deg - line it up, let go: "
                      + ("moving" if tc.win.moving else f"hold still {held:.1f}/{tc.win.hold_s:.0f} s"))
            print("\r\x1b[2K  " + st, end="", flush=True)
        if ph != last["phase"]:
            print("\r\x1b[2K", end="")
            last["phase"] = ph
            msg = {"move": f"  got the start position -> now {what}",
                   "B": "  turning ... keep going, then set it down and keep still",
                   "done": "  got the end position"}.get(ph)
            if msg:
                print(msg, flush=True)
        return ph == "done"

    if not stream(a, a.imu, mode, handler, a.timeout, log):
        print(f"  timed out in phase '{tc.phase}'")
        return {}
    if mode == "flip":
        r = flip_result(tc, cal)
        line("turned", f"{r['turned_deg']:+.1f} deg")
        line("table tilt (for info)", f"{r['table_tilt_deg']:.2f} deg")
        line("level error, calibrated", np.array2string(r["residual_bias_ms2"], precision=3)
             + f" m/s^2 = {r['level_error_deg']:.2f} deg", r["grade"])
        if "calib_bias_ms2" in r:
            line("  info: bias seen in raw data", np.array2string(r["raw_residual_ms2"], precision=3) + " m/s^2")
            line("  info: calibration's bias", np.array2string(r["calib_bias_ms2"], precision=3) + " m/s^2")
        if "warning" in r:
            print("  WARNING:", r["warning"])
        return {"flip": r}
    r = turn_result(tc, turns)
    line("measured / expected", f"{r['measured_deg']:.1f} / {r['expected_deg']:.0f} deg")
    line("gyro scale error", f"{r['scale_error_pct']:+.2f} %", r["grade"])
    if r["grade"] != "PASS":
        print("  (the calibration corrects gyro bias only, not scale. A consistent error here across repeats\n"
              "   means the gyro needs a scale factor; a random one means the car wasn't realigned exactly)")
    return {"turn": r}


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    ap.add_argument("test", choices=["still", "poses", "flip", "turn", "all"])
    ap.add_argument("--imu", default="vesc")
    ap.add_argument("--calib", default=None)
    ap.add_argument("--seconds", type=float, default=60.0, help="still: how long")
    ap.add_argument("--poses", type=int, default=6)
    ap.add_argument("--turns", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=180.0)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.sim:
        use_sim_paths(cfg)
    path = Path(a.calib) if a.calib else cfg.resolve(cfg.imus[a.imu].extra.get("calib"))
    if not path.exists():
        sys.exit(f"no calibration at {path}")
    cal = ImuCalibration.load(path)
    print(f"verifying {path}")
    log = RunLogger(f"verify_{a.imu}", meta={"calib": str(path), "test": a.test, "sim": bool(a.sim), "turns": a.turns,
                                            "poses": a.poses})
    tests = ["still", "poses", "flip", "turn"] if a.test == "all" else [a.test]
    results = {}
    for t in tests:
        if t == "still":
            results.update(run_still(a, cal, log))
        elif t == "poses":
            results.update(run_poses(a, cal, log))
        else:
            results.update(run_turn(a, cal, log, t))
    log.close()
    grades = [v.get("grade") or v.get("grade_gyro") for v in results.values() if isinstance(v, dict)]
    grades += [results["still"]["grade_acc"]] if "still" in results else []
    overall = "FAIL" if "FAIL" in grades else ("OK" if "OK" in grades else "PASS")
    print(f"\noverall: {overall}   (PASS = as good as expected, OK = usable, FAIL = recalibrate)\nlog: {log.dir}")


if __name__ == "__main__":
    main()
