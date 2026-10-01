#!/usr/bin/env python3
"""Guided IMU calibration. Writes calib/<imu>_calib.json and a raw log under logs/.

Steps (default: gyro then accel):
  gyro   keep the car perfectly still for --gyro-seconds  -> gyro bias
  accel  turn the IMU (or the whole car) into --poses different orientations and hold each ~2 s.
         Poses are captured automatically when it is still and pointing somewhere new. Use the 6 faces
         (+x up, -x up, +y up, ...) and then tilted in-between orientations. 12+ poses = full
         scale + cross-axis fit, 6-11 = bias + per-axis scale only.
  mount  car assembled, on level ground: hold still ~3 s, then push it straight forward
         -> prints the rpy_deg to put in config/car.yaml for this IMU

Before calibrating the VESC IMU: in VESC Tool set App > IMU accel/gyro offsets to 0 (the values the VESC
sends already include them) and note the rot_roll/pitch/yaw you use. Then close VESC Tool.

    python tools/calibrate_imu.py --imu vesc
    python tools/calibrate_imu.py --imu bno --step mount
    python tools/calibrate_imu.py --imu vesc --sim          # full run against the simulator
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from racer_imu.calibration import (ImuCalibration, PoseCollector, detect_static_segments, fit_accel,  # noqa: E402
                                   mount_rotation, rpy_deg, stamp)
from racer_imu.cli import add_common_args, setup, use_sim_paths  # noqa: E402
from racer_imu.frames import load_config  # noqa: E402
from racer_imu.logger import RunLogger  # noqa: E402
from racer_imu.types import G, ImuSample  # noqa: E402

SIM_SCENARIO = {"gyro": "still", "accel": "poses", "mount": "push"}


def collect(args, imu, step, seconds, on_sample=None, log=None, label=""):
    """Stream one IMU for up to `seconds` (on_sample returning True stops early). Returns samples."""
    cfg, sources, sim = setup(args, [imu], SIM_SCENARIO[step] if args.sim else None)
    if not sources:
        sys.exit(f"no port for '{imu}' (set it in car.yaml or pass --{imu})")
    src = sources[0]
    src.start()
    out = []
    t_end = time.monotonic() + seconds
    try:
        while time.monotonic() < t_end:
            time.sleep(0.02)
            done = False
            for it in src.drain():
                if not isinstance(it, ImuSample):
                    continue
                it.label = label
                out.append(it)
                if on_sample and on_sample(it):
                    done = True
                if log:
                    log.log(it, it.label)
            if done:
                break
            if src.last_error and src.count == 0 and time.monotonic() > t_end - seconds + 3:
                sys.exit(f"[{imu}] no data: {src.last_error}")
    finally:
        src.stop()
        src.join(1.0)
        if sim:
            sim.stop()
    print(f"  got {len(out)} samples at ~{src.rate_hz:.0f} Hz")
    return cfg, out


def step_gyro(args, cal, log):
    print(f"\n[gyro] keep it perfectly still for {args.gyro_seconds:.0f} s ...")
    _, s = collect(args, args.imu, "gyro", args.gyro_seconds, log=log, label="gyro_still")
    g = np.array([x.gyro for x in s])
    acc = np.array([x.acc for x in s])
    if np.max(np.std(g, axis=0)) > 0.02 or np.max(np.std(acc, axis=0)) > 0.3:
        print("  WARNING: it moved during the capture, bias may be off. Re-run this step.")
    cal.gyro_bias = g.mean(axis=0) + cal.dev_gyro_offset    # total bias = what's left + what the VESC removes
    cal.meta["gyro"] = {"n": len(s), "bias_deg_s": np.degrees(cal.gyro_bias).round(4).tolist(),
                        "noise_deg_s": np.degrees(g.std(axis=0)).round(4).tolist(), "time": stamp()}
    print(f"  gyro bias [deg/s]: {np.degrees(cal.gyro_bias).round(3)}   noise {np.degrees(g.std(axis=0)).round(3)}")


FACES = {"+X up": (0, 1), "-X up": (0, -1), "+Y up": (1, 1), "-Y up": (1, -1), "+Z up": (2, 1), "-Z up": (2, -1)}


def describe_direction(u):
    """'+Z up' if gravity is within 20 deg of a sensor axis, else 'tilted (x, y, z)'."""
    i = int(np.argmax(np.abs(u)))
    if abs(u[i]) > np.cos(np.radians(20)):
        return f"{'+' if u[i] > 0 else '-'}{'XYZ'[i]} up"
    return "tilted (%+.2f %+.2f %+.2f)" % tuple(u)


def faces_done(poses):
    done = set()
    for p in poses:
        u = p / np.linalg.norm(p)
        for name, (i, sign) in FACES.items():
            if sign * u[i] > np.cos(np.radians(30)):
                done.add(name)
    return done


def step_accel(args, cal, log):
    print(f"\n[accel] rest the IMU in {args.poses} different orientations, ~2 s each (auto-captured).")
    print("  Start with the 6 faces (each side facing up), then tilted positions in between.")
    print("  Live line: [captured/needed]  which sensor axis points up  what it is waiting for")
    print("             faces: * = done, . = still needed\n")
    pc = PoseCollector(hold_s=args.hold)
    state = {"k": 0, "last_print": 0.0}
    need_deg = np.degrees(np.arccos(pc.min_cos))

    def status_line(s):
        done = faces_done(pc.poses)
        faces = " ".join(f"{n.split()[0]}{'*' if n in done else '.'}" for n in FACES)
        what = {"moving": "moving: rest it",
                "holding": f"hold {pc.held_s:.1f}/{args.hold:.1f}s",
                "settling": "settling (vibration?)",
                "same": f"= pose {pc.nearest_idx + 1} ({pc.nearest_deg:.0f}<{need_deg:.0f}deg) tilt more",
                "captured": "captured"}[pc.state]
        return (f"[{state['k']}/{args.poses}] {describe_direction(pc.direction):<26} {what:<30}"
                f" faces {faces}")

    def on_sample(s):
        s = ImuSample(s.t, s.acc, s.gyro - cal.eff_gyro_bias(), s.quat, s.src)   # bias-free gyro for stillness
        m = pc.add(s)
        if m is not None:
            state["k"] += 1
            u = m / np.linalg.norm(m)
            missing = [n for n in FACES if n not in faces_done(pc.poses)]
            hint = f"next: try {missing[0]}" if missing else "all 6 faces done, now tilted positions"
            print(f"\r\x1b[2K  pose {state['k']:2d}/{args.poses} captured: {describe_direction(u):<26}"
                  f" -> {hint}", flush=True)
        elif s.t - state["last_print"] > 0.25:
            state["last_print"] = s.t
            print("\r\x1b[2K" + status_line(s), end="", flush=True)
        return state["k"] >= args.poses

    collect(args, args.imu, "accel", args.timeout, on_sample, log=None)
    print()
    if len(pc.poses) < 6:
        sys.exit(f"only {len(pc.poses)} poses captured; need at least 6")
    for i, samples in enumerate(pc.pose_samples):
        for s in samples:
            log.log(s, f"pose_{i}")
    A_inv, b, info = fit_accel(pc.poses)             # b = bias left in the data the VESC sends
    cal.A_inv, cal.acc_bias = A_inv, b + cal.dev_acc_offset
    cal.meta["accel"] = {**info, "pose_means": np.round(pc.poses, 5).tolist(), "time": stamp(), "log": str(log.dir),
                         "device_offsets_accel_g_at_capture": (cal.dev_acc_offset / G).tolist()}
    after = np.linalg.norm((np.array(pc.poses) - b) @ A_inv.T, axis=1)
    print(f"  model {info['model']} ({info['n_poses']} poses)")
    print(f"  bias [m/s^2]: {cal.acc_bias.round(4)}" + (f"   (of which the VESC removes {cal.dev_acc_offset.round(4)})"
                                                     if np.any(cal.dev_acc_offset) else ""))
    print(f"  A_inv:\n{np.array2string(A_inv, precision=5, prefix='    ')}")
    print(f"  |a| error rms: before {info['rms_before_ms2']:.4f}  after {info['rms_after_ms2']:.4f} m/s^2"
          f"   (max after {np.abs(after - G).max():.4f})")
    if info["rms_after_ms2"] > 0.05:
        print("  WARNING: large residual; some poses were probably not still. Re-run.")


def step_mount(args, cal, log):
    print("\n[mount] car on level ground: hold still ~3 s, then push it straight FORWARD ...")
    _, s = collect(args, args.imu, "mount", args.mount_seconds, log=log, label="mount")
    s = [cal.apply(x) for x in s]
    t = np.array([x.t for x in s])
    acc = np.array([x.acc for x in s])
    gyro = np.array([x.gyro for x in s])
    segs = detect_static_segments(t, acc, gyro, window_s=1.5)
    if not segs:
        sys.exit("never still for 1.5 s; try again")
    i0, i1 = segs[0]
    R = mount_rotation(acc[i0:i1].mean(axis=0), acc[i1:])
    rpy = rpy_deg(R)
    cal.meta["mount"] = {"rpy_deg": rpy.round(3).tolist(), "time": stamp()}
    print(f"  still for {t[i1 - 1] - t[i0]:.1f} s, then push detected")
    print(f"\n  -> in config/car.yaml, imus.{args.imu}.rpy_deg: [{rpy[0]:.2f}, {rpy[1]:.2f}, {rpy[2]:.2f}]")


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    ap.add_argument("--imu", required=True, help="IMU name from car.yaml (vesc, bno)")
    ap.add_argument("--step", default="imu", choices=["imu", "gyro", "accel", "mount", "all"],
                    help="imu = gyro+accel (default), all = gyro+accel+mount")
    ap.add_argument("--poses", type=int, default=12)
    ap.add_argument("--hold", type=float, default=1.5, help="seconds still before a pose is captured")
    ap.add_argument("--gyro-seconds", type=float, default=10.0)
    ap.add_argument("--mount-seconds", type=float, default=12.0)
    ap.add_argument("--timeout", type=float, default=300.0, help="max seconds for the accel step")
    ap.add_argument("--out", default=None, help="calibration file (default: from car.yaml)")
    a = ap.parse_args()
    steps = {"imu": ["gyro", "accel"], "all": ["gyro", "accel", "mount"]}.get(a.step, [a.step])

    cfg = load_config(a.config)
    if a.sim:
        use_sim_paths(cfg)
    if a.imu not in cfg.imus:
        sys.exit(f"unknown IMU '{a.imu}'; car.yaml has {list(cfg.imus)}")
    out = Path(a.out) if a.out else cfg.resolve(cfg.imus[a.imu].extra.get("calib", f"calib/{a.imu}_calib.json"))
    cal = ImuCalibration.load_or_identity(out)   # keep results of steps not re-run
    log = RunLogger(f"calib_{a.imu}", meta={"imu": a.imu, "steps": steps, "sim": bool(a.sim)})
    if a.imu.startswith("vesc"):
        dev = np.any(cal.dev_acc_offset) or np.any(cal.dev_gyro_offset)
        print("Reminder: close VESC Tool. VESC IMU offsets must be "
              + ("the ones recorded by tools/vesc_offsets.py --written." if dev else "0 (none recorded)."))
    for st in steps:
        {"gyro": step_gyro, "accel": step_accel, "mount": step_mount}[st](a, cal, log)
    cal.meta.update({"imu": a.imu, "updated": stamp(), "log": str(log.dir), "sim": bool(a.sim)})
    log.close()
    cal.save(out)
    print(f"\nsaved {out}\nraw log {log.dir}")
    if "accel" in steps:
        print(f"plot:  python tools/plot_calibration.py {out}")


if __name__ == "__main__":
    main()
