#!/usr/bin/env python3
"""Put the calibration's bias into the VESC itself (App Settings -> IMU offsets), without double-correcting.

What the VESC firmware can store: accel offsets [G] and gyro offsets [deg/s], subtracted after its own
IMU rotation and before its orientation filter. It has NO field for scale / cross-axis (A_inv), so
that part always stays in racer_imu. Writing the offsets helps anything that reads the VESC's own data:
its orientation filter (less yaw drift), VESC Tool's IMU page, the F1TENTH ROS vesc driver.

    python tools/vesc_offsets.py              1. show the values to type into VESC Tool
    python tools/vesc_offsets.py --written    2. after "Write App Configuration": record them here
    python tools/vesc_offsets.py --cleared    after setting the VESC offsets back to 0

Recording matters: once the VESC subtracts the offsets, racer_imu must subtract only what is left.
Leave the VESC's IMU rotation (Imu Rotation Roll/Pitch/Yaw) unchanged: the calibration was measured
with the current rotation, and the car mounting is handled by rpy_deg in config/car.yaml.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from racer_imu.calibration import ImuCalibration  # noqa: E402
from racer_imu.frames import load_config  # noqa: E402
from racer_imu.types import G  # noqa: E402


def vesc_values(cal: ImuCalibration):
    """Offsets rounded the way VESC Tool stores them (3 decimals)."""
    return np.round(cal.acc_bias / G, 3), np.round(np.degrees(cal.gyro_bias), 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--imu", default="vesc")
    ap.add_argument("--calib", default=None, help="calibration file (default: from car.yaml)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--written", action="store_true", help="record that the shown offsets are now in the VESC")
    g.add_argument("--cleared", action="store_true", help="record that the VESC offsets are back to 0")
    a = ap.parse_args()

    cfg = load_config(a.config)
    path = Path(a.calib) if a.calib else cfg.resolve(cfg.imus[a.imu].extra.get("calib"))
    if not path or not path.exists():
        sys.exit(f"no calibration file at {path}: run tools/calibrate_imu.py first")
    cal = ImuCalibration.load(path)
    acc_g, gyro_dps = vesc_values(cal)
    dev_acc_g, dev_gyro = np.round(cal.dev_acc_offset / G, 3), np.round(np.degrees(cal.dev_gyro_offset), 3)

    if a.written:
        cal.dev_acc_offset = acc_g * G              # the rounded values really stored in the VESC
        cal.dev_gyro_offset = np.radians(gyro_dps)
        cal.save(path)
        print(f"recorded in {path}: the VESC now subtracts accel {acc_g} G, gyro {gyro_dps} deg/s")
        print(f"racer_imu will subtract only the remainder: accel {cal.eff_acc_bias().round(4)} m/s^2, "
              f"gyro {np.degrees(cal.eff_gyro_bias()).round(4)} deg/s (rounding), plus scale/cross-axis")
        return
    if a.cleared:
        cal.dev_acc_offset = np.zeros(3)
        cal.dev_gyro_offset = np.zeros(3)
        cal.save(path)
        print(f"recorded in {path}: VESC offsets are 0, racer_imu subtracts the full bias again")
        return

    print(f"calibration: {path}\n")
    print("VESC Tool -> App Settings -> IMU  (connect, change these six fields, then 'Write App Configuration')\n")
    print(f"  {'':8s} {'Accel Offset [G]':>18s} {'Gyro Offset [deg/s]':>22s}")
    for i, ax in enumerate("XYZ"):
        print(f"  {ax:8s} {acc_g[i]:>18.3f} {gyro_dps[i]:>22.3f}")
    print("\n  Leave 'Imu Rotation Roll/Pitch/Yaw' as they are.")
    if np.any(dev_acc_g) or np.any(dev_gyro):
        print(f"\n  Currently recorded as already in the VESC: accel {dev_acc_g} G, gyro {dev_gyro} deg/s")
        if np.allclose(dev_acc_g, acc_g) and np.allclose(dev_gyro, gyro_dps):
            print("  -> the VESC already has these values; nothing to do.")
    print("\nThen close VESC Tool and run:  python tools/vesc_offsets.py --written")
    print("\nNot writable to the VESC (stays in racer_imu): scale/cross-axis A_inv =")
    print(np.array2string(cal.A_inv, precision=4, prefix="  "))
    print("The gyro bias creeps with temperature: the VESC's copy is a fixed snapshot, while racer_imu keeps "
          "correcting the remainder online.")


if __name__ == "__main__":
    main()
