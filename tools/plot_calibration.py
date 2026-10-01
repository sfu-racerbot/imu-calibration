#!/usr/bin/env python3
"""Before/after plots for an accelerometer calibration (idea from michaelwro/accelerometer-calibration).

A good calibration puts every static pose on the sphere of radius g and makes |a| - g small.

    python tools/plot_calibration.py calib/vesc_calib.json
    python tools/plot_calibration.py calib/vesc_calib.json --save vesc_calib.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from racer_imu.calibration import ImuCalibration  # noqa: E402
from racer_imu.logger import load_imu_csv  # noqa: E402
from racer_imu.types import G  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("calib")
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    import matplotlib
    if a.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cal = ImuCalibration.load(a.calib)
    acc_meta = cal.meta.get("accel")
    if not acc_meta:
        sys.exit("this file has no accel calibration yet")
    poses = np.array(acc_meta["pose_means"])
    raw = poses
    csv = Path(acc_meta.get("log", "")) / f"{cal.meta.get('imu', '')}_imu.csv"
    if csv.is_file():   # use every sample of every pose, not only the means
        pts = [s.acc for s in load_imu_csv(csv) if s.label.startswith("pose_")]
        if pts:
            raw = np.array(pts)
    b_cap = cal.acc_bias - np.array(acc_meta.get("device_offsets_accel_g_at_capture", [0, 0, 0])) * G
    cal_pts = (raw - b_cap) @ cal.A_inv.T

    RAW, CAL = "#d95f02", "#1b9e77"
    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.3)

    # 3D: both point clouds on the same axes
    ax3 = fig.add_subplot(gs[0, 0:2], projection="3d")
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:15j]
    ax3.plot_wireframe(G * np.cos(u) * np.sin(v), G * np.sin(u) * np.sin(v), G * np.cos(v), color="#ccc", lw=0.5)
    ax3.scatter(*raw.T, s=3, c=RAW, alpha=0.6, label="raw", depthshade=False)
    ax3.scatter(*cal_pts.T, s=3, c=CAL, alpha=0.6, label="calibrated", depthshade=False)
    ax3.set_title("static poses vs sphere of radius g")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.legend(loc="upper left")
    ax3.set_box_aspect((1, 1, 1))

    # 2D side views: raw and calibrated overlaid on a circle of radius g
    th = np.linspace(0, 2 * np.pi, 200)
    for k, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)]):
        ax = fig.add_subplot(gs[1, k])
        ax.plot(G * np.cos(th), G * np.sin(th), color="#999", lw=0.8, label="radius g")
        ax.scatter(raw[:, i], raw[:, j], s=4, c=RAW, alpha=0.5, label="raw")
        ax.scatter(cal_pts[:, i], cal_pts[:, j], s=4, c=CAL, alpha=0.5, label="calibrated")
        ax.set_xlabel(f"{'xyz'[i]} [m/s²]")
        ax.set_ylabel(f"{'xyz'[j]} [m/s²]")
        ax.set_title(f"{'xyz'[i]}-{'xyz'[j]} view", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8, loc="center")

    # magnitude error histogram, overlaid
    ax = fig.add_subplot(gs[0, 2])
    e0, e1 = np.linalg.norm(raw, axis=1) - G, np.linalg.norm(cal_pts, axis=1) - G
    lo, hi = np.percentile(np.r_[e0, e1], [0.5, 99.5])
    bins = np.linspace(lo, hi, 60)
    ax.hist(e0, bins, alpha=0.55, color=RAW, label=f"raw   rms {np.sqrt(np.mean(e0**2)):.4f}")
    ax.hist(e1, bins, alpha=0.55, color=CAL, label=f"cal   rms {np.sqrt(np.mean(e1**2)):.4f}")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("|a| − g  [m/s²]")
    ax.set_title("magnitude error, all pose samples", fontsize=10)
    ax.legend(fontsize=8)

    # per-pose error, overlaid on one axis
    ax = fig.add_subplot(gs[0, 3])
    pm = np.linalg.norm(poses, axis=1) - G
    pc = np.linalg.norm((poses - b_cap) @ cal.A_inv.T, axis=1) - G
    idx = np.arange(1, len(poses) + 1)
    ax.plot(idx, pm, "o-", color=RAW, label="raw")
    ax.plot(idx, pc, "o-", color=CAL, label="calibrated")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(idx)
    ax.set_xlabel("pose #")
    ax.set_ylabel("|mean a| − g  [m/s²]")
    ax.set_title(f"per pose ({acc_meta['model']} model)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # the numbers
    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    gb = np.degrees(cal.gyro_bias)
    ax.text(0, 1, "accel bias [m/s²]\n  " + np.array2string(cal.acc_bias, precision=4)
            + "\n\nA_inv\n" + np.array2string(cal.A_inv, precision=4, prefix="  ")
            + f"\n\ngyro bias [deg/s]\n  {np.array2string(gb, precision=3)}"
            + f"\n\n|a|−g rms (pose means)\n  raw {np.sqrt(np.mean(pm**2)):.4f}  cal {np.sqrt(np.mean(pc**2)):.4f}",
            va="top", family="monospace", fontsize=9, transform=ax.transAxes)
    fig.suptitle(f"{a.calib}   (raw = orange, calibrated = green)")
    if a.save:
        fig.savefig(a.save, dpi=110)
        print(f"saved {a.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
