#!/usr/bin/env python3
"""Live before/after view of an IMU calibration.

Rotate the IMU (or the car) slowly through many orientations:
  left       every reading as a point in 3D. A perfect accelerometer at rest lies on the sphere of
             radius g. Raw points (orange) sit on a shifted, squashed ball; calibrated points (green)
             should hug the sphere.
  top right  |a| - g over time, raw vs calibrated (only meaningful while the IMU is still)
  mid right  gyro magnitude, raw vs bias-corrected (should sit near 0 while still)
  bottom     RMS numbers over the still moments seen so far

    python tools/calib_live.py --vesc /dev/ttyACM0
    python tools/calib_live.py --sim                     # simulator 'poses' scenario
    python tools/calib_live.py --vesc /dev/ttyACM0 --snapshot out.png --seconds 30
"""
import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from racer_imu.calibration import ImuCalibration  # noqa: E402
from racer_imu.cli import add_common_args, setup  # noqa: E402
from racer_imu.types import G, ImuSample  # noqa: E402

RAW, CAL = "#d95f02", "#1b9e77"
WINDOW_S = 20.0
STILL_GYRO = np.radians(3.0)     # below this rotation rate a sample counts as "still"


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter),
                         sim_default="poses")
    ap.add_argument("--imu", default=None, help="IMU name from car.yaml (default: the one you gave a port for, else vesc)")
    ap.add_argument("--calib", default=None, help="calibration file (default: from car.yaml)")
    ap.add_argument("--snapshot", default=None, help="run headless and save a PNG at the end")
    ap.add_argument("--seconds", type=float, default=30.0, help="with --snapshot: how long to run")
    a = ap.parse_args()
    imu = a.imu or (a.only[0] if a.only else ("bno" if a.bno and not a.vesc else "vesc"))

    import matplotlib
    if a.snapshot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    cfg, sources, sim = setup(a, only=[imu], processes=True)
    if not sources:
        sys.exit(f"no port for '{imu}' (pass --{imu} PORT or set it in config/car.yaml)")
    src = sources[0]
    path = Path(a.calib) if a.calib else cfg.resolve(cfg.imus[imu].extra.get("calib"))
    cal = ImuCalibration.load(path) if path and path.exists() else None
    src.start()

    pts_raw: deque = deque(maxlen=4000)
    pts_cal: deque = deque(maxlen=4000)
    hist: deque = deque()                 # (t, |a|-g raw, |a|-g cal, |w| raw, |w| cal)
    still_err = {"raw": [], "cal": []}
    state = {"n": 0}

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1], height_ratios=[1, 1, 0.45], hspace=0.4)
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    u, v = np.mgrid[0:2 * np.pi:32j, 0:np.pi:16j]
    ax3.plot_wireframe(G * np.cos(u) * np.sin(v), G * np.sin(u) * np.sin(v), G * np.cos(v), color="#ccc", lw=0.4)
    sc_raw = ax3.scatter([], [], [], s=4, c=RAW, label="raw", depthshade=False)
    sc_cal = ax3.scatter([], [], [], s=4, c=CAL, label="calibrated", depthshade=False)
    lim = G * 1.15
    ax3.set_xlim(-lim, lim)
    ax3.set_ylim(-lim, lim)
    ax3.set_zlim(-lim, lim)
    ax3.set_box_aspect((1, 1, 1))
    ax3.set_xlabel("x [m/s²]")
    ax3.set_ylabel("y [m/s²]")
    ax3.set_zlabel("z [m/s²]")
    ax3.legend(loc="upper left")
    ax3.set_title("accelerometer readings vs sphere of radius g")

    ax_a = fig.add_subplot(gs[0, 1])
    (l_ar,) = ax_a.plot([], [], color=RAW, lw=1, label="raw")
    (l_ac,) = ax_a.plot([], [], color=CAL, lw=1, label="calibrated")
    ax_a.axhline(0, color="k", lw=0.6)
    ax_a.set_title("|a| − g  [m/s²]   (0 = perfect, when still)", fontsize=10)
    ax_a.legend(fontsize=8, loc="upper left")
    ax_a.grid(alpha=0.3)

    ax_g = fig.add_subplot(gs[1, 1])
    (l_gr,) = ax_g.plot([], [], color=RAW, lw=1, label="raw")
    (l_gc,) = ax_g.plot([], [], color=CAL, lw=1, label="bias removed")
    ax_g.set_title("gyro magnitude  [deg/s]", fontsize=10)
    ax_g.legend(fontsize=8, loc="upper left")
    ax_g.grid(alpha=0.3)
    ax_g.set_xlabel("time [s]")

    ax_t = fig.add_subplot(gs[2, 1])
    ax_t.axis("off")
    txt = ax_t.text(0, 1, "", va="top", family="monospace", fontsize=10, transform=ax_t.transAxes)
    title = fig.suptitle("", fontsize=11)
    cal_name = str(path) if cal else "NO CALIBRATION FILE: showing raw only (run tools/calibrate_imu.py)"

    def update(_):
        for it in src.drain():
            if not isinstance(it, ImuSample):
                continue
            state["n"] += 1
            c_acc = cal.A_inv @ (it.acc - cal.eff_acc_bias()) if cal else it.acc
            c_gyr = it.gyro - cal.eff_gyro_bias() if cal else it.gyro
            er, ec = np.linalg.norm(it.acc) - G, np.linalg.norm(c_acc) - G
            wr, wc = np.linalg.norm(it.gyro), np.linalg.norm(c_gyr)
            hist.append((it.t, er, ec, np.degrees(wr), np.degrees(wc)))
            if wc < STILL_GYRO:
                still_err["raw"].append(er)
                still_err["cal"].append(ec)
            if state["n"] % 4 == 0:
                pts_raw.append(it.acc)
                pts_cal.append(c_acc)
        if not hist:
            title.set_text(f"[{imu}] waiting for data ...   {src.last_error}")
            return
        t_now = hist[-1][0]
        while hist and hist[0][0] < t_now - WINDOW_S:
            hist.popleft()
        H = np.array(hist)
        t = H[:, 0] - t_now
        l_ar.set_data(t, H[:, 1])
        l_gr.set_data(t, H[:, 3])
        if cal:
            l_ac.set_data(t, H[:, 2])
            l_gc.set_data(t, H[:, 4])
        for ax in (ax_a, ax_g):
            ax.set_xlim(-WINDOW_S, 0)
            ax.relim()
            ax.autoscale_view(scalex=False)
        if pts_raw:
            R = np.array(pts_raw)
            sc_raw._offsets3d = (R[:, 0], R[:, 1], R[:, 2])
            if cal:
                C = np.array(pts_cal)
                sc_cal._offsets3d = (C[:, 0], C[:, 1], C[:, 2])

        def rms(x):
            return np.sqrt(np.mean(np.square(x))) if x else float("nan")
        lines = [f"still samples: {len(still_err['raw'])}",
                 f"|a| − g RMS while still   raw {rms(still_err['raw']):.4f}   calibrated {rms(still_err['cal']):.4f}"
                 + " m/s²" if cal else f"|a| − g RMS while still   raw {rms(still_err['raw']):.4f} m/s²"]
        if cal:
            lines.append(f"acc bias {np.round(cal.acc_bias, 3)} m/s²   gyro bias "
                         f"{np.round(np.degrees(cal.gyro_bias), 2)} deg/s")
        txt.set_text("\n".join(lines))
        title.set_text(f"[{imu}] {getattr(src, 'port', '')}  {src.rate_hz:.0f} Hz   "
                       f"{'SIM ' + a.sim if sim else 'LIVE'}\ncalibration: {cal_name}")

    try:
        if a.snapshot:
            t_end = time.monotonic() + a.seconds
            while time.monotonic() < t_end:
                time.sleep(0.1)
                update(0)
            fig.savefig(a.snapshot, dpi=110)
            print(f"saved {a.snapshot}")
        else:
            _anim = FuncAnimation(fig, update, interval=150, cache_frame_data=False)  # noqa: F841
            plt.show()
    finally:
        src.stop()
        src.join(1.0)


if __name__ == "__main__":
    main()
