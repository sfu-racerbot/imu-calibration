#!/usr/bin/env python3
"""Car-frame GUI: where the IMUs sit, what they feel, and where the car goes / how much it drifts.

    python tools/car_view.py                          # live, ports from config/car.yaml
    python tools/car_view.py --sim drift              # simulator (shows ground truth dashed)
    python tools/car_view.py --replay logs/<run_dir>  # play back a recorded run
    python tools/car_view.py --sim drift --log        # also record raw + estimates
    python tools/car_view.py --sim drift --snapshot out.png --seconds 15   # headless, save one image
    python tools/car_view.py --vesc /dev/ttyACM0 --drive --log   # drive with the keyboard + log

Driving (--drive): click the window, Enter = arm, arrows/WASD = throttle + steering, Space = brake,
Esc = disarm. Throttle is duty-cycle limited (drive.max_duty in car.yaml, or --max-duty) and releases
by itself 0.3 s after the window stops sending (dead-man). WHEELS OFF THE GROUND the first time.

Left: top-down car (x forward = up on screen), IMU positions + mounting axes, acceleration arrows
      (thin = what each IMU feels, thick = moved to the car center), velocity arrow and drift angle beta.
Middle: estimated path (and truth in sim).  Right: yaw rate from each IMU (they overlap when time sync is
good), lateral acceleration, beta, speeds.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from racer_imu.cli import add_common_args, setup, use_sim_paths  # noqa: E402
from racer_imu.frames import load_config  # noqa: E402
from racer_imu.logger import RunLogger, load_run  # noqa: E402
from racer_imu.pipeline import Pipeline  # noqa: E402
from racer_imu.teleop import Teleop, TeleopParams  # noqa: E402

COLORS = {"vesc": "#d95f02", "bno": "#1b9e77", "center": "#222222", "est": "#7570b3", "truth": "#888888"}
ACC_SCALE = 0.025     # m of arrow per m/s^2
VEL_SCALE = 0.06      # m of arrow per m/s
WINDOW_S = 10.0


def scr(x, y):
    """car frame (x fwd, y left) -> screen (right, up)."""
    return -np.asarray(y), np.asarray(x)


class Replay:
    """Feeds a recorded run into the pipeline on a virtual clock."""

    def __init__(self, run_dir, speed=1.0):
        streams = load_run(run_dir)
        self.items = sorted([x for v in streams.values() for x in v], key=lambda s: s.t)
        self.names = [k for k in streams if k != "wheel"]
        self.i = 0
        self.speed = speed
        self.t_wall0 = None
        self.t_data0 = self.items[0].t if self.items else 0.0

    def now(self):
        if self.t_wall0 is None:
            self.t_wall0 = time.monotonic()
        return self.t_data0 + (time.monotonic() - self.t_wall0) * self.speed

    def pump(self, pipe):
        now = self.now()
        while self.i < len(self.items) and self.items[self.i].t <= now:
            pipe.feed(self.items[self.i])
            self.i += 1
        return now


def draw_car(ax, cfg, names):
    import matplotlib.patches as mp
    c = cfg.car
    ax.add_patch(mp.Rectangle(scr(-c.length / 2, c.width / 2), c.width, c.length, fill=True,
                              fc="#f2f2f2", ec="#555", lw=1.5, zorder=0))
    for sx in (1, -1):
        for sy in (1, -1):
            wx, wy = sx * c.wheelbase / 2, sy * c.track / 2
            ax.add_patch(mp.Rectangle(scr(wx - c.wheel_radius, wy + c.wheel_width / 2), c.wheel_width,
                                      2 * c.wheel_radius, fc="#333", ec="none", zorder=1))
    # center frame
    ax.annotate("", xy=scr(0.08, 0), xytext=scr(0, 0), arrowprops=dict(arrowstyle="->", color="r", lw=2))
    ax.annotate("", xy=scr(0, 0.08), xytext=scr(0, 0), arrowprops=dict(arrowstyle="->", color="g", lw=2))
    ax.text(*scr(0.09, -0.01), "x", color="r")
    ax.text(*scr(0.0, 0.10), "y", color="g")
    ax.plot(*scr(0, 0), "k+", ms=12, mew=2)
    ax.text(*scr(-0.02, -0.01), "center", fontsize=8, va="top")
    for n in names:
        ex = cfg.imus[n]
        col = COLORS.get(n, "b")
        ax.plot(*scr(ex.r[0], ex.r[1]), "s", color=col, ms=10, zorder=5)
        ax.text(*scr(ex.r[0] + 0.02, ex.r[1] - 0.02), f"{n}\n({ex.r[0]:+.2f}, {ex.r[1]:+.2f}, {ex.r[2]:+.2f})",
                color=col, fontsize=8)
        for k, cc in ((0, "r"), (1, "g")):   # sensor x/y axes, drawn in the car frame
            d = ex.R[:, k] * 0.05
            ax.annotate("", xy=scr(ex.r[0] + d[0], ex.r[1] + d[1]), xytext=scr(ex.r[0], ex.r[1]),
                        arrowprops=dict(arrowstyle="-|>", color=cc, lw=1, alpha=0.6))
    ax.set_aspect("equal")
    lim = max(c.length, c.width) * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title("car frame (top view, x forward)")
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    ap.add_argument("--replay", default=None, help="run directory under logs/")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed factor")
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--no-calib", action="store_true", help="ignore calib/*.json")
    ap.add_argument("--no-sync", action="store_true", help="ignore calib/sync.json offsets")
    ap.add_argument("--snapshot", default=None, help="run headless and save a PNG at the end")
    ap.add_argument("--seconds", type=float, default=15.0, help="with --snapshot: how long to run")
    ap.add_argument("--drive", action="store_true", help="enable keyboard driving through the VESC")
    ap.add_argument("--max-duty", type=float, default=None, help="override drive.max_duty from car.yaml")
    a = ap.parse_args()

    import matplotlib
    if a.snapshot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if a.drive:                                   # free the keys matplotlib uses for its own shortcuts
        for k in list(plt.rcParams):
            if k.startswith("keymap."):
                plt.rcParams[k] = []
    from matplotlib.animation import FuncAnimation

    sim = replay = None
    if a.replay:
        cfg = load_config(a.config)
        replay = Replay(a.replay, a.speed)
        names = [n for n in replay.names if n in cfg.imus]
        meta = Path(a.replay) / "meta.json"
        if meta.exists() and json.loads(meta.read_text()).get("sim"):
            use_sim_paths(cfg)
        sources = []
    else:
        cfg, sources, sim = setup(a, processes=True)
        names = [s.src_name for s in sources]
    if not names:
        sys.exit("no IMUs")
    log = RunLogger("drive" + ("_sim" if sim else ""), meta={"sim": bool(sim), "scenario": a.sim}) if a.log else None
    pipe = Pipeline(cfg, sources, names=names, logger=log, use_calib=not a.no_calib, use_sync=not a.no_sync).start()
    drive_src = teleop = None
    if a.drive:
        drive_src = next((s for s in sources if s.src_name.startswith("vesc") and hasattr(s, "set_drive")), None)
        if drive_src is None:
            sys.exit("--drive needs a live VESC (not --replay)")
        tp = TeleopParams.from_dict(cfg.section("drive"))
        if a.max_duty is not None:
            tp.max_duty = a.max_duty
        teleop = Teleop(tp)
        print(f"driving enabled: max duty {tp.max_duty:.2f}. Click the window, Enter = arm, Esc = disarm.")

    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(4, 3, width_ratios=[1.05, 1.1, 1.25], hspace=0.45, wspace=0.18)
    ax_car = fig.add_subplot(gs[:3, 0])
    ax_txt = fig.add_subplot(gs[3, 0])
    ax_txt.axis("off")
    ax_path = fig.add_subplot(gs[:, 1])
    ax_r, ax_ay, ax_b, ax_v = (fig.add_subplot(gs[i, 2]) for i in range(4))
    draw_car(ax_car, cfg, names)

    # dynamic artists on the car panel
    q_imu = {n: ax_car.quiver([0], [0], [0], [0], color=COLORS.get(n, "b"), angles="xy", scale_units="xy",
                              scale=1, width=0.006, zorder=6) for n in names}
    q_ctr = ax_car.quiver([0], [0], [0], [0], color=COLORS["center"], angles="xy", scale_units="xy", scale=1,
                          width=0.012, zorder=7)
    q_vel = ax_car.quiver([0], [0], [0], [0], color=COLORS["est"], angles="xy", scale_units="xy", scale=1,
                          width=0.012, zorder=7)
    txt = ax_txt.text(0.0, 1.0, "", transform=ax_txt.transAxes, family="monospace", fontsize=9, va="top")
    ax_car.legend(handles=[plt.Line2D([], [], color=COLORS["center"], lw=3, label="accel @ center"),
                           plt.Line2D([], [], color=COLORS["est"], lw=3, label="velocity"),
                           *[plt.Line2D([], [], color=COLORS.get(n, "b"), lw=1.5, label=f"accel @ {n}")
                             for n in names]], loc="upper left", fontsize=8)

    (l_path,) = ax_path.plot([], [], color=COLORS["est"], lw=2, label="estimate")
    (l_truth,) = ax_path.plot([], [], "--", color=COLORS["truth"], lw=1.5, label="truth (sim)")
    (l_car,) = ax_path.plot([], [], color="k", lw=2)
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.set_title("path (world, start = origin)")
    ax_path.set_xlabel("x [m]")
    ax_path.set_ylabel("y [m]")
    ax_path.grid(alpha=0.3)
    ax_path.legend(loc="upper left", fontsize=8)

    def strip(ax, title, series):
        lines = {k: ax.plot([], [], lw=1.3, label=lbl, color=col, ls=ls)[0] for k, lbl, col, ls in series}
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left", ncol=len(series))
        ax.tick_params(labelsize=8)
        return lines

    L_r = strip(ax_r, "yaw rate [rad/s]  (IMU traces overlap when synced)",
                [(f"r_{n}", n, COLORS.get(n, "b"), "-") for n in names] + [("r_est", "EKF", COLORS["est"], ":")])
    L_ay = strip(ax_ay, "accel at center [m/s^2]", [("ax", "a_x", "#1f77b4", "-"), ("ay", "a_y", "#d62728", "-")])
    L_b = strip(ax_b, "sideslip beta [deg]", [("beta", "estimate", COLORS["est"], "-"),
                                               ("beta_t", "truth", COLORS["truth"], "--")])
    L_v = strip(ax_v, "speed [m/s]", [("vw", "wheel", "#e7298a", "-"), ("vx", "EKF vx", COLORS["est"], "-"),
                                      ("vy", "EKF vy", "#66a61e", "-")])
    mode = "SIM " + a.sim if sim else ("REPLAY " + Path(a.replay).name if replay else "LIVE")
    title = fig.suptitle(mode, fontsize=11)
    state = {"last_lim": 0.0, "was_armed": False}
    if teleop:
        fig.canvas.mpl_connect("key_press_event", lambda e: teleop.press(e.key))
        fig.canvas.mpl_connect("key_release_event", lambda e: teleop.release(e.key))

    def drive_tick():
        cmd = teleop.update()
        if cmd is not None:
            drive_src.set_drive(**cmd, hold_s=0.3)
            state["was_armed"] = True
        elif state["was_armed"]:
            drive_src.stop_drive()
            state["was_armed"] = False
        if log:
            log.log_row("drive", {"t": time.monotonic(), "armed": int(teleop.armed),
                                  **{k: float(v) for k, v in (cmd or {}).items()}})

    def update(_):
        if teleop:
            drive_tick()
        new = pipe.process(replay.pump(pipe)) if replay else pipe.poll()
        if sim:
            for h in new:
                h["truth"] = sim.truth_at(h["t"])
        H = pipe.history
        if not H:
            return
        o = H[-1]
        est, c = o["est"], o["center"]
        # car panel
        for n in names:
            if n in c.per_imu:
                r = cfg.imus[n].r
                f = c.per_imu[n]["f_car"]
                q_imu[n].set_offsets([scr(r[0], r[1])])
                q_imu[n].set_UVC(*scr(f[0] * ACC_SCALE, f[1] * ACC_SCALE))
        q_ctr.set_UVC(*scr(c.acc[0] * ACC_SCALE, c.acc[1] * ACC_SCALE))
        q_vel.set_UVC(*scr(est["vx"] * VEL_SCALE, est["vy"] * VEL_SCALE))
        st = pipe.status()
        rates = "  ".join(f"{k} {v['rate_hz']:.0f}Hz" for k, v in st.items() if "rate_hz" in v)
        tb = f" (truth {np.degrees(o['truth']['beta']):+6.1f})" if "truth" in o else ""
        txt.set_text(f"speed {est['speed']:5.2f} m/s   wheel {o['v_wheel']:5.2f} m/s   slip {est['wheel_slip']:+.2f}\n"
                     f"beta  {np.degrees(est['beta']):+6.1f} deg{tb}   r {est['r']:+6.2f} rad/s\n"
                     f"a_c   ({c.acc[0]:+5.2f}, {c.acc[1]:+5.2f}) m/s^2   "
                     f"roll {np.degrees(c.roll):+5.1f}  pitch {np.degrees(c.pitch):+5.1f} deg\n"
                     f"sync frames {st['sync']['produced']} (skipped {st['sync']['skipped']})   {rates}\n"
                     f"gyro bias (online) {st['online']['gyro_bias_deg_s']:+.2f} +- {st['online']['std_deg_s']:.2f} deg/s"
                     + ("   PARKED: re-zeroing gyro" if st['online']['standstill'] else "")
                     + "".join(f"\n! {m}" for m in st['online']['health'])
                     + (f"\n{teleop.status()}" if teleop else ""))
        # path
        xs = np.array([h["est"]["x"] for h in H])
        ys = np.array([h["est"]["y"] for h in H])
        l_path.set_data(xs, ys)
        if sim:
            tr = [h["truth"] for h in H if "truth" in h]
            l_truth.set_data([t["x"] for t in tr], [t["y"] for t in tr])
        L, W = cfg.car.length / 2, cfg.car.width / 2
        corners = np.array([[L, W], [L, -W], [-L, -W], [-L, W], [L, W], [L * 1.4, 0], [L, -W]])
        cp, sp = np.cos(est["psi"]), np.sin(est["psi"])
        l_car.set_data(est["x"] + cp * corners[:, 0] - sp * corners[:, 1],
                       est["y"] + sp * corners[:, 0] + cp * corners[:, 1])
        if time.monotonic() - state["last_lim"] > 0.5:
            state["last_lim"] = time.monotonic()
            if sim:
                xs = np.r_[xs, l_truth.get_xdata()]
                ys = np.r_[ys, l_truth.get_ydata()]
            cx, cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
            half = max(np.ptp(xs), np.ptp(ys)) / 2 + 1.0
            ax_path.set_xlim(cx - half, cx + half)
            ax_path.set_ylim(cy - half, cy + half)
        # strips
        tnow = o["t"]
        sel = [h for h in H if h["t"] > tnow - WINDOW_S]
        t = np.array([h["t"] - tnow for h in sel])
        for n in names:
            L_r[f"r_{n}"].set_data(t, [(cfg.imus[n].R @ h["frame"].imus[n].gyro)[2] for h in sel])
        L_r["r_est"].set_data(t, [h["est"]["r"] for h in sel])
        L_ay["ax"].set_data(t, [h["center"].acc[0] for h in sel])
        L_ay["ay"].set_data(t, [h["center"].acc[1] for h in sel])
        L_b["beta"].set_data(t, [np.degrees(h["est"]["beta"]) for h in sel])
        if sim:
            L_b["beta_t"].set_data(t, [np.degrees(h["truth"]["beta"]) if "truth" in h else np.nan for h in sel])
        L_v["vw"].set_data(t, [h["v_wheel"] for h in sel])
        L_v["vx"].set_data(t, [h["est"]["vx"] for h in sel])
        L_v["vy"].set_data(t, [h["est"]["vy"] for h in sel])
        for ax in (ax_r, ax_ay, ax_b, ax_v):
            ax.set_xlim(-WINDOW_S, 0)
            ax.relim()
            ax.autoscale_view(scalex=False)
        title.set_text(f"{mode}   t = {tnow - H[0]['t']:.1f} s" + ("   [DRIVE ARMED]" if teleop and teleop.armed else ""))
        title.set_color("red" if teleop and teleop.armed else "black")

    try:
        if a.snapshot:
            t_end = time.monotonic() + a.seconds
            while time.monotonic() < t_end:
                time.sleep(0.05)
                update(0)
            fig.savefig(a.snapshot, dpi=110)
            print(f"saved {a.snapshot}")
        else:
            _anim = FuncAnimation(fig, update, interval=50, cache_frame_data=False)  # noqa: F841
            plt.show()
    finally:
        if drive_src is not None:
            drive_src.stop_drive()            # release the motor before shutting down
            time.sleep(0.1)
        pipe.stop()
        if getattr(pipe, "saved_bias_path", None):
            print(f"saved learned gyro bias -> {pipe.saved_bias_path} (used as the start value next time)")
        if log:
            print(f"saved log {log.dir}")


if __name__ == "__main__":
    main()
