#!/usr/bin/env python3
"""Measure speed_to_erpm_gain (motor ERPM per m/s of car speed) by pushing the car a known distance.

The VESC's tachometer counts 6 steps per electrical motor revolution, and ERPM is electrical revolutions
per minute, so:   gain = ERPM / v = (dtacho / 6 * 60 / dt) / (ds / dt) = 10 * dtacho / ds
(dt cancels: push speed doesn't matter).

    python tools/measure_erpm_gain.py --vesc /dev/ttyACM0 --distance 3.0
    python tools/measure_erpm_gain.py --vesc /dev/ttyACM0 --distance 3.0 --runs 3 --write

Each run: car still on the start mark -> push it straight, by hand, exactly `distance` meters -> stop and
let go. Push at walking pace: very slow pushes can be undercounted by the sensorless motor observer.
For more runs, push it back the same distance (direction doesn't matter, the size of the count does).
Run 1 must be FORWARD: its sign decides the sign of the gain (so forward reads as positive speed).
--write puts the averaged gain into config/car.yaml.
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from racer_imu.cli import add_common_args, setup  # noqa: E402
from racer_imu.types import WheelSample  # noqa: E402

STILL_S = 1.5        # tachometer unchanged this long = standing still
MOVE_COUNTS = 6      # more than one electrical revolution = it moved


def one_run(src, k, runs, distance, timeout):
    """Returns (dtacho, peak_erpm) for one push, or None on timeout."""
    state, t0_tacho, last_change, last_tacho, peak = "still", None, time.monotonic(), None, 0.0
    t_end = time.monotonic() + timeout
    print(f"\nrun {k}/{runs}: " + ("push FORWARD " if k == 1 else "push ") + f"{distance:g} m straight, then let go")
    while time.monotonic() < t_end:
        time.sleep(0.02)
        for it in src.drain():
            if not isinstance(it, WheelSample) or not np.isfinite(it.tacho):
                continue
            now = time.monotonic()
            if last_tacho is None or it.tacho != last_tacho:
                last_change, last_tacho = now, it.tacho
            still_for = now - last_change
            if state == "still":
                if still_for >= STILL_S:
                    t0_tacho, state = it.tacho, "ready"
                    print("\r\x1b[2K  start recorded -> push now", flush=True)
                else:
                    print(f"\r\x1b[2K  keep it still on the start mark {still_for:.1f}/{STILL_S:.1f} s", end="", flush=True)
            elif state == "ready":
                if abs(it.tacho - t0_tacho) > MOVE_COUNTS:
                    state = "pushing"
            if state == "pushing":
                peak = max(peak, abs(it.erpm))
                d = it.tacho - t0_tacho
                if still_for >= STILL_S:
                    print(f"\r\x1b[2K  stopped: {d:+.0f} counts", flush=True)
                    return d, peak
                print(f"\r\x1b[2K  pushing... {d:+.0f} counts ({abs(it.erpm):.0f} ERPM)"
                      + (f"   stop at {distance:g} m and let go" if still_for > 0.3 else ""), end="", flush=True)
    print(f"\n  timed out ({state})")
    return None


def write_gain(cfg_path: Path, gain: float):
    """Replace the speed_to_erpm_gain line in car.yaml, keeping the rest of the file (and its comments)."""
    text = cfg_path.read_text()
    line = (f"speed_to_erpm_gain: {gain:.1f}    # erpm = gain * v[m/s] + offset; "
            f"measured {time.strftime('%Y-%m-%d')} (tools/measure_erpm_gain.py)")
    new, n = re.subn(r"speed_to_erpm_gain:[^\n]*", line, text, count=1)
    if n != 1:
        sys.exit("could not find speed_to_erpm_gain in the config")
    cfg_path.write_text(new)


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter),
                         sim_default="push")
    ap.add_argument("--distance", type=float, required=True, help="how far you push it each run [m]")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=90.0, help="per run [s]")
    ap.add_argument("--write", action="store_true", help="store the result in config/car.yaml")
    a = ap.parse_args()
    cfg, sources, sim = setup(a, only=["vesc"])
    if not sources:
        sys.exit("no VESC port (pass --vesc /dev/ttyACM0)")
    src = sources[0]
    src.wheel_every = 1          # read the tachometer on every poll
    src.start()
    counts, peaks = [], []
    try:
        for k in range(1, a.runs + 1):
            r = one_run(src, k, a.runs, a.distance, a.timeout)
            if r is None:
                break
            counts.append(r[0])
            peaks.append(r[1])
    finally:
        src.stop()
        src.join(1.0)
        if sim:
            sim.stop()
    if not counts:
        sys.exit("no complete run")
    mags = np.abs(counts)
    sign = 1.0 if counts[0] > 0 else -1.0
    gains = 10.0 * mags / a.distance
    gain = sign * float(np.mean(gains))
    print("\n" + "\n".join(f"  run {i + 1}: {c:+.0f} counts -> {g:.1f}   (peak {p:.0f} ERPM)"
                           for i, (c, g, p) in enumerate(zip(counts, gains, peaks))))
    print(f"\nspeed_to_erpm_gain = {gain:.1f}" + (f"   (spread {np.ptp(gains) / np.mean(gains) * 100:.1f} % over {len(gains)} runs)"
                                                 if len(gains) > 1 else ""))
    print(f"  currently in car.yaml: {cfg.car.speed_to_erpm_gain:.1f}  -> "
          f"wheel speeds were off by {(cfg.car.speed_to_erpm_gain / gain - 1) * 100:+.1f} %")
    if sign < 0:
        print("  the forward push counted DOWN, so the gain is negative: forward motion will read as positive speed")
    if min(peaks) < 300:
        print("  note: a push was very slow (< 300 ERPM); repeat at walking pace if the runs disagree")
    if a.write:
        if sim:
            print("  (--write ignored in --sim)")
        else:
            write_gain(cfg.path, gain)
            print(f"  written to {cfg.path}")


if __name__ == "__main__":
    main()
