#!/usr/bin/env python3
"""Pole-zero plot and response of the VESC's built-in IMU low-pass filters (accel vs gyro, overlaid).

The firmware (bldc imu/imu.c + util/digital_filter.c) runs every enabled axis through a 2nd-order
low-pass biquad, after rotation/offsets and before the AHRS:

    K = tan(pi * Fc),  Fc = cutoff_Hz / sample_rate_Hz,  Q = 0.707,  norm = 1 / (1 + K/Q + K^2)
    a0 = K^2 norm,  a1 = 2 a0,  a2 = a0,  b1 = 2 (K^2 - 1) norm,  b2 = (1 - K/Q + K^2) norm
    H(z) = (a0 + a1 z^-1 + a2 z^-2) / (1 + b1 z^-1 + b2 z^-2)
A cutoff of 0 Hz means the filter is OFF (H = 1: no poles, no zeros, no delay).

Read your values in VESC Tool -> App Settings -> IMU: "Sample Rate", "Accel Low Pass Filter X/Y/Z",
"Gyro Low Pass Filter".

    python tools/imu_filter_plot.py --rate 200 --accel-hz 30 --gyro-hz 50
    python tools/imu_filter_plot.py --rate 1000 --accel-hz 40 40 60 --gyro-hz 80 --save filt.png

Also marks the Nyquist frequency of our polling rate: anything the filter passes above it aliases
into the logged data. And shows the filter's delay, which adds to the IMU's time offset.
"""
import argparse
import sys

import numpy as np
from scipy import signal

ACC_COLORS = ("#d95f02", "#e7298a", "#a6761d")   # accel x, y, z
GYRO_COLOR = "#1b9e77"


def vesc_biquad(cutoff_hz: float, rate_hz: float):
    """(b, a) in scipy convention, exactly as the firmware's biquad_config(BQ_LOWPASS)."""
    Fc = cutoff_hz / rate_hz
    if not 0 < Fc < 0.5:
        raise ValueError(f"cutoff {cutoff_hz} Hz must be between 0 and rate/2 = {rate_hz / 2} Hz")
    K = np.tan(np.pi * Fc)
    Q = 0.707
    norm = 1 / (1 + K / Q + K * K)
    a0 = K * K * norm
    b = np.array([a0, 2 * a0, a0])
    a = np.array([1.0, 2 * (K * K - 1) * norm, (1 - K / Q + K * K) * norm])
    return b, a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--rate", type=float, default=200.0, help="IMU sample rate on the VESC [Hz] (firmware default 200)")
    ap.add_argument("--accel-hz", type=float, nargs="+", default=[0.0], help="accel cutoff: one value or X Y Z [Hz]")
    ap.add_argument("--gyro-hz", type=float, default=0.0, help="gyro cutoff [Hz]")
    ap.add_argument("--poll-hz", type=float, default=200.0, help="how fast our driver polls the VESC [Hz]")
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    acc = a.accel_hz * 3 if len(a.accel_hz) == 1 else a.accel_hz
    if len(acc) != 3:
        sys.exit("--accel-hz takes 1 or 3 values")

    filters = []   # (label, color, b, a, cutoff)
    seen = {}
    for axis, (hz, col) in zip("xyz", zip(acc, ACC_COLORS)):
        if hz > 0:
            key = round(hz, 6)
            if key in seen:                          # same cutoff as another axis: merge the label
                seen[key][0] += axis
                continue
            b, aa = vesc_biquad(hz, a.rate)
            seen[key] = [f"accel {axis}", col, b, aa, hz]
    filters += list(seen.values())
    if a.gyro_hz > 0:
        b, aa = vesc_biquad(a.gyro_hz, a.rate)
        filters.append(["gyro", GYRO_COLOR, b, aa, a.gyro_hz])
    if not filters:
        print("All cutoffs are 0: the VESC IMU filters are OFF (H(z) = 1, no poles or zeros, no delay).")
        print("Pass the values from VESC Tool, e.g.  --accel-hz 30 --gyro-hz 50")
        return

    import matplotlib
    if a.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    ax_pz, ax_mag, ax_gd, ax_step = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    # pole-zero, all filters on one unit circle
    th = np.linspace(0, 2 * np.pi, 400)
    ax_pz.plot(np.cos(th), np.sin(th), color="#999", lw=0.8)
    ax_pz.axhline(0, color="#ccc", lw=0.6)
    ax_pz.axvline(0, color="#ccc", lw=0.6)
    report = []
    for i, (label, col, b, aa, hz) in enumerate(filters):
        z, p, _ = signal.tf2zpk(b, aa)
        ax_pz.plot(p.real, p.imag, "x", color=col, ms=12, mew=2.5, label=f"{label} poles ({hz:g} Hz)")
        # both zeros sit at z = -1 for every filter: offset the markers slightly so each colour shows
        ax_pz.plot(z.real, z.imag + (i - (len(filters) - 1) / 2) * 0.06, "o", mfc="none", color=col, ms=10,
                   mew=2, label=f"{label} zeros (double, z = -1)")
        w, gd = signal.group_delay((b, aa), w=2048, fs=a.rate)
        report.append(f"{label:10s} fc {hz:6.1f} Hz  poles {np.round(p, 4)}  |p| {np.abs(p[0]):.4f}"
                      f"  delay at DC {gd[0] * 1e3 / a.rate:.2f} ms")
    ax_pz.set_aspect("equal")
    ax_pz.set_xlim(-1.3, 1.3)
    ax_pz.set_ylim(-1.3, 1.3)
    ax_pz.set_title("pole-zero (z-plane)   x = pole, o = zero")
    ax_pz.set_xlabel("Re z")
    ax_pz.set_ylabel("Im z")
    ax_pz.legend(fontsize=8, loc="lower left")
    ax_pz.grid(alpha=0.3)

    # magnitude response
    for label, col, b, aa, hz in filters:
        f, h = signal.freqz(b, aa, worN=4096, fs=a.rate)
        ax_mag.plot(f, 20 * np.log10(np.maximum(np.abs(h), 1e-6)), color=col, lw=2, label=f"{label} ({hz:g} Hz)")
        ax_mag.axvline(hz, color=col, ls=":", lw=1)
    ax_mag.axhline(-3, color="#999", ls="--", lw=0.8)
    ax_mag.axvline(a.poll_hz / 2, color="k", ls="--", lw=1.2, label=f"our poll Nyquist ({a.poll_hz / 2:g} Hz)")
    ax_mag.set_xlim(0, a.rate / 2)
    ax_mag.set_ylim(-60, 5)
    ax_mag.set_xlabel("frequency [Hz]")
    ax_mag.set_ylabel("gain [dB]")
    ax_mag.set_title(f"magnitude response (VESC samples at {a.rate:g} Hz)")
    ax_mag.legend(fontsize=8)
    ax_mag.grid(alpha=0.3)

    # group delay (adds to the IMU's timing offset)
    for label, col, b, aa, hz in filters:
        w, gd = signal.group_delay((b, aa), w=2048, fs=a.rate)
        ax_gd.plot(w, gd / a.rate * 1e3, color=col, lw=2, label=label)
    ax_gd.set_xlim(0, a.rate / 2)
    ax_gd.set_xlabel("frequency [Hz]")
    ax_gd.set_ylabel("delay [ms]")
    ax_gd.set_title("group delay (how late the filtered signal is)")
    ax_gd.legend(fontsize=8)
    ax_gd.grid(alpha=0.3)

    # step response
    n = int(0.15 * a.rate) + 2
    for label, col, b, aa, hz in filters:
        y = signal.lfilter(b, aa, np.ones(n))
        ax_step.step(np.arange(n) / a.rate * 1e3, y, where="post", color=col, lw=2, label=label)
    ax_step.axhline(1, color="#999", lw=0.8)
    ax_step.set_xlabel("time [ms]")
    ax_step.set_ylabel("output")
    ax_step.set_title("step response (sudden change in acceleration / rotation)")
    ax_step.legend(fontsize=8)
    ax_step.grid(alpha=0.3)

    fig.suptitle("VESC IMU low-pass biquads (2nd-order Butterworth, Q = 0.707)", fontsize=12)
    fig.tight_layout()
    print("\n".join(report))
    for label, col, b, aa, hz in filters:
        f, h = signal.freqz(b, aa, worN=4096, fs=a.rate)
        g = np.interp(a.poll_hz / 2, f, np.abs(h)) if a.poll_hz / 2 < a.rate / 2 else None
        if g is not None and g > 0.1:
            print(f"note: {label} still passes {20 * np.log10(g):.1f} dB at our poll Nyquist ({a.poll_hz / 2:g} Hz):"
                  " content above it will alias in the logs. Lower the cutoff or poll faster.")
    if a.save:
        fig.savefig(a.save, dpi=110)
        print(f"saved {a.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
