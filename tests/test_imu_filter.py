"""The VESC biquad formula (bldc util/digital_filter.c) is a 2nd-order Butterworth low-pass."""
import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from imu_filter_plot import vesc_biquad  # noqa: E402


def test_matches_butterworth():
    for fc, fs in [(30, 200), (50, 1000), (5, 200)]:
        b, a = vesc_biquad(fc, fs)
        bb, ab = signal.butter(2, fc, fs=fs)
        assert np.allclose(b, bb, rtol=2e-3) and np.allclose(a, ab, rtol=2e-3, atol=1e-4)
        f, h = signal.freqz(b, a, worN=[fc], fs=fs)
        assert abs(20 * np.log10(abs(h[0])) + 3.0) < 0.05          # -3 dB at the cutoff
        z, p, _ = signal.tf2zpk(b, a)
        assert np.allclose(z, -1) and np.all(np.abs(p) < 1)       # double zero at -1, stable poles
