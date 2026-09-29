import numpy as np

from racer_imu.clock_sync import ClockMapper, estimate_time_offset


def test_clock_mapper_removes_jitter_and_drift():
    rng = np.random.default_rng(0)
    cm = ClockMapper()
    errs = []
    for k in range(200 * 60):
        t = k / 200
        t_dev = 5.0 + t * (1 + 50e-6)           # offset + 50 ppm drift
        t_recv = t + 0.001 + rng.exponential(0.002)   # min latency 1 ms + jitter
        est = cm.add(t_dev, t_recv)
        if t > 10:
            errs.append(est - (t + 0.001))
    errs = np.array(errs)
    assert np.abs(errs).max() < 0.5e-3, np.abs(errs).max()


def test_time_offset_recovered():
    rng = np.random.default_rng(1)
    t = np.arange(0, 20, 0.005)
    sig = lambda x: np.sin(2 * np.pi * 1.3 * x) + 0.5 * np.sin(2 * np.pi * 2.9 * x + 1)  # noqa: E731
    true_late = 0.0123                     # b's stamps are 12.3 ms late
    tb = np.arange(0, 20, 0.0047)
    xa = sig(t) + rng.normal(0, 0.02, len(t))
    xb = sig(tb - true_late) + rng.normal(0, 0.02, len(tb))
    dt, c = estimate_time_offset(t, xa, tb, xb)
    assert abs(dt - (-true_late)) < 0.5e-3 and c > 0.9
