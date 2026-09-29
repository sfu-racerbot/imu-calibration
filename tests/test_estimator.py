"""End-to-end offline: simulator -> calibration (true values) -> sync -> fusion -> EKF vs ground truth."""
import numpy as np

from racer_imu.calibration import ImuCalibration
from racer_imu.frames import load_config
from racer_imu.pipeline import Pipeline
from racer_imu.sim import expected_sync_offset, generate_offline


def _run(scenario, duration=None, ext_vel_hz=0.0):
    cfg = load_config()
    streams, traj, truth = generate_offline(cfg, scenario, duration)
    calibs = {n: ImuCalibration(np.linalg.inv(s.scale), s.acc_bias, s.gyro_bias) for n, s in truth.items()}
    pipe = Pipeline(cfg, names=list(truth), calibs=calibs, offsets={"bno": expected_sync_offset(truth)})
    items = sorted([x for k in streams for x in streams[k]], key=lambda s: s.t)
    rng = np.random.default_rng(5)
    outs = []
    for i, it in enumerate(items):
        pipe.feed(it)
        if i % 50 == 0:
            new = pipe.process(it.t)
            outs += new
            # simulated lidar odometry: noisy true car-frame velocity at ext_vel_hz
            if ext_vel_hz and new and int(new[-1]["t"] * ext_vel_hz) != int(new[0]["t"] * ext_vel_hz - 1e-9):
                tr = traj.truth(new[-1]["t"] - 1000.0)
                pipe.ekf.update_velocity(tr["vx"] + rng.normal(0, 0.1), tr["vy"] + rng.normal(0, 0.1), 0.1)
    outs += pipe.process(items[-1].t + 1)
    return outs, traj


def _beta_rms_deg(outs, traj):
    e = [o["est"]["beta"] - traj.truth(o["t"] - 1000.0)["beta"] for o in outs
         if abs(traj.truth(o["t"] - 1000.0)["vx"]) > 1.0]
    return float(np.degrees(np.sqrt(np.mean(np.square(e))))), len(e)


def test_drift_beta_with_wheel_spin():
    """35 % wheel spin during the slide: wheel speed over-reads vx, so beta is off by a few degrees."""
    outs, traj = _run("drift")
    t0 = 1000.0
    err_beta, err_r = [], []
    for o in outs:
        tr = traj.truth(o["t"] - t0)
        if abs(tr["vx"]) > 1.0:
            err_beta.append(o["est"]["beta"] - tr["beta"])
            err_r.append(o["est"]["r"] - tr["r"])
    err_beta = np.degrees(err_beta)
    assert len(err_beta) > 1000
    assert np.sqrt(np.mean(np.square(err_beta))) < 6.0, np.sqrt(np.mean(np.square(err_beta)))
    assert np.sqrt(np.mean(np.square(err_r))) < 0.05


def test_drift_beta_with_external_velocity():
    rms, n = _beta_rms_deg(*_run("drift", ext_vel_hz=20.0))
    assert n > 1000 and rms < 2.5, rms


def test_slalom_beta():
    rms, n = _beta_rms_deg(*_run("slalom"))
    assert n > 1000 and rms < 2.0, rms


def test_circle_path():
    outs, traj = _run("circle")
    o = outs[len(outs) // 2]
    tr = traj.truth(o["t"] - 1000.0)
    assert np.hypot(o["est"]["x"] - tr["x"], o["est"]["y"] - tr["y"]) < 1.0
