"""Online calibration while driving: gyro re-zeroing at stops, lidar corrections, late-measurement handling."""
import numpy as np

from racer_imu.calibration import ImuCalibration
from racer_imu.estimator import EkfParams, wrap
from racer_imu.frames import load_config
from racer_imu.pipeline import Pipeline
from racer_imu.sim import default_truth, expected_sync_offset, generate_offline

T0 = 1000.0   # generate_offline's host-clock epoch
DRIFT = np.radians(0.02)   # gyro bias creeps 0.02 deg/s every second (~1.2 deg/s per minute): a warming sensor


def _run(scenario, online=True, lidar=None, lidar_hz=10.0, latency=0.06, outlier_at=None, seed=0):
    cfg = load_config()
    truth = default_truth(cfg)
    for st in truth.values():
        st.gyro_bias_drift = np.array([0.0, 0.0, DRIFT])
    streams, traj, truth = generate_offline(cfg, scenario, truth=truth, seed=seed)
    calibs = {n: ImuCalibration(np.linalg.inv(s.scale), s.acc_bias, s.gyro_bias) for n, s in truth.items()}
    pipe = Pipeline(cfg, names=list(truth), calibs=calibs, offsets={"bno": expected_sync_offset(truth)},
                    persist_bias=False)
    pipe.ekf.p = EkfParams(online_gyro_bias=online)
    if not online:
        pipe.ekf.P[6, 6] = 0.0
    rng = np.random.default_rng(seed + 7)
    items = sorted([x for k in streams for x in streams[k]], key=lambda s: s.t)
    outs, next_lidar = [], T0 + 1.0
    for i, it in enumerate(items):
        pipe.feed(it)
        if i % 20:
            continue
        outs += pipe.process(it.t)
        if lidar and outs and outs[-1]["t"] >= next_lidar + latency:
            t_meas = next_lidar                      # scan taken at t_meas, result arrives `latency` later
            tr = traj.truth(t_meas - T0)
            if lidar == "heading":
                psi = tr["psi"] + rng.normal(0, np.radians(1.0))
                if outlier_at is not None and abs(t_meas - T0 - outlier_at) < 0.05:
                    psi += np.radians(90)               # a bad scan match
                pipe.lidar_heading(t_meas, psi, np.radians(1.0))
            elif lidar == "pose":
                pipe.lidar_pose(t_meas, tr["x"] + rng.normal(0, 0.03), tr["y"] + rng.normal(0, 0.03),
                                tr["psi"] + rng.normal(0, np.radians(1.0)), 0.03, np.radians(1.0))
            next_lidar += 1.0 / lidar_hz
    outs += pipe.process(items[-1].t + 1)
    return outs, traj, pipe


def _heading_err_deg(outs, traj):
    o = outs[-1]
    return abs(np.degrees(wrap(o["est"]["psi"] - traj.truth(o["t"] - T0)["psi"])))


def test_rezero_at_stops_beats_fixed_bias():
    off, traj, _ = _run("stopgo", online=False)
    on, _, pipe = _run("stopgo", online=True)
    e_off, e_on = _heading_err_deg(off, traj), _heading_err_deg(on, traj)
    t_end = on[-1]["t"] - T0
    true_bias = DRIFT * t_end
    est_bias = on[-1]["est"]["gyro_bias"]
    assert e_on < 0.3 * e_off, (e_on, e_off)
    assert abs(est_bias - true_bias) < np.radians(0.1), (np.degrees(est_bias), np.degrees(true_bias))
    assert sum(o["standstill"] for o in on) > 500               # it did detect the stops
    moving = [o for o in on if abs(traj.truth(o["t"] - T0)["vx"]) > 0.5]
    assert not any(o["standstill"] for o in moving)             # and never while moving


def test_lidar_heading_calibrates_gyro_without_stopping():
    """circle never stops mid-run: only the lidar heading can keep the bias estimate right."""
    outs, traj, pipe = _run("circle", online=True, lidar="heading", outlier_at=20.0)
    t_end = outs[-1]["t"] - T0
    est_bias = outs[-1]["est"]["gyro_bias"]
    assert abs(est_bias - DRIFT * t_end) < np.radians(0.1), (np.degrees(est_bias), np.degrees(DRIFT * t_end))
    assert _heading_err_deg(outs, traj) < 2.0
    assert pipe.ekf.rejected.get("heading", 0) >= 1               # the 90 deg bad scan was thrown out


def test_late_lidar_pose_is_moved_forward():
    outs, traj, pipe = _run("circle", online=True, lidar="pose", latency=0.08)
    errs = [np.hypot(o["est"]["x"] - traj.truth(o["t"] - T0)["x"], o["est"]["y"] - traj.truth(o["t"] - T0)["y"])
            for o in outs[len(outs) // 4:]]
    assert np.sqrt(np.mean(np.square(errs))) < 0.08, np.sqrt(np.mean(np.square(errs)))   # 2 m/s * 80 ms = 16 cm uncompensated
