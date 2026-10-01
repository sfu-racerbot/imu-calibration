"""Each verification check passes a correct calibration and catches a deliberately wrong one."""
import numpy as np

from racer_imu.calibration import ImuCalibration
from racer_imu.frames import load_config
from racer_imu.sim import default_truth, generate_offline
from racer_imu.types import G
from racer_imu.verify import PoseCheck, TurnCheck, check_still, flip_result, turn_result


def _vesc(scenario, gyro_scale=1.0):
    cfg = load_config()
    truth = default_truth(cfg)
    truth["vesc"].gyro_scale = gyro_scale
    streams, traj, truth = generate_offline(cfg, scenario, truth=truth)
    st = truth["vesc"]
    good = ImuCalibration(np.linalg.inv(st.scale), st.acc_bias.copy(), st.gyro_bias.copy())
    return streams["vesc"], good


def _turn(samples, cal, min_deg):
    tc = TurnCheck(min_turn_deg=min_deg)
    for s in samples:
        if tc.add(cal.apply(s), s) == "done":
            break
    return tc


def test_still_catches_gyro_bias():
    samples, good = _vesc("still")
    assert check_still([good.apply(s) for s in samples])["grade_gyro"] == "PASS"
    bad = ImuCalibration(good.A_inv, good.acc_bias, good.gyro_bias - np.radians([0, 0, 0.1]))
    r = check_still([bad.apply(s) for s in samples])
    assert abs(r["heading_drift_deg_min"] - 6.0) < 0.5 and r["grade_gyro"] == "FAIL"    # 0.1 deg/s = 6 deg/min


def test_poses_catch_scale_error():
    samples, good = _vesc("poses")
    for cal, expect in [(good, "PASS"), (ImuCalibration(good.A_inv * 1.01, good.acc_bias, good.gyro_bias), "FAIL")]:
        pc = PoseCheck(cal)
        for s in samples:
            pc.add(s)
        assert pc.result()["n"] >= 6 and pc.result()["grade"] == expect, pc.result()


def test_flip_measures_leftover_bias_on_a_tilted_table():
    samples, good = _vesc("flip")
    r = flip_result(_turn(samples, good, 90), good)
    assert abs(abs(r["turned_deg"]) - 180) < 3 and r["grade"] == "PASS" and "warning" not in r
    assert np.linalg.norm(r["residual_bias_ms2"]) < 0.01 and 1.5 < r["table_tilt_deg"] < 3.5
    off = np.array([0.12, -0.09, 0.0])                                   # calibration bias wrong by this much
    bad = ImuCalibration(good.A_inv, good.acc_bias + off, good.gyro_bias)
    r = flip_result(_turn(samples, bad, 90), bad)
    assert abs(np.linalg.norm(r["residual_bias_ms2"]) - np.linalg.norm(off[:2])) < 0.015 and r["grade"] == "FAIL"


def test_turn_measures_gyro_scale():
    samples, good = _vesc("turn", gyro_scale=1.02)
    r = turn_result(_turn(samples, good, 300), 1)
    assert abs(r["scale_error_pct"] - 2.0) < 0.3 and r["grade"] == "OK"
