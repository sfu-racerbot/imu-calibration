import numpy as np

from racer_imu.calibration import PoseCollector, fit_accel, kabsch, mount_rotation
from racer_imu.frames import rpy_to_R
from racer_imu.types import G, ImuSample


def _poses(n, rng):
    v = rng.normal(size=(n, 3))
    v[:6] = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    return G * v / np.linalg.norm(v, axis=1, keepdims=True)


def test_full_fit_recovers_truth():
    rng = np.random.default_rng(0)
    S = np.array([[1.03, 0.01, -0.005], [0.01, 0.97, 0.008], [-0.005, 0.008, 1.01]])
    b = np.array([0.2, -0.15, 0.3])
    f = _poses(24, rng)
    raw = f @ S.T + b + rng.normal(0, 0.005, f.shape)
    A_inv, b_est, info = fit_accel(raw)
    assert info["model"] == "full"
    assert np.allclose(b_est, b, atol=0.01)
    assert np.allclose(A_inv @ S, np.eye(3), atol=2e-3)
    assert info["rms_after_ms2"] < 0.01 < info["rms_before_ms2"]


def test_diag_fit_with_six_poses():
    rng = np.random.default_rng(1)
    S, b = np.diag([1.02, 0.98, 1.01]), np.array([0.1, 0.05, -0.2])
    raw = _poses(6, rng) @ S.T + b
    A_inv, b_est, info = fit_accel(raw)
    assert info["model"] == "diag" and np.allclose(b_est, b, atol=1e-3)
    assert np.allclose(np.diag(A_inv) * np.diag(S), 1, atol=1e-3)


def test_pose_collector_skips_repeats():
    pc = PoseCollector(hold_s=1.0)
    got = []
    for k, u in enumerate([[0, 0, 1]] * 300 + [[0, 0, 1]] * 300 + [[1, 0, 0]] * 300):
        m = pc.add(ImuSample(k * 0.005, G * np.array(u, float), np.zeros(3)))
        if m is not None:
            got.append(m)
    assert len(got) == 2


def test_mount_rotation_and_kabsch():
    R_true = rpy_to_R(*np.radians([4.0, -3.0, 30.0]))   # sensor -> car
    still = R_true.T @ np.array([0, 0, G])
    push = np.array([R_true.T @ np.array([2.0, 0.0, G])] * 20)
    R = mount_rotation(still, np.vstack([np.tile(still, (10, 1)), push]))
    assert np.allclose(R, R_true, atol=1e-6)
    rng = np.random.default_rng(2)
    src = rng.normal(size=(100, 3))
    assert np.allclose(kabsch(src, src @ R_true.T), R_true, atol=1e-9)


def test_device_offsets_are_not_subtracted_twice():
    """Once the VESC subtracts (rounded) offsets itself, apply() on its data must give the same result."""
    from racer_imu.calibration import ImuCalibration
    rng = np.random.default_rng(3)
    cal = ImuCalibration(np.eye(3) + 0.01 * rng.normal(size=(3, 3)), np.array([-0.013, 0.146, 0.079]),
                         np.radians([-0.448, 0.066, 0.469]))
    raw = ImuSample(0.0, np.array([0.1, -0.2, 9.9]), np.radians([1.0, -2.0, 3.0]))
    before = cal.apply(raw)
    cal.dev_acc_offset = np.round(cal.acc_bias / G, 3) * G           # what VESC Tool stores (3 decimals)
    cal.dev_gyro_offset = np.radians(np.round(np.degrees(cal.gyro_bias), 3))
    from_vesc = ImuSample(0.0, raw.acc - cal.dev_acc_offset, raw.gyro - cal.dev_gyro_offset)
    after = cal.apply(from_vesc)
    assert np.allclose(before.acc, after.acc, atol=1e-12) and np.allclose(before.gyro, after.gyro, atol=1e-12)


def test_device_offsets_roundtrip_file(tmp_path):
    from racer_imu.calibration import ImuCalibration
    cal = ImuCalibration(acc_bias=np.array([0.1, 0.2, 0.3]), dev_acc_offset=np.array([0.098, 0.196, 0.294]),
                         dev_gyro_offset=np.radians([0.5, 0.0, -0.5]))
    cal.save(tmp_path / "c.json")
    back = ImuCalibration.load(tmp_path / "c.json")
    assert np.allclose(back.dev_acc_offset, cal.dev_acc_offset) and np.allclose(back.dev_gyro_offset, cal.dev_gyro_offset)
