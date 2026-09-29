import numpy as np

from racer_imu.frames import CenterFuser, ImuExtrinsic, R_to_quat, R_to_rpy, quat_to_R, rpy_to_R
from racer_imu.types import G, ImuSample


def test_rotation_helpers():
    R = rpy_to_R(0.1, -0.2, 2.5)
    assert np.allclose(R_to_rpy(R), [0.1, -0.2, 2.5])
    assert np.allclose(quat_to_R(R_to_quat(R)), R)


def test_pure_rotation_gives_zero_center_accel():
    """Car spinning at constant yaw rate about its center: both IMUs feel centripetal accel, center feels 0."""
    ext = {"front": ImuExtrinsic("front", np.array([0.15, 0.02, 0.05]), rpy_to_R(0, 0, np.pi / 2)),
           "rear": ImuExtrinsic("rear", np.array([-0.12, -0.03, 0.04]), np.eye(3))}
    fuser = CenterFuser(ext)
    w = np.array([0, 0, 3.0])
    for k in range(50):
        smp = {}
        for n, e in ext.items():
            f_car = np.cross(w, np.cross(w, e.r)) + np.array([0, 0, G])
            smp[n] = ImuSample(k * 0.005, e.R.T @ f_car, e.R.T @ w, R_to_quat(e.R), n)
        c = fuser.update(k * 0.005, smp)
    assert np.allclose(c.acc, 0, atol=1e-9) and np.allclose(c.gyro, w)
    assert np.linalg.norm(c.per_imu["front"]["f_car"][:2]) > 1.0   # the IMUs themselves did feel it


def test_dual_accel_alpha():
    ext = {"a": ImuExtrinsic("a", np.array([0.15, 0, 0]), np.eye(3)),
           "b": ImuExtrinsic("b", np.array([-0.15, 0, 0]), np.eye(3))}
    fuser = CenterFuser(ext, alpha_tau=1e-6)
    alpha = np.array([0, 0, 5.0])
    for k in range(3):
        smp = {n: ImuSample(k * 0.005, np.cross(alpha, e.r) + [0, 0, G], np.zeros(3), src=n) for n, e in ext.items()}
        c = fuser.update(k * 0.005, smp)
    assert np.allclose(c.alpha, alpha, atol=1e-6) and np.allclose(c.acc, 0, atol=1e-9)
