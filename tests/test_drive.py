"""Drive commands reach the (simulated) VESC, and the dead-man switch releases the motor."""
import time

from racer_imu import vesc_protocol as vp
from racer_imu.frames import load_config
from racer_imu.sim import SimServer
from racer_imu.vesc_driver import VescDriver


def test_drive_commands_and_deadman():
    sim = SimServer(load_config(), "still").start()
    drv = VescDriver(sim.ports["vesc"])
    drv.start()
    try:
        t_end = time.monotonic() + 0.5
        while time.monotonic() < t_end:          # GUI-like refresh at 20 Hz
            drv.set_drive(duty=0.1, servo=0.6)
            time.sleep(0.05)
        assert abs(sim.last_drive[vp.COMM_SET_DUTY] - 0.1) < 1e-6
        assert abs(sim.last_drive[vp.COMM_SET_SERVO_POS] - 0.6) < 1e-6
        time.sleep(0.6)                           # stop refreshing -> expires after 0.3 s
        assert sim.last_drive["last_cmd"] == vp.COMM_SET_CURRENT
        assert sim.last_drive[vp.COMM_SET_CURRENT] == 0.0
        assert drv.count > 150                    # IMU polling kept going while driving
    finally:
        drv.stop()
        drv.join(1)
        sim.stop()


def test_teleop_ramp_autorepeat_and_disarm():
    from racer_imu.teleop import Teleop, TeleopParams
    tp = Teleop(TeleopParams(max_duty=0.1, ramp_per_s=0.5))
    assert tp.update(0.0) is None                     # disarmed by default
    tp.press("enter", 0.0)
    tp.press("up", 0.0)
    t = 0.0
    while t < 0.5:                                    # autorepeat: release+press pairs every 30 ms
        t += 0.03
        tp.release("up", t)
        tp.press("up", t + 0.001)
        cmd = tp.update(t + 0.01)
    assert abs(cmd["duty"] - 0.1) < 1e-9              # ramped all the way up, never dropped out
    tp.release("up", t + 0.02)
    for k in range(20):
        cmd = tp.update(t + 0.2 + k * 0.05)
    assert cmd.get("current") == 0.0 and "duty" not in cmd   # back to coasting after release
    tp.press("escape", t + 2)
    assert tp.update(t + 2.1) is None
