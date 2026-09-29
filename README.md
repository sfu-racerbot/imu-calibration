# racer_imu: VESC IMU + BNO085, calibrated, time-synced, in the car frame

Two IMUs on the car:
- **VESC built-in IMU**, read over USB with the VESC binary protocol.
- **BNO085**, read through a microcontroller running `firmware/bno085_stream`.

This stack calibrates both, puts them on one clock, moves their readings to the car center, and estimates
path, heading and drift (sideslip β). All of it runs against a built-in simulator, so it can be tested without hardware.

```
VESC ──USB──► VescDriver ─┐                                      ┌─► CenterFuser ─► EKF ─► car_view GUI
                          ├─► calibration ─► Synchronizer (200 Hz grid)
MCU+BNO085 ─USB─► BnoDriver ┘   (calib/*.json)  (calib/sync.json)   └─► logs/<run>/*.csv
```

## Quick start (no hardware)
```bash
pip install -r requirements.txt           # everything except pytest is usually already there
python -m pytest -q tests                 # protocol, calibration, sync, frames, EKF
python tools/car_view.py --sim drift      # GUI against the simulator (ground truth dashed)
python tools/imu_monitor.py --sim drift   # raw numbers
```
In `--sim` mode every tool reads and writes `calib/sim/`, so real calibration files are never touched.
The full calibration flow also runs in sim mode:
`calibrate_imu.py --imu vesc --sim`, `--imu bno --sim`, `sync_calibrate.py --sim`.

## At the lab
1. **Serial access.** `ls /dev/ttyACM*`. If you get "permission denied", run `sudo usermod -aG dialout $USER` and log in again.
   Put the ports in `config/car.yaml`. Close VESC Tool, since only one program can hold the port.
2. **VESC Tool.** Set the IMU filter the way you want it. Set **accel/gyro offsets to 0**: the VESC applies
   them before sending, and our calibration replaces them. Note the `rot_*` values you use.
3. **Flash the BNO085 sketch.** Install the "Adafruit BNO08x" library, then upload `firmware/bno085_stream`. On boot the
   serial monitor should show `# bno085_stream ready` followed by `$B,...` lines.
4. **`python tools/imu_monitor.py`.** Both rates should be about 200 Hz. Check the axis signs: tilt nose-down and roll left, and watch which axis changes.
5. **`python tools/calibrate_imu.py --imu vesc`** and **`--imu bno`**. Keep it still for 10 s, then hold 12+
   orientations (the 6 faces, then tilted ones); each pose is captured automatically. Check the result with
   `tools/plot_calibration.py` (per-pose plots) and `tools/calib_live.py` (live raw vs calibrated). Naming only one
   port (e.g. `--vesc /dev/ttyACM0`) makes any tool use just that IMU.
6. **Tape-measure** each IMU chip's position from the car center (x forward, y left, z up). Enter it in `car.yaml`.
7. **`python tools/calibrate_imu.py --imu vesc --step mount`** (and for `bno`). Put the car on level ground, keep it still for about 3 s,
   then push it straight forward. Copy the printed `rpy_deg` into `car.yaml`.
8. **`python tools/sync_calibrate.py`.** Shake and twist the car for 15 s. This writes the time offset to `calib/sync.json` and checks
   the two mounting rotations against each other.
9. **`python tools/car_view.py --log`.** Drive. Replay later with `--replay logs/<run>`.

## How the pieces work
| File | What it does |
|---|---|
| `racer_imu/vesc_protocol.py` | Frame format `02 len payload crc16 03`, `COMM_GET_IMU_DATA` (65), `COMM_GET_VALUES_SELECTIVE` (50), CAN forwarding. Checked against `../vsec/vesc_tool` source |
| `racer_imu/vesc_driver.py` | Polls the IMU at 200 Hz and ERPM every 4th poll. Each sample is timestamped at the midpoint of request and reply; replies with a slow round trip are dropped |
| `racer_imu/bno_driver.py` | Parses and checksums lines. Maps the sensor's µs clock to host time (`ClockMapper`: lower-envelope fit that removes USB jitter and tracks the MCU clock's drift) |
| `racer_imu/calibration.py` | Ellipsoid fit (same model as Magneto, which michaelwro/accelerometer-calibration uses) → `A_inv`, bias; gyro bias; mounting rotation from gravity plus a push; Kabsch alignment |
| `racer_imu/sync.py` | Applies per-stream offsets, then interpolates every IMU onto one 200 Hz grid (slerp for quaternions) |
| `racer_imu/frames.py` | `car.yaml` loader. Center-frame math: `f_c = R f − α×r − ω×(ω×r)`, with α from the front−back accelerometer difference |
| `racer_imu/estimator.py` | EKF with state `[x, y, ψ, vx, vy, r]`; wheel speed, gyro, a weak no-slip update, and standstill updates |
| `racer_imu/procs.py` | Drivers and simulator in child processes, so GUI redraws can't stall the drivers or distort timestamps |
| `racer_imu/sim.py` | Scenarios (still, push, circle, slalom, drift, shake, poses). Fakes the VESC and MCU on `/dev/pts/*` with known errors |

**VESC IMU data has no device timestamp**, so the VESC is the time reference (offset 0). Other streams are shifted to match it,
using a cross-correlation of |ω|. |ω| is the same everywhere on a rigid car, whatever way each IMU is mounted.

## Known limits (measured in `tests/test_estimator.py`)
- Circle and slalom driving: β error about 1° RMS.
- A 35%-wheel-spin power slide gives about **5° RMS β error**. The wheels over-read forward speed, and nothing else measures sideways velocity.
  `PlanarEkf.update_velocity()` takes an external car-frame velocity, such as F1TENTH lidar odometry. With it at 20 Hz, the error drops to **1.8°**.
- The BNO085 sketch sends the latest gyro and quaternion with each accel report, so they can be up to one report period (5 ms) old.
- The sketch has only been checked on the host: its line formatter was compiled with g++ and its output parsed. It has **not been compiled for a board** (no arduino-cli here).

## Online calibration while driving (no timers)
The gyro bias creeps with temperature, so the EKF keeps estimating it as a state. It gets corrected
**whenever there is evidence**, never on a clock:
- **Every stop:** `StandstillDetector` (wheels ~0 and steady IMU for 0.5 s) tells the EKF the car isn't
  rotating, which re-zeroes the gyro.
- **Every lidar reading:** `pipe.lidar_pose(t_scan, x, y, psi)`, `lidar_heading`, `lidar_velocity` and
  `lidar_yaw_rate`. Each is fed in gradually (no jumps for the controller) and checked for outliers.
  Late results are moved forward to "now" using the car's own motion since the scan. The first pose
  sets the map frame. A lidar heading keeps the gyro calibrated even without stopping.
- **Health monitor** (reports only): warns about a large bias change, a parked |a| ≠ g (accel
  calibration stale), or a long time without any heading reference.
- The learned bias is saved to `calib/online_gyro_bias.json` on exit and used as the start value next
  time. It's ignored automatically after a recalibration.
- The full accelerometer calibration still needs `calibrate_imu.py`, because it needs the car held in
  many orientations.

Measured in `tests/test_online.py`, with the gyro bias creeping 1.2 °/s per minute:

| | heading error |
|---|---|
| stop-and-go, fixed calibration | 30.9° after 62 s |
| stop-and-go, online | 1.6° |
| circling, never stopping, no lidar | 14.9° |
| circling, lidar heading at 10 Hz | 0.4° |
