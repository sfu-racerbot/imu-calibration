import numpy as np
import pytest

from racer_imu.bno_driver import LineDecoder, format_line, parse_line
from racer_imu.clock_sync import Unwrapper


def test_roundtrip():
    line = format_line(123456, [0.1, -9.8, 0.2], [0.01, 0.02, -0.03], [1, 0, 0, 0])
    d = parse_line(line)
    assert d["t_us"] == 123456 and np.allclose(d["acc"], [0.1, -9.8, 0.2])


def test_bad_checksum_rejected():
    line = format_line(1, [0, 0, 9.8], [0, 0, 0], [1, 0, 0, 0]).replace("9.8000", "9.9000")
    with pytest.raises(ValueError):
        parse_line(line)


def test_comment_and_wrap():
    assert parse_line("# hello") is None
    u = Unwrapper(32)
    assert u(0xFFFFFF00) == 0xFFFFFF00 and u(0x10) == 0x100000010


def test_decoder_host_times_monotonic():
    dec = LineDecoder()
    ts = [dec.decode(format_line(int(k * 5000), [0, 0, 9.8], [0, 0, 0], [1, 0, 0, 0]), 100 + k * 0.005 + 0.001).t
          for k in range(400)]
    assert np.all(np.diff(ts) > 0)


def test_nan_gyro_line_skipped():
    """The sketch prints nan until its first gyro report; those lines must not reach the EKF."""
    line = "$B,100,0.1,0.2,9.8,0.0,nan,0.0,1.0,0.0,0.0,0.0,3"
    from racer_imu.bno_driver import checksum
    body = line[1:]
    assert LineDecoder().decode(f"{line}*{checksum(body):02X}", 1.0) is None
