import numpy as np

from racer_imu import vesc_protocol as vp


def test_request_bytes_match_vesc_tool():
    # 02 | len 3 | 41 FF FF | crc 37 92 | 03  (checked by hand against packet.cpp)
    assert vp.build_get_imu() == bytes.fromhex("020341ffff379203")


def test_imu_roundtrip_and_resync():
    vals = {n: float(i) * 0.37 - 2.0 for i, n in enumerate(vp.IMU_FIELDS)}
    frame = vp.pack_frame(vp.encode_imu_reply(vals, vesc_id=7))
    assert frame[1] == 68 and len(frame) == 73
    p = vp.FrameParser()
    # garbage before, split across reads, a corrupted copy in the middle
    bad = bytearray(frame)
    bad[10] ^= 0xFF
    payloads = p.feed(b"\x00\x99" + frame[:20]) + p.feed(frame[20:] + bytes(bad) + frame)
    assert len(payloads) == 2 and p.crc_errors >= 1
    d = vp.parse_imu(payloads[0])
    assert d["vesc_id"] == 7
    for n in vp.IMU_FIELDS:
        assert np.isclose(d[n], vals[n], rtol=1e-6)


def test_long_frame_header():
    payload = bytes(range(256)) * 2
    assert vp.FrameParser().feed(vp.pack_frame(payload)) == [payload]


def test_values_selective_roundtrip():
    frame = vp.pack_frame(vp.encode_values_selective_reply({"rpm": -12345.0, "v_in": 16.4, "tachometer": 99}))
    d = vp.parse_values_selective(vp.FrameParser().feed(frame)[0])
    assert d["rpm"] == -12345 and np.isclose(d["v_in"], 16.4) and d["tachometer"] == 99


def test_forward_can_wraps_payload():
    f = vp.forward_can(5, vp.build_get_imu())
    assert vp.FrameParser().feed(f)[0] == bytes([vp.COMM_FORWARD_CAN, 5, 0x41, 0xFF, 0xFF])
