"""VESC serial protocol, mirrored from vesc_tool (packet.cpp, commands.cpp, vbytearray.cpp).

Frame:  0x02 | len(1)            | payload | crc16 hi | crc16 lo | 0x03   (payload <= 255 B)
        0x03 | len(2, big endian)| payload | crc16 hi | crc16 lo | 0x03   (payload <= 65535 B)
        0x04 | len(3, big endian)| ...
The CRC is CRC16-XMODEM (poly 0x1021, init 0) over the payload only. payload[0] is the command id.
"""
from __future__ import annotations

import struct

COMM_SET_DUTY = 5
COMM_SET_CURRENT = 6
COMM_SET_CURRENT_BRAKE = 7
COMM_SET_RPM = 8
COMM_SET_SERVO_POS = 12
COMM_FORWARD_CAN = 34
COMM_GET_VALUES_SELECTIVE = 50
COMM_GET_IMU_DATA = 65

IMU_FIELDS = ("roll", "pitch", "yaw",
              "acc_x", "acc_y", "acc_z",
              "gyro_x", "gyro_y", "gyro_z",
              "mag_x", "mag_y", "mag_z",
              "q0", "q1", "q2", "q3")
IMU_MASK_ALL = 0xFFFF

# COMM_GET_VALUES_SELECTIVE fields we care about: bit -> (name, struct fmt, scale)
VALUES_FIELDS = {
    0: ("temp_mos", ">h", 1e1),
    1: ("temp_motor", ">h", 1e1),
    2: ("current_motor", ">i", 1e2),
    3: ("current_in", ">i", 1e2),
    4: ("id", ">i", 1e2),
    5: ("iq", ">i", 1e2),
    6: ("duty_now", ">h", 1e3),
    7: ("rpm", ">i", 1e0),
    8: ("v_in", ">h", 1e1),
    9: ("amp_hours", ">i", 1e4),
    10: ("amp_hours_charged", ">i", 1e4),
    11: ("watt_hours", ">i", 1e4),
    12: ("watt_hours_charged", ">i", 1e4),
    13: ("tachometer", ">i", 1e0),
    14: ("tachometer_abs", ">i", 1e0),
    15: ("fault_code", ">b", 1e0),
}
VALUES_MASK_WHEEL = (1 << 7) | (1 << 8) | (1 << 13)  # rpm, v_in, tachometer


def _crc_table():
    table = []
    for i in range(256):
        c = i << 8
        for _ in range(8):
            c = ((c << 1) ^ 0x1021) if c & 0x8000 else (c << 1)
        table.append(c & 0xFFFF)
    return table


_CRC_TABLE = _crc_table()


def crc16(data: bytes) -> int:
    c = 0
    for b in data:
        c = (_CRC_TABLE[((c >> 8) ^ b) & 0xFF] ^ (c << 8)) & 0xFFFF
    return c


def pack_frame(payload: bytes) -> bytes:
    n = len(payload)
    if n == 0 or n > 0xFFFFFF:
        raise ValueError("bad payload length")
    if n <= 0xFF:
        head = bytes([2, n])
    elif n <= 0xFFFF:
        head = bytes([3, n >> 8, n & 0xFF])
    else:
        head = bytes([4, n >> 16, (n >> 8) & 0xFF, n & 0xFF])
    return head + payload + struct.pack(">H", crc16(payload)) + b"\x03"


class FrameParser:
    """Incremental decoder: feed() raw bytes, get back complete, CRC-checked payloads."""

    MAX_LEN = 512   # our replies are < 100 B; a small limit makes false headers rarer

    def __init__(self):
        self.buf = bytearray()
        self.crc_errors = 0

    def _try(self, i: int):
        """Try to decode a frame starting at buf[i]. Returns ('ok', payload, total) | ('bad',) | ('wait',)."""
        b = self.buf
        if b[i] not in (2, 3, 4):
            return ("bad",)
        nlen = b[i] - 1                        # 2 -> 1 length byte, 3 -> 2, 4 -> 3
        if len(b) - i < 1 + nlen:
            return ("wait",)
        n = int.from_bytes(b[i + 1:i + 1 + nlen], "big")
        if n == 0 or n > self.MAX_LEN:
            return ("bad",)
        total = 1 + nlen + n + 3
        if len(b) - i < total:
            return ("wait",)
        payload = bytes(b[i + 1 + nlen:i + 1 + nlen + n])
        crc = int.from_bytes(b[i + 1 + nlen + n:i + total - 1], "big")
        if b[i + total - 1] != 3 or crc != crc16(payload):
            return ("bad",)
        return ("ok", payload, total)

    def feed(self, data: bytes) -> list[bytes]:
        self.buf += data
        out = []
        while self.buf:
            r = self._try(0)
            if r[0] == "ok":
                out.append(r[1])
                del self.buf[:r[2]]
            elif r[0] == "bad":
                if self.buf[0] in (2, 3, 4):
                    self.crc_errors += 1
                del self.buf[0]                 # resync one byte later
            else:
                # Waiting for more bytes. If the "header" was really a corrupted byte, a complete valid frame
                # may already sit later in the buffer: jump to it instead of waiting forever.
                for i in range(1, len(self.buf)):
                    if self.buf[i] in (2, 3, 4) and self._try(i)[0] == "ok":
                        self.crc_errors += 1
                        del self.buf[:i]
                        break
                else:
                    break
        return out


# "Auto" float32 (vbAppendDouble32Auto). Same bit layout as IEEE-754 single for normal numbers;
# subnormals are flushed to zero by the firmware, which struct handles identically in practice.
def encode_float32_auto(x: float) -> bytes:
    if abs(x) < 1.5e-38:
        x = 0.0
    return struct.pack(">f", x)


def decode_float32_auto(b: bytes) -> float:
    return struct.unpack(">f", b)[0]


def build_get_imu(mask: int = IMU_MASK_ALL) -> bytes:
    return pack_frame(struct.pack(">BH", COMM_GET_IMU_DATA, mask))


def build_get_values_selective(mask: int = VALUES_MASK_WHEEL) -> bytes:
    return pack_frame(struct.pack(">BI", COMM_GET_VALUES_SELECTIVE, mask))


# ---- drive commands (no reply). Scaling as in vesc_tool Commands::setDutyCycle/setCurrent/... ----
def build_set_duty(duty: float) -> bytes:
    """Duty cycle -1..1 (fraction of battery voltage). Sign = direction."""
    return pack_frame(struct.pack(">Bi", COMM_SET_DUTY, int(round(duty * 1e5))))


def build_set_current(amps: float) -> bytes:
    """Motor current [A]. 0 releases the motor (coast)."""
    return pack_frame(struct.pack(">Bi", COMM_SET_CURRENT, int(round(amps * 1e3))))


def build_set_current_brake(amps: float) -> bytes:
    return pack_frame(struct.pack(">Bi", COMM_SET_CURRENT_BRAKE, int(round(amps * 1e3))))


def build_set_rpm(erpm: float) -> bytes:
    return pack_frame(struct.pack(">Bi", COMM_SET_RPM, int(round(erpm))))


def build_set_servo_pos(pos: float) -> bytes:
    """Servo output 0..1 (0.5 = centre). Needs 'Enable Servo Output' in the VESC app config."""
    pos = min(max(pos, 0.0), 1.0)
    return pack_frame(struct.pack(">Bh", COMM_SET_SERVO_POS, int(round(pos * 1e3))))


def forward_can(can_id: int, frame: bytes) -> bytes:
    """Wrap a request for a VESC behind CAN (frame = output of build_*)."""
    payload = _unframe(frame)
    return pack_frame(bytes([COMM_FORWARD_CAN, can_id]) + payload)


def _unframe(frame: bytes) -> bytes:
    p = FrameParser().feed(frame)
    if len(p) != 1:
        raise ValueError("not a single frame")
    return p[0]


def parse_imu(payload: bytes) -> dict:
    """Parse a COMM_GET_IMU_DATA reply payload. Units as sent: rad, g, deg/s."""
    if payload[0] != COMM_GET_IMU_DATA:
        raise ValueError("not an IMU reply")
    mask = struct.unpack_from(">H", payload, 1)[0]
    pos = 3
    out = {"mask": mask}
    for bit, name in enumerate(IMU_FIELDS):
        if mask & (1 << bit):
            out[name] = decode_float32_auto(payload[pos:pos + 4])
            pos += 4
    if len(payload) > pos:
        out["vesc_id"] = payload[pos]
    return out


def encode_imu_reply(values: dict, mask: int = IMU_MASK_ALL, vesc_id: int | None = 0) -> bytes:
    """Build the reply payload the firmware would send (used by the simulator and tests)."""
    b = bytearray(struct.pack(">BH", COMM_GET_IMU_DATA, mask))
    for bit, name in enumerate(IMU_FIELDS):
        if mask & (1 << bit):
            b += encode_float32_auto(values.get(name, 0.0))
    if vesc_id is not None:
        b.append(vesc_id)
    return bytes(b)


def parse_values_selective(payload: bytes) -> dict:
    if payload[0] != COMM_GET_VALUES_SELECTIVE:
        raise ValueError("not a GET_VALUES_SELECTIVE reply")
    mask = struct.unpack_from(">I", payload, 1)[0]
    pos = 5
    out = {"mask": mask}
    for bit in range(16):
        if mask & (1 << bit):
            name, fmt, scale = VALUES_FIELDS[bit]
            v = struct.unpack_from(fmt, payload, pos)[0]
            pos += struct.calcsize(fmt)
            out[name] = v / scale
    return out


def encode_values_selective_reply(values: dict, mask: int = VALUES_MASK_WHEEL) -> bytes:
    b = bytearray(struct.pack(">BI", COMM_GET_VALUES_SELECTIVE, mask))
    for bit in range(16):
        if mask & (1 << bit):
            name, fmt, scale = VALUES_FIELDS[bit]
            b += struct.pack(fmt, int(round(values.get(name, 0.0) * scale)))
    return bytes(b)
