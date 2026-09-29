"""Keyboard teleop -> VESC drive command (duty + servo), independent of the GUI toolkit.

Keys (the GUI window must have focus):
    Enter          arm            Esc        disarm (motor released)
    Up / w         forward        Down / s   reverse
    Left / a       steer left     Right / d  steer right
    Space          brake
Throttle ramps toward +-max_duty while a key is held and back to 0 when released (faster).
Holding a key makes the OS send repeated press/release pairs, so a key counts as held until
`release_grace_s` after its last press unless no new press arrives.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

FWD, REV, LEFT, RIGHT = {"up", "w"}, {"down", "s"}, {"left", "a"}, {"right", "d"}
BRAKE, ARM, DISARM = {" "}, {"enter"}, {"escape"}


@dataclass
class TeleopParams:
    max_duty: float = 0.10          # full throttle key = this duty (0.10 = 10 % of battery voltage)
    ramp_per_s: float = 0.25        # how fast duty rises [duty/s]; falls 3x faster
    brake_current: float = 5.0      # [A] while Space is held
    servo_center: float = 0.5
    servo_range: float = 0.25       # full steer = center +- range
    steer_invert: bool = False      # flip if Left steers right

    @classmethod
    def from_dict(cls, d: dict | None):
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Teleop:
    p: TeleopParams = field(default_factory=TeleopParams)
    release_grace_s: float = 0.12
    armed: bool = False
    duty: float = 0.0
    _press: dict = field(default_factory=dict)
    _release: dict = field(default_factory=dict)
    _t: float | None = None

    def press(self, key: str, now: float | None = None):
        now = time.monotonic() if now is None else now
        key = (key or "").lower()
        if key in ARM:
            self.armed = True
            self.duty = 0.0
        elif key in DISARM:
            self.armed = False
            self.duty = 0.0
        self._press[key] = now

    def release(self, key: str, now: float | None = None):
        self._release[(key or "").lower()] = time.monotonic() if now is None else now

    def held(self, keys: set, now: float) -> bool:
        for k in keys:
            tp = self._press.get(k)
            if tp is None:
                continue
            tr = self._release.get(k, -1.0)
            if tp > tr or now - tp < self.release_grace_s:
                return True
        return False

    def update(self, now: float | None = None) -> dict | None:
        """Command for VescDriver.set_drive(**cmd), or None when disarmed."""
        now = time.monotonic() if now is None else now
        dt = 0.0 if self._t is None else min(now - self._t, 0.2)
        self._t = now
        if not self.armed:
            self.duty = 0.0
            return None
        p = self.p
        steer = (1 if self.held(LEFT, now) else 0) - (1 if self.held(RIGHT, now) else 0)
        if p.steer_invert:
            steer = -steer
        servo = p.servo_center + steer * p.servo_range
        if self.held(BRAKE, now):
            self.duty = 0.0
            return {"brake": p.brake_current, "servo": servo}
        target = p.max_duty * ((1 if self.held(FWD, now) else 0) - (1 if self.held(REV, now) else 0))
        rate = p.ramp_per_s * (1 if abs(target) > abs(self.duty) else 3)
        step = rate * dt
        self.duty += max(-step, min(step, target - self.duty))
        if abs(self.duty) < 1e-4 and target == 0:
            self.duty = 0.0
            return {"current": 0.0, "servo": servo}     # coast instead of duty 0 (duty 0 brakes)
        return {"duty": self.duty, "servo": servo}

    def status(self) -> str:
        if not self.armed:
            return "DRIVE disarmed  (Enter = arm)"
        return f"DRIVE ARMED  duty {self.duty:+.3f}  (arrows/WASD, Space brake, Esc disarm)"
