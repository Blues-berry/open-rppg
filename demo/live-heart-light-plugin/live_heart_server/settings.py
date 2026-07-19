from __future__ import annotations

import threading

from .utils import as_bool, clamp_int

class OverlaySettings:
    def __init__(self):
        self._lock = threading.RLock()
        self.pulse = True
        self.light_enabled = False
        self.brightness = 72
        self.temperature = 4800
        self.light_x = 50
        self.light_y = 38
        self.light_z = 45
        self.light_range = 58
        self.light_angle_enabled = False
        self.light_angle = 0
        self.light_revision = 1

    def update(self, payload: dict):
        with self._lock:
            touched = False
            if "pulse" in payload:
                self.pulse = bool(payload["pulse"])
                touched = True
            if "light_enabled" in payload:
                self.light_enabled = self._as_bool(payload["light_enabled"], self.light_enabled)
                touched = True
            if "brightness" in payload:
                self.brightness = clamp_int(payload["brightness"], 20, 100, self.brightness)
                touched = True
            if "temperature" in payload:
                self.temperature = clamp_int(payload["temperature"], 2700, 6500, self.temperature)
                touched = True
            if "light_x" in payload:
                self.light_x = clamp_int(payload["light_x"], 0, 100, self.light_x)
                touched = True
            if "light_y" in payload:
                self.light_y = clamp_int(payload["light_y"], 0, 100, self.light_y)
                touched = True
            if "light_z" in payload:
                self.light_z = clamp_int(payload["light_z"], 0, 100, self.light_z)
                touched = True
            if "light_range" in payload:
                self.light_range = clamp_int(payload["light_range"], 15, 120, self.light_range)
                touched = True
            if "light_angle_enabled" in payload:
                self.light_angle_enabled = as_bool(payload["light_angle_enabled"], self.light_angle_enabled)
                touched = True
            if "light_angle" in payload:
                self.light_angle = clamp_int(payload["light_angle"], -75, 75, self.light_angle)
                touched = True
            if touched:
                self.light_revision += 1
            return self.status()

    def status(self):
        with self._lock:
            return {
                "pulse": self.pulse,
                "light_enabled": self.light_enabled,
                "brightness": self.brightness,
                "temperature": self.temperature,
                "light_x": self.light_x,
                "light_y": self.light_y,
                "light_z": self.light_z,
                "light_range": self.light_range,
                "light_angle_enabled": self.light_angle_enabled,
                "light_angle": self.light_angle,
                "light_revision": self.light_revision,
            }

    @staticmethod
    def _as_bool(value, default):
        return as_bool(value, default)


