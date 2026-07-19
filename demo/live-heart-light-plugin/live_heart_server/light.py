from __future__ import annotations

import cv2
import numpy as np

from .utils import as_bool, clamp_int

def fill_light_geometry(frame_shape, light_settings: dict | None):
    height, width = frame_shape[:2]
    light_settings = light_settings or {}
    light_x = clamp_int(light_settings.get("light_x"), 0, 100, 50)
    light_y = clamp_int(light_settings.get("light_y"), 0, 100, 38)
    light_z = clamp_int(light_settings.get("light_z"), 0, 100, 45)
    light_range = clamp_int(light_settings.get("light_range"), 15, 120, 58)
    light_angle_enabled = as_bool(light_settings.get("light_angle_enabled"), False)
    light_angle = clamp_int(light_settings.get("light_angle"), -75, 75, 0)
    center_x = (light_x / 100.0) * max(1, width - 1)
    center_y = (light_y / 100.0) * max(1, height - 1)
    z_scale = 1.35 - (light_z / 100.0) * 0.75
    radius = (light_range / 100.0) * min(width, height) * 0.5 * z_scale
    intensity_scale = 1.25 - (light_z / 100.0) * 0.65
    return {
        "center_x": center_x,
        "center_y": center_y,
        "radius": max(8.0, radius),
        "light_z": light_z,
        "intensity_scale": max(0.35, intensity_scale),
        "angle_enabled": light_angle_enabled,
        "angle": light_angle,
    }


def apply_virtual_fill_light(frame_bgr: np.ndarray, light_settings: dict | None):
    if not light_settings or not light_settings.get("light_enabled"):
        return frame_bgr

    brightness = clamp_int(light_settings.get("brightness"), 20, 100, 72)
    temperature = clamp_int(light_settings.get("temperature"), 2700, 6500, 4800)
    height, width = frame_bgr.shape[:2]
    intensity = (brightness - 20) / 80.0

    warm = max(0.0, min(1.0, (4800 - temperature) / 2100.0))
    cool = max(0.0, min(1.0, (temperature - 4800) / 1700.0))
    light_color = np.array(
        [
            1.0 - warm * 0.22 + cool * 0.32,
            1.0 + warm * 0.04 - cool * 0.03,
            1.0 + warm * 0.34 - cool * 0.18,
        ],
        dtype=np.float32,
    )

    geometry = fill_light_geometry(frame_bgr.shape, light_settings)
    center_x = geometry["center_x"]
    center_y = geometry["center_y"]
    radius = geometry["radius"]
    yy, xx = np.ogrid[:height, :width]
    dx = xx - center_x
    dy = yy - center_y
    if geometry["angle_enabled"]:
        theta = np.deg2rad(geometry["angle"])
        along = dx * np.cos(theta) + dy * np.sin(theta)
        across = -dx * np.sin(theta) + dy * np.cos(theta)
        distance = np.sqrt((along / 1.35) ** 2 + (across / 0.72) ** 2)
    else:
        distance = np.sqrt(dx ** 2 + dy ** 2)
    mask = np.clip(1.0 - distance / max(1.0, radius), 0.0, 1.0)
    mask = mask * mask * (3.0 - 2.0 * mask)
    spill = np.clip(1.0 - distance / max(1.0, radius * 1.85), 0.0, 1.0) * 0.18
    light_mask = np.clip(mask + spill, 0.0, 1.0).astype(np.float32)[..., None]

    frame = frame_bgr.astype(np.float32)
    depth = geometry["intensity_scale"]
    lift = (28.0 + intensity * 92.0) * depth
    gain = 1.0 + light_mask * (0.22 + intensity * 0.72) * depth
    fill = light_mask * lift * light_color
    frame = frame * gain + fill
    return np.clip(frame, 0, 255).astype(np.uint8)


def draw_virtual_fill_light_marker(frame_bgr: np.ndarray, light_settings: dict | None):
    marked = frame_bgr.copy()
    temperature = clamp_int((light_settings or {}).get("temperature"), 2700, 6500, 4800)
    geometry = fill_light_geometry(marked.shape, light_settings)
    center_x = geometry["center_x"]
    center_y = geometry["center_y"]
    radius = geometry["radius"]
    center = (int(round(center_x)), int(round(center_y)))
    radius_px = int(round(radius))
    warm = max(0.0, min(1.0, (4800 - temperature) / 2100.0))
    cool = max(0.0, min(1.0, (temperature - 4800) / 1700.0))
    marker_color = (
        int(78 + cool * 120),
        int(198 + warm * 32),
        int(244 - cool * 46),
    )

    glow = marked.copy()
    if geometry["angle_enabled"]:
        axes = (radius_px, max(8, int(radius_px * 0.58)))
        cv2.ellipse(glow, center, axes, geometry["angle"], 0, 360, marker_color, -1, lineType=cv2.LINE_AA)
    else:
        cv2.circle(glow, center, radius_px, marker_color, -1, lineType=cv2.LINE_AA)
    marked = cv2.addWeighted(glow, 0.12, marked, 0.88, 0)
    if geometry["angle_enabled"]:
        axes = (radius_px, max(8, int(radius_px * 0.58)))
        cv2.ellipse(marked, center, axes, geometry["angle"], 0, 360, marker_color, 3, lineType=cv2.LINE_AA)
        theta = np.deg2rad(geometry["angle"])
        arrow_end = (
            int(round(center_x + np.cos(theta) * radius * 0.7)),
            int(round(center_y + np.sin(theta) * radius * 0.7)),
        )
        cv2.arrowedLine(marked, center, arrow_end, marker_color, 3, cv2.LINE_AA, 0, 0.18)
    else:
        cv2.circle(marked, center, radius_px, marker_color, 3, lineType=cv2.LINE_AA)
    cv2.circle(marked, center, 13, (255, 255, 255), 3, lineType=cv2.LINE_AA)
    cv2.circle(marked, center, 7, marker_color, -1, lineType=cv2.LINE_AA)
    cv2.putText(
        marked,
        "VIRTUAL FILL LIGHT",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (245, 248, 248),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        marked,
        f"Z {geometry['light_z']}%  ANGLE {geometry['angle'] if geometry['angle_enabled'] else 'OFF'}",
        (24, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (245, 248, 248),
        2,
        lineType=cv2.LINE_AA,
    )
    return marked
