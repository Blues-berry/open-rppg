from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .config import RECORDING_EXPORTS_DIR, RECORDINGS_DIR
from .utils import safe_float

class RecordingManager:
    def __init__(self):
        self._lock = threading.RLock()
        self.enabled = False
        self.state = "idle"
        self.error = None
        self.session_id = None
        self.capture_started_at = None
        self.started_at = None
        self.started_elapsed = None
        self.duration = 0.0
        self.frame_count = 0
        self.fps = None
        self.size = None
        self.path = None
        self.sidecar_path = None
        self._writer = None
        self._recording_index = 0
        self.export_state = "idle"
        self.export_error = None
        self.exports = {}

    def begin_capture_session(self, session_id: str, started_at: float | None):
        with self._lock:
            self._close_writer_locked()
            self.session_id = session_id
            self.capture_started_at = started_at
            self.started_at = None
            self.started_elapsed = None
            self.duration = 0.0
            self.frame_count = 0
            self.fps = None
            self.size = None
            self.path = None
            self.sidecar_path = None
            self.error = None
            self.state = "idle"
            self.export_state = "idle"
            self.export_error = None
            self.exports = {}
            self._recording_index = 0

    def set_enabled(self, enabled: bool):
        with self._lock:
            self.enabled = bool(enabled)
            if not self.enabled:
                self._close_writer_locked()
            elif self.state == "error":
                self.error = None
                self.state = "idle"
            return self.status()

    def wants_frame(self):
        with self._lock:
            return bool(self.enabled and self.session_id and self.state != "error")

    def write_frame(self, frame_bgr: np.ndarray, session_elapsed: float, fps):
        with self._lock:
            if not self.enabled or not self.session_id:
                return
            if self.state == "error":
                return
            if self._writer is None:
                try:
                    self._open_writer_locked(frame_bgr, session_elapsed, fps)
                except Exception as exc:
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.state = "error"
                    return
            try:
                self._writer.write(frame_bgr)
                self.frame_count += 1
                if self.started_elapsed is not None:
                    self.duration = max(0.0, float(session_elapsed) - self.started_elapsed)
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                self.state = "error"
                self._close_writer_locked()

    def end_capture_session(self, highlight_snapshot: dict | None = None):
        with self._lock:
            self._close_writer_locked()
            self._write_sidecar_locked(highlight_snapshot or {})
            return self.status()

    def status(self):
        with self._lock:
            file_path = str(self.path) if self.path else None
            return {
                "enabled": self.enabled,
                "state": self.state,
                "error": self.error,
                "session_id": self.session_id,
                "file": file_path,
                "sidecar": str(self.sidecar_path) if self.sidecar_path else None,
                "started_at": self.started_at,
                "started_elapsed": safe_float(self.started_elapsed),
                "duration": safe_float(self.duration),
                "frames": self.frame_count,
                "fps": safe_float(self.fps),
                "size": self.size,
                "exportable": self._has_finalized_video_locked(),
            }

    def export_status(self):
        with self._lock:
            return {"state": self.export_state, "error": self.export_error}

    def export_info(self, highlight_id: str):
        with self._lock:
            return dict(self.exports.get(highlight_id) or {})

    def can_export(self, highlight: dict):
        with self._lock:
            if highlight.get("status") != "confirmed":
                return False
            if not self._has_finalized_video_locked():
                return False
            bounds = self._clip_bounds_locked(highlight)
            return bounds is not None and bounds[1] > 0.2

    def export_highlight(self, highlight: dict | None):
        if not highlight:
            return {"state": "failed", "error": "highlight not found"}, 404
        if highlight.get("status") != "confirmed":
            return {"state": "failed", "error": "highlight is still observing"}, 409
        highlight_id = str(highlight.get("id") or "")
        with self._lock:
            if not self._has_finalized_video_locked():
                return {"state": "failed", "error": "stop or disable recording before exporting"}, 409
            bounds = self._clip_bounds_locked(highlight)
            if bounds is None:
                return {"state": "failed", "error": "highlight is outside the recorded range"}, 409
            input_start, duration = bounds
            input_path = self.path
            session_id = self.session_id or "session"
            export_id = f"{session_id}-{highlight_id}"
            output_path = RECORDING_EXPORTS_DIR / f"{export_id}.mp4"
            self.export_state = "exporting"
            self.export_error = None

        try:
            self._export_clip(input_path, output_path, input_start, duration)
            body = {
                "state": "done",
                "highlight_id": highlight_id,
                "export_id": export_id,
                "path": str(output_path),
                "download_url": f"/api/highlights/download?id={export_id}",
                "start": safe_float(highlight.get("start")),
                "end": safe_float(highlight.get("end")),
            }
            with self._lock:
                self.export_state = "done"
                self.exports[highlight_id] = body
            return body, 200
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self.export_state = "failed"
                self.export_error = message
            return {"state": "failed", "error": message}, 500

    def download_path(self, export_id: str):
        with self._lock:
            for item in self.exports.values():
                if item.get("export_id") == export_id:
                    path = Path(item.get("path") or "")
                    if path.exists() and path.is_file():
                        return path
            return None

    def _open_writer_locked(self, frame_bgr: np.ndarray, session_elapsed: float, fps):
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        height, width = frame_bgr.shape[:2]
        actual_fps = safe_float(fps) or 30.0
        actual_fps = min(120.0, max(1.0, actual_fps))
        self._recording_index += 1
        suffix = "" if self._recording_index == 1 else f"-part{self._recording_index}"
        path = RECORDINGS_DIR / f"{self.session_id}{suffix}.mp4"
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            actual_fps,
            (int(width), int(height)),
        )
        if not writer.isOpened():
            raise ValueError(f"Cannot open recording writer: {path}")
        self._writer = writer
        self.state = "recording"
        self.error = None
        self.started_at = time.time()
        self.started_elapsed = float(session_elapsed)
        self.duration = 0.0
        self.frame_count = 0
        self.fps = actual_fps
        self.size = {"width": int(width), "height": int(height)}
        self.path = path
        self.sidecar_path = path.with_suffix(".json")

    def _close_writer_locked(self):
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass
            self._writer = None
        if self.state == "recording":
            self.state = "finalized"

    def _write_sidecar_locked(self, highlight_snapshot: dict):
        if not self.path or not self.sidecar_path:
            return
        try:
            payload = {
                "session_id": self.session_id,
                "video_path": str(self.path),
                "recording": self.status(),
                "samples": highlight_snapshot.get("samples", []),
                "highlights": highlight_snapshot.get("items", []),
                "written_at": time.time(),
            }
            with self.sidecar_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _has_finalized_video_locked(self):
        return self.path is not None and self.path.exists() and self.path.is_file() and self.state == "finalized"

    def _clip_bounds_locked(self, highlight: dict):
        if self.started_elapsed is None or self.duration <= 0:
            return None
        start = safe_float(highlight.get("start"))
        end = safe_float(highlight.get("end"))
        if start is None or end is None or end <= start:
            return None
        recording_start = self.started_elapsed
        recording_end = self.started_elapsed + self.duration
        clip_start = max(recording_start, start)
        clip_end = min(recording_end, end)
        duration = clip_end - clip_start
        if duration <= 0:
            return None
        return clip_start - recording_start, duration

    @staticmethod
    def _export_clip(input_path: Path, output_path: Path, input_start: float, duration: float):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            command = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{input_start:.3f}",
                "-i",
                str(input_path),
                "-t",
                f"{duration:.3f}",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ]
            try:
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                if output_path.exists() and output_path.stat().st_size > 0:
                    return
            except Exception:
                pass
        RecordingManager._export_clip_opencv(input_path, output_path, input_start, duration)

    @staticmethod
    def _export_clip_opencv(input_path: Path, output_path: Path, input_start: float, duration: float):
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open recording: {input_path}")
        fps = safe_float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            cap.release()
            raise ValueError("recording has invalid dimensions")
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise ValueError(f"Cannot open export writer: {output_path}")
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, input_start) * 1000.0)
        frames_to_write = max(1, int(round(duration * fps)))
        written = 0
        try:
            while written < frames_to_write:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                writer.write(frame)
                written += 1
        finally:
            cap.release()
            writer.release()
        if written == 0:
            raise ValueError("export did not contain any frames")


