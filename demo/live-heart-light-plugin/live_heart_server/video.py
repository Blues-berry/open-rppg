from __future__ import annotations

import tempfile
import threading
import time
import uuid
from pathlib import Path

import cv2

from .config import HR_WINDOW_SECONDS, UPLOAD_CHUNK_BYTES, VIDEO_EXTENSIONS
from .utils import safe_float
import rppg

class VideoAnalysisWorker:
    def __init__(self, model_name: str = "FacePhys.rlap"):
        self.model_name = model_name
        self._lock = threading.RLock()
        self._thread = None
        self.state = "idle"
        self.error = None
        self.filename = None
        self.started_at = None
        self.completed_at = None
        self.frames_total = None
        self.frames_processed = 0
        self.duration = None
        self.fps = None
        self.width = None
        self.height = None
        self.result = None
        self.analysis_ms = None
        self.no_face_count = 0
        self.signal_frames = 0
        self.temp_path = None

    def start_from_upload(self, stream, length: int, filename: str):
        with self._lock:
            if self.state in {"saving", "queued", "processing"}:
                return self.status(), 409
            self._clear_locked()
            self.state = "saving"
            self.filename = Path(filename or "upload.mp4").name or "upload.mp4"
            self.started_at = time.time()

        try:
            path = self._save_stream(stream, length, self.filename)
        except Exception as exc:
            with self._lock:
                self.state = "failed"
                self.error = f"{type(exc).__name__}: {exc}"
                self.completed_at = time.time()
            return self.status(), 400

        with self._lock:
            self.temp_path = path
            self.state = "queued"
            self._thread = threading.Thread(target=self._run, args=(path,), daemon=True)
            self._thread.start()
            return self.status(), 202

    def reset(self):
        with self._lock:
            if self.state in {"saving", "queued", "processing"}:
                return self.status(), 409
            self._clear_locked()
            return self.status(), 200

    def status(self):
        with self._lock:
            progress = None
            if self.frames_total and self.frames_total > 0:
                progress = min(1.0, self.frames_processed / self.frames_total)
            return {
                "state": self.state,
                "error": self.error,
                "filename": self.filename,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "frames_total": self.frames_total,
                "frames_processed": self.frames_processed,
                "progress": safe_float(progress),
                "duration": safe_float(self.duration),
                "fps": safe_float(self.fps),
                "width": self.width,
                "height": self.height,
                "analysis_ms": safe_float(self.analysis_ms),
                "no_face_count": self.no_face_count,
                "signal_frames": self.signal_frames,
                "result": self.result,
            }

    def _clear_locked(self):
        self.state = "idle"
        self.error = None
        self.filename = None
        self.started_at = None
        self.completed_at = None
        self.frames_total = None
        self.frames_processed = 0
        self.duration = None
        self.fps = None
        self.width = None
        self.height = None
        self.result = None
        self.analysis_ms = None
        self.no_face_count = 0
        self.signal_frames = 0
        self.temp_path = None

    def _save_stream(self, stream, length: int, filename: str):
        suffix = Path(filename).suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            suffix = ".mp4"
        temp_path = Path(tempfile.gettempdir()) / f"live-heart-upload-{uuid.uuid4().hex}{suffix}"
        remaining = length
        with temp_path.open("wb") as handle:
            while remaining > 0:
                chunk = stream.read(min(UPLOAD_CHUNK_BYTES, remaining))
                if not chunk:
                    raise ValueError("upload ended before Content-Length was reached")
                handle.write(chunk)
                remaining -= len(chunk)
        return temp_path

    def _run(self, path: Path):
        started = time.perf_counter()
        cap = None
        model = None
        try:
            with self._lock:
                self.state = "processing"

            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise ValueError("Cannot open uploaded video")

            fps = safe_float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
            frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
            duration = frames_total / fps if frames_total and fps else None
            with self._lock:
                self.fps = fps
                self.frames_total = frames_total
                self.width = width
                self.height = height
                self.duration = duration

            model = rppg.Model(self.model_name)
            model.face_detection_threads = 1
            model.face_resampling_threads = 1
            model.face_detect_per_n = 3
            first_ts = None
            frame_index = 0

            with model:
                while True:
                    ok, bgr = cap.read()
                    if not ok or bgr is None:
                        break
                    pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                    ts = pos_msec / 1000 if pos_msec and pos_msec > 0 else frame_index / fps
                    if first_ts is None:
                        first_ts = ts
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    model.update_frame(rgb, ts - first_ts)
                    frame_index += 1
                    if frame_index % 10 == 0:
                        self._mark_progress(frame_index, model)

            self._mark_progress(frame_index, model)
            result = model.hr(start=-HR_WINDOW_SECONDS, return_hrv=False) or model.hr(return_hrv=False)
            statistic = getattr(model, "statistic", {}) or {}
            with self._lock:
                self.frames_processed = frame_index
                self.signal_frames = int(getattr(model, "n_signal", 0) or 0)
                self.no_face_count = int(statistic.get("null", 0) or 0)
                self.result = {
                    "hr": safe_float(result.get("hr") if isinstance(result, dict) else None),
                    "SQI": safe_float(result.get("SQI") if isinstance(result, dict) else None),
                    "latency": safe_float(result.get("latency") if isinstance(result, dict) else None),
                }
                self.analysis_ms = (time.perf_counter() - started) * 1000
                self.completed_at = time.time()
                self.state = "done"
        except Exception as exc:
            with self._lock:
                self.error = f"{type(exc).__name__}: {exc}"
                self.analysis_ms = (time.perf_counter() - started) * 1000
                self.completed_at = time.time()
                self.state = "failed"
        finally:
            if cap is not None:
                cap.release()
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    def _mark_progress(self, frame_index: int, model):
        statistic = getattr(model, "statistic", {}) or {}
        with self._lock:
            self.frames_processed = frame_index
            self.signal_frames = int(getattr(model, "n_signal", 0) or 0)
            self.no_face_count = int(statistic.get("null", 0) or 0)


