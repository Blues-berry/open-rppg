from __future__ import annotations

import json
import time
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import (
    FACE_HEIGHT,
    FACE_WIDTH,
    MAX_UPLOAD_BYTES,
    MJPEG_BOUNDARY,
    MJPEG_STREAM_FPS,
    PLUGIN_DIR,
)
from .utils import as_bool

class PluginHandler(SimpleHTTPRequestHandler):
    app = None
    server_version = "LiveHeartOpenRppg/0.2"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PLUGIN_DIR), **kwargs)

    def end_headers(self):
        # This console changes in tandem with its ES modules. Never allow a
        # browser to combine a stale module with a newer index.html.
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/model/status":
            self._send_json(self.app.runtime.status())
            return
        if parsed.path == "/api/model/start":
            self.app.runtime.start_async()
            self._send_json(self.app.runtime.status())
            return
        if parsed.path == "/api/capture/status":
            self._send_json(self.app.capture.status())
            return
        if parsed.path == "/api/capture/devices":
            self._send_json(self.app.list_devices())
            return
        if parsed.path == "/api/overlay/state":
            self._send_json(self.app.overlay_state())
            return
        if parsed.path == "/api/overlay/settings":
            self._send_json(self.app.settings.status())
            return
        if parsed.path == "/api/video/status":
            self._send_json(self.app.video_analysis.status())
            return
        if parsed.path == "/api/agent/state":
            self._send_json(self.app.agent.status())
            return
        if parsed.path == "/api/highlights/download":
            self._handle_highlight_download(parsed)
            return
        if parsed.path == "/api/capture/light-preview.mjpg":
            self._send_light_preview_mjpeg()
            return
        if parsed.path == "/api/capture/preview.mjpg":
            self._send_mjpeg(self._query_light_mode(parse_qs(parsed.query)))
            return
        if parsed.path == "/api/capture/snapshot.jpg":
            light_settings = self._capture_light_settings(self._query_light_mode(parse_qs(parsed.query)))
            self._send_jpeg(self.app.capture.snapshot_jpeg(light_settings))
            return

        self.path = parsed.path
        if self.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/model/start":
            self.app.runtime.start_async()
            self._send_json(self.app.runtime.status())
            return
        if parsed.path == "/api/model/reset":
            self.app.runtime.reset_session()
            self.app.highlight_tracker.reset_metrics()
            self._send_json(self.app.runtime.status())
            return
        if parsed.path == "/api/capture/start":
            self._send_json(self.app.capture.start(self._read_json_body()))
            return
        if parsed.path == "/api/capture/stop":
            self._send_json(self.app.capture.stop(reset_model=True))
            return
        if parsed.path == "/api/overlay/settings":
            self._send_json(self.app.settings.update(self._read_json_body()))
            return
        if parsed.path == "/api/highlights/recording":
            body = self._read_json_body()
            self.app.recording.set_enabled(as_bool(body.get("enabled"), False))
            self._send_json(self.app.highlights_state())
            return
        if parsed.path == "/api/highlights/export":
            body = self._read_json_body()
            highlight = self.app.highlight_tracker.find(str(body.get("highlight_id") or ""))
            payload, status = self.app.recording.export_highlight(highlight)
            self._send_json(payload, status=status)
            return
        if parsed.path == "/api/video/reset":
            body, status = self.app.video_analysis.reset()
            self._send_json(body, status=status)
            return
        if parsed.path == "/api/video/analyze":
            self._handle_video_upload(parsed)
            return
        if parsed.path == "/api/agent/message":
            snapshot = self.app.overlay_state(include_agent=False, observe_agent=False)
            body, status = self.app.agent.submit_user_message(self._read_json_body().get("text", ""), snapshot)
            self._send_json(body, status=status)
            return
        if parsed.path == "/api/agent/reset":
            body, status = self.app.agent.reset()
            self._send_json(body, status=status)
            return
        if parsed.path == "/api/agent/enable":
            body, status = self.app.agent.enable_api(self._read_json_body())
            self._send_json(body, status=status)
            return
        if parsed.path == "/api/agent/disable":
            body, status = self.app.agent.disable_api()
            self._send_json(body, status=status)
            return
        if parsed.path not in {"/api/model/frame", "/api/model/face"}:
            self.send_error(404, "Unknown endpoint")
            return

        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length)
        query = parse_qs(parsed.query)
        ts = None
        if "ts" in query:
            try:
                ts = float(query["ts"][0])
            except ValueError:
                ts = None
        if parsed.path == "/api/model/frame":
            self._send_json(self.app.runtime.submit_jpeg(payload, ts))
            return

        width = self._query_int(query, "w", FACE_WIDTH)
        height = self._query_int(query, "h", FACE_HEIGHT)
        hasface = self._query_bool(query, "hasface", True)
        self._send_json(self.app.runtime.submit_face(payload, ts, width, height, hasface))

    def _handle_video_upload(self, parsed):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            self._send_json({"state": "failed", "error": "empty upload"}, status=400)
            return
        if length > MAX_UPLOAD_BYTES:
            self._send_json({
                "state": "failed",
                "error": f"upload too large, max {MAX_UPLOAD_BYTES // (1024 * 1024)}MB",
            }, status=413)
            return
        query = parse_qs(parsed.query)
        filename = query.get("name", ["upload.mp4"])[0]
        body, status = self.app.video_analysis.start_from_upload(self.rfile, length, filename)
        self._send_json(body, status=status)

    def _handle_highlight_download(self, parsed):
        query = parse_qs(parsed.query)
        export_id = query.get("id", [""])[0]
        path = self.app.recording.download_path(export_id)
        if path is None:
            self.send_error(404, "Highlight export not found")
            return
        self._send_file(path, "video/mp4")

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _query_int(query, key, default):
        try:
            return int(query.get(key, [default])[0])
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _query_bool(query, key, default):
        value = query.get(key, [default])[0]
        if isinstance(value, bool):
            return value
        return str(value).lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _query_light_mode(query):
        value = str(query.get("light", ["auto"])[0]).lower()
        if value in {"0", "false", "off", "raw"}:
            return "raw"
        if value in {"1", "true", "on", "simulated"}:
            return "simulated"
        return "auto"

    def _capture_light_settings(self, light_mode: str):
        current = dict(self.app.settings.status())
        if light_mode == "raw":
            current["light_enabled"] = False
        elif light_mode == "simulated":
            current["light_enabled"] = True
        return current

    def _send_json(self, body, status=200):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def _send_jpeg(self, data: bytes):
        try:
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def _send_file(self, path: Path, content_type: str):
        data_size = path.stat().st_size
        try:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(data_size))
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def _send_mjpeg(self, light_mode="auto"):
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}")
        self.end_headers()
        try:
            while True:
                frame = self.app.capture.snapshot_jpeg(self._capture_light_settings(light_mode))
                self.wfile.write(f"--{MJPEG_BOUNDARY}\r\n".encode("ascii"))
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                time.sleep(1 / MJPEG_STREAM_FPS)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def _send_light_preview_mjpeg(self):
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}")
        self.end_headers()
        try:
            while True:
                frame = self.app.capture.light_preview_jpeg(self.app.settings.status())
                self.wfile.write(f"--{MJPEG_BOUNDARY}\r\n".encode("ascii"))
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                time.sleep(1 / MJPEG_STREAM_FPS)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def log_message(self, format, *args):
        return



def make_handler(app):
    class BoundPluginHandler(PluginHandler):
        pass

    BoundPluginHandler.app = app
    return BoundPluginHandler
