import ast
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = REPO_ROOT / "demo" / "live-heart-light-plugin"
SERVER_DIR = PLUGIN_DIR / "live_heart_server"
sys.path.insert(0, str(PLUGIN_DIR))

from live_heart_server.app import LiveHeartApp  # noqa: E402
from live_heart_server.http_handler import PluginHandler, make_handler  # noqa: E402


class LiveHttpHandlerTest(unittest.TestCase):
    def test_static_methods_do_not_reference_self(self):
        issues = []
        for path in SERVER_DIR.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                decorators = {
                    getattr(decorator, "id", getattr(decorator, "attr", ""))
                    for decorator in node.decorator_list
                }
                if "staticmethod" not in decorators:
                    continue
                if any(isinstance(child, ast.Name) and child.id == "self" for child in ast.walk(node)):
                    issues.append(f"{path}:{node.lineno}:{node.name}")
        self.assertEqual([], issues)

    def test_capture_light_settings_uses_copy_and_query_mode(self):
        source = {"light_enabled": False, "pulse": True}
        handler = object.__new__(PluginHandler)
        handler.app = SimpleNamespace(settings=SimpleNamespace(status=lambda: source))

        raw = handler._capture_light_settings("raw")
        simulated = handler._capture_light_settings("simulated")
        auto = handler._capture_light_settings("auto")

        self.assertIsNot(raw, source)
        self.assertFalse(raw["light_enabled"])
        self.assertTrue(simulated["light_enabled"])
        self.assertFalse(auto["light_enabled"])
        self.assertFalse(source["light_enabled"])

    def test_capture_snapshot_and_mjpeg_routes_return_frames(self):
        app = LiveHeartApp()
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with urlopen(f"{base_url}/api/capture/snapshot.jpg?light=simulated", timeout=5) as response:
                data = response.read(16)
                self.assertEqual(200, response.status)
                self.assertEqual("image/jpeg", response.headers.get("Content-Type"))
                self.assertTrue(data.startswith(b"\xff\xd8\xff"))

            with urlopen(f"{base_url}/api/capture/preview.mjpg?light=simulated", timeout=5) as response:
                chunk = response.read(512)
                self.assertEqual(200, response.status)
                self.assertIn("multipart/x-mixed-replace", response.headers.get("Content-Type", ""))
                self.assertIn(b"Content-Type: image/jpeg", chunk)
                self.assertIn(b"\xff\xd8\xff", chunk)
        finally:
            server.shutdown()
            server.server_close()
            app.shutdown()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
