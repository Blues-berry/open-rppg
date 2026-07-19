import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = REPO_ROOT / "demo" / "live-heart-light-plugin"
sys.path.insert(0, str(PLUGIN_DIR))

from live_heart_server.agent import AgentWorker, CompatibleClient  # noqa: E402


def stable_snapshot():
    return {
        "capture": {"state": "running", "input_fps": 30},
        "model": {"ready": True, "has_face": True, "hr": 74.0, "SQI": 0.62, "hr_window_seconds": 10},
        "output": {"bpm": 74.0, "confidence": 0.62, "status": "stable", "reason": "ready"},
        "settings": {"light_enabled": False, "light_revision": 1, "pulse": True},
    }


class AgentModeTest(unittest.TestCase):
    def test_default_local_mode_never_calls_client_and_generates_auto_reply(self):
        worker = AgentWorker()
        worker.client.create_message = lambda *_args, **_kwargs: self.fail("local mode must not call an API")
        worker.observe(stable_snapshot())
        status = worker.status()
        self.assertEqual("local", status["mode"])
        self.assertTrue(status["local_auto_enabled"])
        self.assertEqual("auto", status["history"][-1]["event"])
        self.assertTrue(status["latest"]["subtitle"])

    def test_enable_and_disable_hide_secret_and_invalidate_api_mode(self):
        worker = AgentWorker()
        with patch("live_heart_server.agent.save_agent_config") as save:
            status, code = worker.enable_api({
                "protocol": "openai",
                "base_url": "https://example.test/v1",
                "api_key": "super-secret-key",
                "model": "example-model",
            })
        self.assertEqual(200, code)
        self.assertEqual("api", status["mode"])
        self.assertTrue(status["has_api_key"])
        self.assertNotIn("super-secret-key", str(status))
        save.assert_called_once()
        status, code = worker.disable_api()
        self.assertEqual(200, code)
        self.assertEqual("local", status["mode"])

    def test_client_builds_both_supported_endpoint_shapes(self):
        client = CompatibleClient()
        client.protocol, client.base_url, client.api_key = "anthropic", "https://a.example/v1", "key"
        self.assertEqual("https://a.example/v1/messages", client._messages_url())
        self.assertEqual("key", client._headers()["x-api-key"])
        client.protocol, client.base_url = "openai", "https://o.example"
        self.assertEqual("https://o.example/v1/chat/completions", client._messages_url())
        self.assertEqual("Bearer key", client._headers()["Authorization"])


if __name__ == "__main__":
    unittest.main()
