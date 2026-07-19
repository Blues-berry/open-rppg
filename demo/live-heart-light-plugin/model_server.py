"""Local Open-rppg capture server for the live heart light plugin.

Run from the repository root:

    python demo/live-heart-light-plugin/model_server.py

Then open:

    http://127.0.0.1:8020/
"""

from __future__ import annotations

import atexit
from http.server import ThreadingHTTPServer

from live_heart_server.app import LiveHeartApp
from live_heart_server.config import HOST, PORT
from live_heart_server.http_handler import make_handler


def main():
    app = LiveHeartApp()
    atexit.register(app.shutdown)
    server = ThreadingHTTPServer((HOST, PORT), make_handler(app))
    print(f"Open-rppg live plugin server: http://{HOST}:{PORT}/")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
