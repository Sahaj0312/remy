"""Serve fall_test/public/ over HTTP (port 8788) so Twilio can fetch snapshot images.
alerts.py starts this plus a cloudflared quick tunnel on demand and records the public URL in out/media_tunnel_url.txt."""
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"; PUBLIC.mkdir(exist_ok=True)


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8788
    HTTPServer(("127.0.0.1", port), partial(Quiet, directory=str(PUBLIC))).serve_forever()
