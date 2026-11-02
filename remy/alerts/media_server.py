"""Serve <root>/public/ over HTTP (port 8788) so Twilio can fetch snapshot images.
remy.alerts.whatsapp starts this plus a cloudflared quick tunnel on demand and records the public URL."""
import sys
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler

from remy.common import ROOT

PUBLIC = ROOT / "public"


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    port = int(argv[0]) if argv else 8788
    PUBLIC.mkdir(exist_ok=True)
    HTTPServer(("127.0.0.1", port), partial(Quiet, directory=str(PUBLIC))).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
