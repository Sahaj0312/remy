"""Tiny WhatsApp webhook receiver: answers Meta's GET verification and logs every POST
(message status updates and inbound replies) to events/wa_webhook.jsonl.
  python scripts/wa_webhook.py --port 8787      # then expose with: cloudflared tunnel --url http://localhost:8787
Verify token comes from .env WHATSAPP_VERIFY_TOKEN (default 'remy-verify').
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, env_get  # noqa: E402

LOG = ROOT / "events" / "wa_webhook.jsonl"
VERIFY = env_get("WHATSAPP_VERIFY_TOKEN", required=False) or "remy-verify"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        q = parse_qs(urlparse(self.path).query)
        if q.get("hub.mode") == ["subscribe"] and q.get("hub.verify_token") == [VERIFY]:
            body = q.get("hub.challenge", [""])[0].encode()
            self.send_response(200); self.send_header("Content-Type", "text/plain"); self.end_headers(); self.wfile.write(body)
            print(f"[{time.strftime('%H:%M:%S')}] webhook verified by Meta", flush=True)
        else:
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); raw = self.rfile.read(n)
        self.send_response(200); self.end_headers()
        try:
            data = json.loads(raw)
        except Exception:
            data = {"raw": raw.decode(errors="replace")}
        LOG.parent.mkdir(exist_ok=True)
        with open(LOG, "a") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "body": data}) + "\n")
        # one-line summary
        try:
            for e in data.get("entry", []):
                for ch in e.get("changes", []):
                    v = ch.get("value", {})
                    for s in v.get("statuses", []):
                        err = s.get("errors", [{}])[0] if s.get("errors") else {}
                        print(f"[{time.strftime('%H:%M:%S')}] status {s.get('status')} to {s.get('recipient_id')} id={s.get('id','')[-12:]} "
                              f"{('ERROR ' + str(err.get('code')) + ' ' + str(err.get('title')) + ' :: ' + str((err.get('error_data') or {}).get('details'))) if err else ''}", flush=True)
                    for m in v.get("messages", []):
                        print(f"[{time.strftime('%H:%M:%S')}] inbound {m.get('type')} from {m.get('from')}: {(m.get('text') or {}).get('body')!r}", flush=True)
        except Exception as ex:
            print("parse error", ex, flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--port", type=int, default=8787); a = p.parse_args()
    print(f"webhook receiver on :{a.port}, verify token set, logging to {LOG}", flush=True)
    HTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
