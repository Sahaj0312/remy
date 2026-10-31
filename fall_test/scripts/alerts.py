"""Family alerts over WhatsApp. Two channels, picked by config.json["alerts"]["channel"]:
  "whatsapp" : Meta WhatsApp Cloud API (.env WHATSAPP_TOKEN, WHATSAPP_PHONE_NUMBER_ID)
  "twilio"   : Twilio WhatsApp sandbox (.env TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM)
Family numbers in fall_test/family.json (gitignored), wording in config. --dry-run logs instead of sending.
Images on Twilio need a public URL: a tiny file server + cloudflared quick tunnel is started on demand.

  python scripts/alerts.py --test            # send a hello_world template + a text to every family number
  python scripts/alerts.py --text "hi"       # free-form text (needs the 24 h window: the recipient messaged the number first)
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
from pathlib import Path

import re
import secrets
import shutil
import subprocess

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, env_get, load_config  # noqa: E402

GRAPH = "https://graph.facebook.com/v22.0"
PUBLIC = ROOT / "public"
TUNNEL_FILE = ROOT / "out" / "media_tunnel_url.txt"


def ensure_media_tunnel(timeout_s: float = 25) -> str | None:
    """Start media_server.py + a cloudflared quick tunnel if not running; return the public base URL."""
    if TUNNEL_FILE.exists():
        url = TUNNEL_FILE.read_text().strip()
        try:
            if url and requests.get(url + "/", timeout=8).status_code in (200, 404):
                return url
        except Exception:
            pass
    PUBLIC.mkdir(exist_ok=True); (ROOT / "out").mkdir(exist_ok=True)
    if subprocess.run(["pgrep", "-f", "scripts/media_server.py"], capture_output=True).returncode != 0:
        subprocess.Popen([str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "scripts" / "media_server.py"), "8788"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-f", "cloudflared tunnel --url http://localhost:8788"], capture_output=True)
    log = open(ROOT / "out" / "media_tunnel.log", "w")
    subprocess.Popen(["cloudflared", "tunnel", "--url", "http://localhost:8788"], stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        txt = (ROOT / "out" / "media_tunnel.log").read_text(errors="replace")
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", txt)
        if m:
            url = m.group(0)
            try:
                if requests.get(url + "/", timeout=8).status_code in (200, 404):
                    TUNNEL_FILE.write_text(url); return url
            except Exception:
                pass
        time.sleep(1)
    return None


def publish_file(path: Path) -> str | None:
    """Copy a file into public/ under a random name and return its public URL (or None)."""
    base = ensure_media_tunnel()
    if not base:
        return None
    name = f"{secrets.token_urlsafe(12)}{path.suffix}"
    shutil.copy(path, PUBLIC / name)
    return f"{base}/{name}"


class WhatsApp:
    def __init__(self, dry_run: bool = False):
        self.cfg = load_config()["alerts"]
        self.dry = dry_run or not self.cfg.get("enabled", True)
        self.channel = self.cfg.get("channel", "whatsapp")
        if self.channel == "twilio":
            self.sid = env_get("TWILIO_ACCOUNT_SID", required=not self.dry)
            self.auth = env_get("TWILIO_AUTH_TOKEN", required=not self.dry)
            self.twilio_from = env_get("TWILIO_WHATSAPP_FROM", required=False) or "whatsapp:+14155238886"
        else:
            self.token = env_get("WHATSAPP_TOKEN", required=not self.dry)
            self.phone_id = env_get("WHATSAPP_PHONE_NUMBER_ID", required=not self.dry)
        self.log_path = ROOT / "events" / "alerts.log"

    # ---- Twilio sandbox
    def _twilio(self, to: str, body: str, media_url: str | None = None) -> dict:
        data = {"From": self.twilio_from, "To": f"whatsapp:{to}", "Body": body}
        if media_url:
            data["MediaUrl"] = media_url
        r = requests.post(f"https://api.twilio.com/2010-04-01/Accounts/{self.sid}/Messages.json", data=data,
                          auth=(self.sid, self.auth), timeout=60)
        try:
            out = r.json()
        except Exception:
            out = {"text": r.text[:300]}
        out["_http"] = r.status_code
        return {k: out.get(k) for k in ("sid", "status", "error_code", "error_message", "_http", "message")}

    def _log(self, kind: str, to: str, payload: dict, resp: dict | None):
        self.log_path.parent.mkdir(exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, "to": to, "dry": self.dry,
                                "payload": payload, "response": resp}) + "\n")

    def _post(self, path: str, body: dict | None = None, files=None, data=None) -> dict:
        r = requests.post(f"{GRAPH}/{self.phone_id}/{path}", headers={"Authorization": f"Bearer {self.token}"},
                          json=body, files=files, data=data, timeout=60)
        try:
            out = r.json()
        except Exception:
            out = {"status": r.status_code, "text": r.text[:300]}
        out["_http"] = r.status_code
        return out

    def recipients(self) -> list[dict]:
        fam = ROOT / "family.json"                       # real numbers, gitignored; config holds a placeholder
        if fam.exists():
            return json.load(open(fam))["family"]
        return self.cfg["family"]

    def send_template(self, to: str, name: str = "hello_world", lang: str = "en_US") -> dict:
        if self.channel == "twilio":
            return self.send_text(to, "Remy alerts connected (Twilio sandbox).")
        body = {"messaging_product": "whatsapp", "to": to, "type": "template",
                "template": {"name": name, "language": {"code": lang}}}
        resp = None if self.dry else self._post("messages", body)
        self._log("template", to, body, resp); return resp or {"dry": True}

    def send_text(self, to: str, text: str) -> dict:
        if self.channel == "twilio":
            resp = None if self.dry else self._twilio(to, text)
            self._log("twilio-text", to, {"text": text}, resp); return resp or {"dry": True}
        body = {"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": text, "preview_url": False}}
        resp = None if self.dry else self._post("messages", body)
        self._log("text", to, body, resp); return resp or {"dry": True}

    def upload_media(self, path: Path) -> str | None:
        if self.dry:
            return None
        mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        resp = self._post("media", files={"file": (path.name, path.read_bytes(), mime)},
                          data={"messaging_product": "whatsapp", "type": mime})
        self._log("media", "-", {"file": str(path)}, resp)
        return resp.get("id")

    def send_image(self, to: str, path: Path, caption: str) -> dict:
        if self.channel == "twilio":
            url = None if self.dry else publish_file(path)
            resp = None if self.dry else self._twilio(to, caption, url)
            self._log("twilio-image", to, {"file": str(path), "url": url, "caption": caption}, resp); return resp or {"dry": True}
        media_id = self.upload_media(path)
        body = {"messaging_product": "whatsapp", "to": to, "type": "image",
                "image": ({"id": media_id} if media_id else {"id": "dry"}) | {"caption": caption[:1024]}}
        resp = None if self.dry else self._post("messages", body)
        self._log("image", to, body, resp); return resp or {"dry": True}

    # ---- the two alert shapes
    def calm(self, summary: str, said: str | None, when: str) -> list[dict]:
        name = self.cfg["person_name"]
        text = f"{self.cfg['calm_prefix']} {name} had a fall at {when} but says she's okay.\n{summary}"
        if said:
            text += f'\nShe said: "{said}"'
        return [self.send_text(r["number"], text) for r in self.recipients()]

    def urgent(self, summary: str, said: str | None, when: str, snapshot: Path | None, reason: str) -> list[dict]:
        name = self.cfg["person_name"]
        text = (f"{self.cfg['urgent_prefix']} {name} may need help. Fall detected at {when}. {reason}\n{summary}")
        text += f'\nShe said: "{said}"' if said else "\nShe did not answer the check-in."
        out = []
        for r in self.recipients():
            if snapshot and snapshot.exists():
                out.append(self.send_image(r["number"], snapshot, text))
            else:
                out.append(self.send_text(r["number"], text))
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true")
    p.add_argument("--text")
    p.add_argument("--image")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    wa = WhatsApp(dry_run=a.dry_run)
    for r in wa.recipients():
        if a.test:
            print("template ->", r["name"], wa.send_template(r["number"]))
        if a.text:
            print("text ->", r["name"], wa.send_text(r["number"], a.text))
        if a.image:
            print("image ->", r["name"], wa.send_image(r["number"], Path(a.image), a.text or "test"))


if __name__ == "__main__":
    main()
