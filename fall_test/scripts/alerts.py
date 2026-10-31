"""WhatsApp Cloud API alerts (test number). Secrets in .env: WHATSAPP_TOKEN, WHATSAPP_PHONE_NUMBER_ID.
Family numbers and wording in config.json["alerts"]. --dry-run logs instead of sending.

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

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, env_get, load_config  # noqa: E402

GRAPH = "https://graph.facebook.com/v22.0"


class WhatsApp:
    def __init__(self, dry_run: bool = False):
        self.cfg = load_config()["alerts"]
        self.dry = dry_run or not self.cfg.get("enabled", True)
        self.token = env_get("WHATSAPP_TOKEN", required=not self.dry)
        self.phone_id = env_get("WHATSAPP_PHONE_NUMBER_ID", required=not self.dry)
        self.log_path = ROOT / "events" / "alerts.log"

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
        body = {"messaging_product": "whatsapp", "to": to, "type": "template",
                "template": {"name": name, "language": {"code": lang}}}
        resp = None if self.dry else self._post("messages", body)
        self._log("template", to, body, resp); return resp or {"dry": True}

    def send_text(self, to: str, text: str) -> dict:
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
