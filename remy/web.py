"""A local page for watching the monitor work: the camera view with the boxes and zones drawn on, the current
state, the last alert, and the same log lines the terminal prints. Served on 127.0.0.1 only, by `remy run`
itself, with the standard library. Meant for a screen recording, not for the family.

  GET /            the page
  GET /stream.mjpg the annotated frames as motion JPEG
  GET /frame.jpg   the latest annotated frame (the page uses it with ?still=1, e.g. for a screenshot)
  GET /state.json  {state, since_s, alert, lines[], events, uptime_s}
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>remy</title>
<style>
  :root { --bg:#0e0f12; --panel:#17191e; --line:#2a2d35; --fg:#e8e8e8; --dim:#8b909a;
          --green:#39c26b; --amber:#f2b134; --red:#ef4e4e; --blue:#4ea1ff; }
  * { box-sizing:border-box } body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.4 -apple-system,Inter,system-ui,sans-serif }
  .wrap { display:grid; grid-template-columns: 2fr 1fr; gap:16px; padding:16px; height:100vh }
  .video { background:#000; border-radius:12px; overflow:hidden; display:flex; align-items:center; justify-content:center; border:1px solid var(--line) }
  .video img { width:100%; height:100%; object-fit:contain }
  .side { display:flex; flex-direction:column; gap:12px; min-height:0 }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px 16px }
  .state { font-size:30px; font-weight:700; letter-spacing:.02em }
  .state.watching{color:var(--green)} .state.candidate{color:var(--amber)} .state.event{color:var(--red)} .state.alert{color:var(--blue)}
  .since { color:var(--dim); font-variant-numeric:tabular-nums; font-size:15px; margin-top:2px }
  .chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:10px }
  .chip { font-size:12px; padding:3px 8px; border-radius:999px; border:1px solid var(--line); color:var(--dim) }
  .chip.on { border-color:var(--blue); color:#fff; background:#1b2a3d }
  .alert { white-space:pre-wrap; font-size:14px } .alert .k { color:var(--dim); font-size:12px; text-transform:uppercase; letter-spacing:.06em }
  .log { flex:1; min-height:0; overflow:hidden; font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; display:flex; flex-direction:column; justify-content:flex-end }
  .log div { white-space:pre-wrap; word-break:break-word; color:var(--fg) } .log div.dim { color:var(--dim) }
  .log div.hot { color:#fff } .log div.alert { color:var(--blue) } .log div.event { color:var(--red) }
  .brand { color:var(--dim); font-size:12px; display:flex; justify-content:space-between }
</style></head>
<body><div class="wrap">
  <div class="video"><img id="cam" src="/stream.mjpg" alt="camera"></div>
  <div class="side">
    <div class="card">
      <div class="brand"><span>remy · live</span><span id="up"></span></div>
      <div id="state" class="state watching">WATCHING</div>
      <div id="since" class="since">&nbsp;</div>
      <div class="chips">
        <span class="chip" data-k="YOLO">YOLO pose · local</span>
        <span class="chip" data-k="SAM 3.1">SAM 3.1</span>
        <span class="chip" data-k="Spark">Muse Spark</span>
        <span class="chip" data-k="Transcribe">Muse Voice Transcribe</span>
        <span class="chip" data-k="WhatsApp">WhatsApp</span>
      </div>
    </div>
    <div class="card alert"><div class="k">last message to the family</div><div id="alert">none yet</div></div>
    <div class="card log" id="log"></div>
  </div>
</div>
<script>
const cls = {watching:'watching', candidate:'candidate', event:'event', checkin:'event', alert:'alert'};
const label = {watching:'WATCHING', candidate:'CANDIDATE DROP', event:'FALL CONFIRMED', checkin:'CHECKING IN', alert:'FAMILY TOLD'};
async function tick(){
  try{
    const s = await (await fetch('/state.json',{cache:'no-store'})).json();
    const st = document.getElementById('state'); st.className='state '+(cls[s.state]||'watching'); st.textContent=label[s.state]||s.state.toUpperCase();
    document.getElementById('since').textContent = s.since_s==null ? '\\u00a0' : s.since_s.toFixed(1)+' s since the fall';
    document.getElementById('up').textContent = 'up '+Math.floor(s.uptime_s/60)+' min · '+s.events+' event'+(s.events==1?'':'s');
    document.getElementById('alert').textContent = s.alert || 'none yet';
    const log = document.getElementById('log'); log.innerHTML='';
    for (const l of s.lines.slice(-14)) { const d=document.createElement('div'); d.textContent=l;
      if (/ALERT/.test(l)) d.className='alert'; else if (/EVENT|confirms/.test(l)) d.className='event'; else if (/check-in|triage|answer clip|alert video/.test(l)) d.className='hot'; else d.className='dim';
      log.appendChild(d); }
    const recent = s.lines.slice(-3).join(' ');
    for (const c of document.querySelectorAll('.chip')) c.classList.toggle('on', recent.includes(c.dataset.k) || (c.dataset.k==='YOLO' && s.state!=='watching'));
  }catch(e){}
}
if (location.search.includes('still')) { const c=document.getElementById('cam'); c.src='/frame.jpg'; setInterval(()=>{c.src='/frame.jpg?'+Date.now()}, 500); }
setInterval(tick, 400); tick();
</script></body></html>
"""


class WebView:
    """Shared state between the monitor and the page. The monitor calls frame(), line() and status()."""

    def __init__(self, keep_lines: int = 200):
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.jpeg: bytes | None = None
        self.seq = 0
        self.lines: deque = deque(maxlen=keep_lines)
        self.state = "watching"
        self.since_wall: float | None = None
        self.alert: str | None = None
        self.events = 0
        self.t0 = time.time()
        self.server = None

    # ---- producers
    def frame(self, img: np.ndarray, quality: int = 80) -> None:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return
        with self.cond:
            self.jpeg = buf.tobytes(); self.seq += 1
            self.cond.notify_all()

    def line(self, text: str) -> None:
        with self.lock:
            self.lines.append(text)

    def status(self, state: str, since_wall: float | None = None, alert: str | None = None, events: int | None = None) -> None:
        with self.lock:
            self.state = state
            self.since_wall = since_wall
            if alert is not None:
                self.alert = alert
            if events is not None:
                self.events = events

    # ---- consumers
    def snapshot(self) -> dict:
        with self.lock:
            return {"state": self.state, "since_s": None if self.since_wall is None else round(time.time() - self.since_wall, 1),
                    "alert": self.alert, "lines": list(self.lines), "events": self.events, "uptime_s": round(time.time() - self.t0, 1)}

    def wait_frame(self, last_seq: int, timeout: float = 1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq, timeout=timeout)
            return self.jpeg, self.seq

    # ---- server
    def serve(self, port: int = 8790, host: str = "127.0.0.1") -> str:
        view = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.startswith("/state.json"):
                    body = json.dumps(view.snapshot()).encode()
                    self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
                elif self.path.startswith("/frame.jpg"):
                    with view.lock:
                        jpeg = view.jpeg
                    if jpeg is None:
                        self.send_response(503); self.end_headers(); return
                    self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(jpeg))); self.end_headers(); self.wfile.write(jpeg)
                elif self.path.startswith("/stream.mjpg"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame"); self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    seq = -1
                    try:
                        while True:
                            jpeg, seq2 = view.wait_frame(seq, timeout=1.0)
                            if jpeg is None or seq2 == seq:
                                continue
                            seq = seq2
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        return
                else:
                    body = PAGE.encode()
                    self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        self.server = ThreadingHTTPServer((host, port), H)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return f"http://{host}:{port}"

    def close(self) -> None:
        if self.server is not None:
            self.server.shutdown()


# ----------------------------------------------------------------------------- drawing
COLORS = {"watching": (107, 194, 57), "candidate": (52, 177, 242), "event": (78, 78, 239), "checkin": (78, 78, 239), "alert": (255, 161, 78)}


def annotate(img: np.ndarray, dets: list[dict], main_det: dict | None, zones, state: str, since_s: float | None, caption: str = "") -> np.ndarray:
    """Boxes, the zone outlines, and a banner with the state. Returns a new image."""
    out = img.copy()
    if zones is not None:
        zones.draw(out)
    for d in dets or []:
        x1, y1, x2, y2 = [int(round(v)) for v in d["box"]]
        main = main_det is not None and d is main_det
        cv2.rectangle(out, (x1, y1), (x2, y2), COLORS.get(state, (200, 200, 200)) if main else (120, 120, 120), 2 if main else 1)
        if main:
            cv2.circle(out, (int(d["cx_px"]), int(d["cy_px"])), 4, (0, 255, 255), -1)
    h, w = out.shape[:2]
    bar = out[0:34].copy(); bar[:] = (14, 15, 18)
    out[0:34] = cv2.addWeighted(out[0:34], 0.25, bar, 0.75, 0)
    text = {"watching": "WATCHING", "candidate": "CANDIDATE DROP", "event": "FALL CONFIRMED", "checkin": "CHECKING IN", "alert": "FAMILY TOLD"}.get(state, state.upper())
    if since_s is not None:
        text += f"   {since_s:.1f}s"
    cv2.putText(out, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLORS.get(state, (230, 230, 230)), 2, cv2.LINE_AA)
    stamp = time.strftime("%H:%M:%S")
    (tw, _), _ = cv2.getTextSize(stamp, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(out, stamp, (w - tw - 10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1, cv2.LINE_AA)
    if caption:
        cv2.putText(out, caption[:90], (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    return out
