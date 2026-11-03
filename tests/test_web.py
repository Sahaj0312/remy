import json
import socket
import urllib.request

import numpy as np

from remy.web import WebView, annotate


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def test_page_state_and_stream_are_served():
    view = WebView()
    url = view.serve(free_port())
    try:
        view.line("[12:00:00.000] source tapo"); view.status("event", 1000.0, "✅ test alert", 1)
        view.frame(annotate(np.zeros((90, 160, 3), np.uint8), [{"box": [10, 10, 50, 80], "cx_px": 30, "cy_px": 45}], None, None, "event", 3.2, "cap"))
        html = urllib.request.urlopen(url + "/", timeout=5).read().decode()
        assert "<title>remy</title>" in html and "/stream.mjpg" in html
        st = json.loads(urllib.request.urlopen(url + "/state.json", timeout=5).read())
        assert st["state"] == "event" and st["alert"] == "✅ test alert" and st["events"] == 1 and st["lines"][-1].endswith("source tapo")
        with urllib.request.urlopen(url + "/stream.mjpg", timeout=5) as r:
            head = r.read(200)
        assert b"--frame" in head and b"image/jpeg" in head
    finally:
        view.close()


def test_annotate_returns_a_new_image_of_the_same_size():
    img = np.zeros((360, 640, 3), np.uint8)
    out = annotate(img, [], None, None, "watching", None)
    assert out.shape == img.shape and out is not img and out[0:34].mean() > 0
