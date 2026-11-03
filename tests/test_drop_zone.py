"""A drop that ends on the sofa or bed is not a fall. The drop path asks the zones before the SAM call and
again after it (Nov 3: a fast sit-down with the feet on the coffee table was confirmed by SAM twice)."""
import json

from remy.monitor import State
from remy.vision.zones import Zones


def zones(tmp_path):
    d = {"source": "test", "width": 640, "height": 480, "zones": {
        "floor": [{"polygon": [[0, 240], [640, 240], [640, 480], [0, 480]]}],
        "sofa": [{"polygon": [[400, 200], [640, 200], [640, 330], [400, 330]]}]}}
    p = tmp_path / "zones.json"; p.write_text(json.dumps(d))
    return Zones(p, 640, 480)


def det(box, hip=None):
    x1, y1, x2, y2 = box
    return {"box": box, "cx_px": (x1 + x2) / 2, "cy_px": (y1 + y2) / 2, "hip_px": hip, "conf": 0.9}


def test_a_sit_down_with_the_legs_out_over_the_floor_is_on_furniture(tmp_path):
    s = State(zones(tmp_path), 5.0, 480)
    for k in range(10):                                   # 2 s of records: box over the floor, hips on the sofa
        s.push(10.0 + k / 5, 0.0, det([200, 250, 500, 420], hip=[450, 300]))
    assert s.on_furniture_since(10.0) == (10, 10)


def test_a_fall_onto_the_floor_is_not(tmp_path):
    s = State(zones(tmp_path), 5.0, 480)
    for k in range(10):
        s.push(10.0 + k / 5, 0.0, det([100, 380, 300, 440]))
    assert s.on_furniture_since(10.0) == (0, 10)


def test_only_records_after_the_drop_count(tmp_path):
    s = State(zones(tmp_path), 5.0, 480)
    for k in range(10):
        s.push(k / 5, 0.0, det([420, 220, 620, 320]))        # on the sofa before the drop
    for k in range(5):
        s.push(2.0 + k / 5, 0.0, det([100, 380, 300, 440]))  # then on the floor
    s.push(3.0, 0.0, None)
    assert s.on_furniture_since(2.0) == (0, 5)
    assert s.on_furniture_since(2.0, max_s=0.1) == (0, 1)
