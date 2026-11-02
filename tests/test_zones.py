import json

import pytest

from remy.vision.zones import Zones


@pytest.fixture
def zones_file(tmp_path):
    d = {"source": "test", "width": 640, "height": 480, "zones": {
        "floor": [{"polygon": [[0, 240], [640, 240], [640, 480], [0, 480]]}],
        "sofa": [{"polygon": [[400, 200], [640, 200], [640, 330], [400, 330]]}],
    }}
    p = tmp_path / "zones.json"
    p.write_text(json.dumps(d))
    return p


def test_person_on_the_floor(zones_file):
    z = Zones(zones_file, 640, 480)
    r = z.classify_box([100, 380, 300, 440])        # lying low on the left
    assert r["on_floor"] is True and r["on_furniture"] is False
    assert r["foot_zone"] == "floor"


def test_person_on_the_sofa_is_furniture_even_though_the_feet_touch_the_floor_polygon(zones_file):
    z = Zones(zones_file, 640, 480)
    r = z.classify_box([420, 220, 620, 320])        # box centre inside the sofa, bottom edge on the floor polygon
    assert r["on_furniture"] is True and r["on_floor"] is False


def test_standing_person_outside_every_zone(zones_file):
    z = Zones(zones_file, 640, 480)
    r = z.classify_box([100, 20, 160, 200])
    assert r["foot_zone"] is None and r["on_floor"] is False and r["on_furniture"] is False


def test_zones_scale_to_the_live_frame_size(zones_file):
    z = Zones(zones_file, 320, 240)                 # half the setup frame
    assert z.classify_box([50, 190, 150, 220])["on_floor"] is True
    assert z.classify_box([210, 110, 310, 160])["on_furniture"] is True
