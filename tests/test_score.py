"""Score v2 and the trigger on synthetic tracks. A standing person is a 60x180 box (body length 0.375 of a
480 px frame); a fall moves the centroid down 120 px in 0.4 s and leaves a 180x60 box (ar 0.33)."""
from conftest import make_track

from remy.vision.score import score_track

STAND = (320, 240, 60, 180)
FLAT = (320, 360, 180, 60)


def v2(cfg):
    return cfg["score_v2"]


def trig(cfg):
    return {**cfg["score_v2"], **cfg["trigger"]}


def test_fast_fall_then_still_is_flagged_with_hold_ok(cfg):
    tr = make_track([(3.0, *STAND), (0.4, *FLAT), (4.0, *FLAT)])
    r = score_track(tr, v2(cfg), "v2")
    assert r["flag"] is True
    assert r["best"]["hold"] == "ok"
    assert r["best"]["drop_ok"] and r["best"]["shape_ok"]
    assert r["best"]["drop_rel"] > cfg["score_v2"]["drop_rel_min"]
    assert r["best"]["drop_speed_rel_max"] > cfg["score_v2"]["drop_speed_rel_min"]


def test_fall_at_end_of_clip_is_flagged_with_hold_unknown(cfg):
    tr = make_track([(3.0, *STAND), (0.4, *FLAT), (0.8, *FLAT)])
    r = score_track(tr, v2(cfg), "v2")
    assert r["flag"] is True
    assert r["best"]["hold"] == "unknown"


def test_getting_up_cancels_the_fall(cfg):
    tr = make_track([(3.0, *STAND), (0.4, *FLAT), (1.0, *FLAT), (0.5, *STAND), (2.0, *STAND)])
    r = score_track(tr, v2(cfg), "v2")
    assert r["flag"] is False
    assert r["best"]["hold"] == "failed"
    assert r["best"]["getup_t"] is not None


def test_slow_lie_down_is_not_a_fall(cfg):
    tr = make_track([(3.0, *STAND), (3.0, *FLAT), (4.0, *FLAT)])      # same end pose, 7x slower
    r = score_track(tr, v2(cfg), "v2")
    assert r["flag"] is False


def test_fast_drop_that_stays_tall_fails_the_shape_check_but_trips_the_trigger(cfg):
    crouch = (320, 330, 70, 120)          # centroid drops 90 px fast, box still taller than wide
    tr = make_track([(3.0, *STAND), (0.4, *crouch), (4.0, *crouch)])
    assert score_track(tr, v2(cfg), "v2")["flag"] is False
    assert score_track(tr, trig(cfg), "trigger")["flag"] is True


def test_standing_still_scores_nothing(cfg):
    tr = make_track([(8.0, *STAND)])
    r = score_track(tr, v2(cfg), "v2")
    assert r["flag"] is False and r["best"] is None


def test_too_little_coverage_is_rejected_with_a_reason(cfg):
    tr = make_track([(6.0, None), (0.5, *STAND), (6.0, None)])
    r = score_track(tr, v2(cfg), "v2")
    assert r["flag"] is False
    assert r["reason"] == "insufficient tracking coverage"


def test_multi_object_track_flags_the_clip_when_any_person_falls(cfg):
    faller = make_track([(3.0, *STAND), (0.4, *FLAT), (4.0, *FLAT)])["frames"]
    seated = make_track([(7.4, 100, 300, 70, 110)])["frames"]
    tr = {"clip": "two", "fps": 10.0, "height": 480, "width": 640, "frames": seated, "objects": {"1": seated, "2": faller}}
    r = score_track(tr, v2(cfg), "v2")
    assert r["flag"] is True and r["object_id"] == "2"
    assert set(r["objects_scored"]) == {"1", "2"}
