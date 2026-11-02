from remy.vision.tracking import (OnlineTracks, finalize_objects, link_detections, link_score, merge_broken_tracks,
                                  suppress_duplicates)


def det(x1, y1, x2, y2, conf=0.9):
    return {"box": [x1, y1, x2, y2], "cx_px": (x1 + x2) / 2, "cy_px": (y1 + y2) / 2, "mask_area_px": None, "conf": conf}


def test_link_score_keeps_a_box_that_collapses_from_tall_to_wide():
    tall, wide = [300, 100, 360, 280], [230, 220, 410, 280]
    assert link_score(tall, wide) > 0.15        # IoU alone would be tiny; overlap over the smaller box saves it


def test_suppress_duplicates_drops_a_partial_body_inside_a_better_box():
    full = det(300, 100, 360, 280, 0.9)
    partial = det(305, 110, 355, 200, 0.4)       # inside `full`, low IoU, high overlap
    far = det(50, 100, 110, 280, 0.8)
    kept = suppress_duplicates([partial, full, far], max_overlap=0.5)
    assert full in kept and far in kept and partial not in kept


def test_link_detections_follows_one_person_and_starts_a_new_track_for_a_far_box():
    frames = {i: [det(100 + 5 * i, 100, 160 + 5 * i, 280)] for i in range(10)}
    frames[5].append(det(500, 100, 560, 280))
    tracks = link_detections(frames, max_gap=5, min_score=0.15)
    assert len(tracks) == 2
    long = max(tracks.values(), key=len)
    assert sorted(long) == list(range(10))


def test_merge_broken_tracks_joins_a_track_that_resumes_after_a_short_gap():
    a = {i: det(100, 100, 160, 280) for i in range(0, 5)}
    b = {i: det(102, 100, 162, 280) for i in range(8, 12)}
    merged, id_map = merge_broken_tracks({"1": a, "2": b}, max_gap=5, min_score=0.15)
    assert len(merged) == 1 and id_map["2"] == "1"
    assert sorted(merged["1"]) == [0, 1, 2, 3, 4, 8, 9, 10, 11]


def test_merge_broken_tracks_keeps_overlapping_in_time_tracks_apart():
    a = {i: det(100, 100, 160, 280) for i in range(0, 10)}
    b = {i: det(100, 100, 160, 280) for i in range(5, 15)}
    merged, _ = merge_broken_tracks({"1": a, "2": b}, max_gap=5, min_score=0.15)
    assert len(merged) == 2


def test_finalize_objects_picks_the_moving_person_over_a_seated_bystander():
    mover = {i: det(100 + 20 * i, 100 + 10 * i, 160 + 20 * i, 280 + 10 * i) for i in range(10)}
    seated = {i: det(500, 200, 560, 300) for i in range(10)}
    objects, primary = finalize_objects({"a": seated, "b": mover}, n_frames=12, fps=10.0, W=640, H=480)
    assert primary == "b"
    assert len(objects["b"]) == 12 and objects["b"][11]["present"] is False
    f = objects["b"][0]
    assert 0 < f["cx"] < 1 and 0 < f["cy"] < 1 and f["ar"] == 3.0


def test_online_tracks_keeps_one_id_through_small_motion_and_drops_it_after_the_gap():
    ot = OnlineTracks(fps=5.0, H=480, W=640, max_gap_s=1.0, min_score=0.15, keep_s=8.0)
    for i in range(5):
        ot.update([det(100 + 4 * i, 100, 160 + 4 * i, 280)])
    assert list(ot.tracks) == ["1"]
    tr = ot.as_track()
    assert tr["frames"][0]["i"] == 0 and tr["frames"][-1]["t"] == 0.8
    for _ in range(6):                      # 6 empty ticks > max_gap (5 ticks at 5 fps)
        ot.update([])
    assert ot.tracks == {}


def test_online_tracks_marks_missing_ticks_as_not_present():
    ot = OnlineTracks(fps=5.0, H=480, W=640, max_gap_s=1.0, min_score=0.15, keep_s=8.0)
    ot.update([det(100, 100, 160, 280)]); ot.update([]); ot.update([det(100, 100, 160, 280)])
    frames = ot.as_track()["frames"]
    assert [f["present"] for f in frames] == [True, False, True]
