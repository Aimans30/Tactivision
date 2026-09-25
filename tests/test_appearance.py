"""Unit tests for experimental appearance descriptors / remapping."""

from __future__ import annotations

import numpy as np

from tactivision.tracking.appearance import (
    AppearanceParams,
    appearance_similarity,
    appearance_vector,
    bbox_center,
    clip_bbox,
    crop_quality_ok,
)
from tactivision.tracking.appearance_remap import AppearanceRemapper
from tactivision.tracking.schemas import Track


def _solid_frame(color_bgr: tuple[int, int, int], size: tuple[int, int] = (200, 200)) -> np.ndarray:
    h, w = size
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = color_bgr
    return frame


def test_clip_and_quality_gates():
    assert clip_bbox((-10, -10, 5, 5), 100, 100) == (0, 0, 5, 5)
    assert clip_bbox((10, 10, 12, 12), 100, 100) == (10, 10, 12, 12)
    assert crop_quality_ok((0, 0, 10, 10), AppearanceParams(min_crop_side=16)) is False
    assert crop_quality_ok((0, 0, 40, 40), AppearanceParams()) is True


def test_descriptor_deterministic_and_similarity_range():
    frame = _solid_frame((40, 180, 40))  # green-ish BGR
    bbox = (20.0, 20.0, 100.0, 160.0)
    a = appearance_vector(frame, bbox)
    b = appearance_vector(frame, bbox)
    assert a is not None and b is not None
    assert a.shape == b.shape
    assert np.allclose(a, b)
    sim = appearance_similarity(a, b)
    assert 0.0 <= sim <= 1.0
    assert sim > 0.99


def test_invalid_small_crop_returns_none():
    frame = _solid_frame((0, 0, 255))
    assert appearance_vector(frame, (0.0, 0.0, 8.0, 8.0)) is None


def test_remapper_links_new_id_to_lost_similar_nearby():
    params = AppearanceParams(
        appearance_threshold=0.5,
        max_gap_frames=10,
        max_center_distance_px=50.0,
        min_crop_side=10,
        min_crop_area=100,
    )
    remapper = AppearanceRemapper(params)
    frame = _solid_frame((30, 200, 30))
    # Track 1 present
    t1 = Track(1, "player", 0.9, (40.0, 40.0, 100.0, 160.0))
    out1 = remapper.apply(frame, 0, (t1,))
    assert out1[0].track_id == 1
    # Lost for a few frames (empty)
    remapper.apply(frame, 1, ())
    remapper.apply(frame, 2, ())
    # New ByteTrack id 99 nearby, same appearance
    t99 = Track(99, "player", 0.9, (45.0, 42.0, 105.0, 162.0))
    out2 = remapper.apply(frame, 3, (t99,))
    assert out2[0].track_id == 1
    assert any(e["from_id"] == 99 and e["to_id"] == 1 for e in remapper.remap_events)


def test_remapper_rejects_far_or_dissimilar():
    params = AppearanceParams(
        appearance_threshold=0.9,
        max_gap_frames=5,
        max_center_distance_px=30.0,
        min_crop_side=10,
        min_crop_area=100,
    )
    remapper = AppearanceRemapper(params)
    green = _solid_frame((30, 200, 30))
    red = _solid_frame((30, 30, 200))
    remapper.apply(green, 0, (Track(1, "player", 0.9, (40.0, 40.0, 100.0, 160.0)),))
    remapper.apply(green, 1, ())
    # Far away + different color
    out = remapper.apply(red, 2, (Track(7, "player", 0.9, (150.0, 150.0, 190.0, 190.0)),))
    assert out[0].track_id == 7


def test_bbox_center():
    assert bbox_center((0.0, 0.0, 10.0, 20.0)) == (5.0, 10.0)


def test_remapper_locks_raw_id_no_flipflop():
    """Once raw ID 99 is remapped to 1, later frames must keep that binding."""
    params = AppearanceParams(
        appearance_threshold=0.5,
        max_gap_frames=10,
        max_center_distance_px=50.0,
        min_crop_side=10,
        min_crop_area=100,
    )
    remapper = AppearanceRemapper(params)
    frame = _solid_frame((30, 200, 30))
    remapper.apply(frame, 0, (Track(1, "player", 0.9, (40.0, 40.0, 100.0, 160.0)),))
    remapper.apply(frame, 1, ())
    remapper.apply(frame, 2, (Track(99, "player", 0.9, (45.0, 42.0, 105.0, 162.0)),))
    # Gap then same raw 99 again — must stay locked to 1, not re-associate.
    remapper.apply(frame, 3, ())
    out = remapper.apply(frame, 4, (Track(99, "player", 0.9, (46.0, 43.0, 106.0, 163.0)),))
    assert out[0].track_id == 1
    assert sum(1 for e in remapper.remap_events if e["from_id"] == 99) == 1


def test_remapper_no_duplicate_ids_same_frame():
    """Two new tracks must not claim the same lost gallery ID."""
    params = AppearanceParams(
        appearance_threshold=0.3,
        max_gap_frames=20,
        max_center_distance_px=200.0,
        min_crop_side=10,
        min_crop_area=100,
    )
    remapper = AppearanceRemapper(params)
    frame = _solid_frame((30, 200, 30))
    remapper.apply(frame, 0, (Track(1, "player", 0.9, (40.0, 40.0, 100.0, 160.0)),))
    remapper.apply(frame, 1, ())
    out = remapper.apply(
        frame,
        2,
        (
            Track(50, "player", 0.9, (42.0, 42.0, 102.0, 162.0)),
            Track(51, "player", 0.9, (48.0, 44.0, 108.0, 164.0)),
        ),
    )
    ids = [t.track_id for t in out]
    assert len(ids) == len(set(ids))
    assert 1 in ids
