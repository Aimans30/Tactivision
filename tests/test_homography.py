from tactivision.calibration.homography import (
    estimate_homography,
    mean_reprojection_error,
    on_pitch,
    transform_points,
)
from tactivision.calibration.pitch import PITCH_LENGTH, PITCH_WIDTH, pitch_vertices
import numpy as np


def test_template_fits_statsbomb_frame() -> None:
    vertices = pitch_vertices()
    assert vertices.shape == (32, 2)
    assert vertices[:, 0].min() == 0
    assert vertices[:, 0].max() == PITCH_LENGTH
    assert vertices[:, 1].max() == PITCH_WIDTH


def test_homography_roundtrip_on_known_corners() -> None:
    source = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32)
    target = np.array([[10, 10], [40, 12], [38, 30], [12, 28]], dtype=np.float32)
    matrix, inliers = estimate_homography(source, target, ransac_px=1.0)
    assert matrix is not None
    assert inliers is not None and inliers.all()
    projected = transform_points(source, matrix)
    assert mean_reprojection_error(source, target, matrix, inliers) < 0.05
    assert np.allclose(projected, target, atol=0.05)


def test_on_pitch_rejects_broadcast_misses() -> None:
    points = np.array([[60, 40], [200, 10], [-1, 20]], dtype=np.float32)
    assert on_pitch(points, margin=3).tolist() == [True, False, True]
