import pytest
import numpy as np
from surfmesh.revolve import revolve_curve_along_path, circular_revolve

# ----------------------------- #
# revolve_curve_along_path tests #
# ----------------------------- #

def test_revolve_curve_along_path_basic():
    curve = np.array([[1, 2], [3, 4]])
    revolve_path = np.array([[0, 1], [np.pi/2, 2]])
    mesh = revolve_curve_along_path(curve, revolve_path)
    assert isinstance(mesh, np.ndarray)
    assert mesh.shape == (1, 4, 3)

def test_revolve_curve_along_path_multiple_segments():
    curve = np.array([[0, 0], [1, 1], [2, 0]])
    revolve_path = np.array([[0, 1], [np.pi/2, 2], [np.pi, 3]])
    mesh = revolve_curve_along_path(curve, revolve_path)
    assert mesh.shape == ((3-1)*(3-1), 4, 3)  # (m-1)*(n-1)

def test_revolve_curve_along_path_invalid_curve_shape():
    curve = np.array([1, 2, 3])  # 1D
    revolve_path = np.array([[0, 1], [np.pi/2, 2]])
    with pytest.raises(ValueError, match="Curve must be a \\(n, 2\\) array"):
        revolve_curve_along_path(curve, revolve_path)

def test_revolve_curve_along_path_invalid_revolve_path_shape():
    curve = np.array([[1, 2], [3, 4]])
    revolve_path = np.array([0, 1])  # 1D
    with pytest.raises(ValueError, match="Revolve path must be a \\(m, 2\\) array"):
        revolve_curve_along_path(curve, revolve_path)

# ----------------------------- #
# circular_revolve tests        #
# ----------------------------- #

def test_circular_revolve_basic():
    curve = np.array([
        [1.0, 0.0],
        [2.0, 1.0],
    ])
    segment_resolution = 8
    mesh = circular_revolve(curve, segment_resolution)
    assert isinstance(mesh, np.ndarray)
    assert mesh.shape == (8 * (2-1), 4, 3)  # (segments * (n-1))

def test_circular_revolve_full_circle():
    curve = np.array([
        [1.0, 0.0],
        [2.0, 1.0],
        [2.5, 2.0],
    ])
    segment_resolution = 12
    mesh = circular_revolve(curve, segment_resolution)
    assert mesh.shape == (12 * (3-1), 4, 3)

def test_circular_revolve_partial_revolve():
    curve = np.array([
        [1.0, 0.0],
        [2.0, 1.0],
    ])
    segment_resolution = 4
    mesh = circular_revolve(curve, segment_resolution, start_angle=0, end_angle=np.pi)
    assert mesh.shape == (4 * (2-1), 4, 3)

def test_circular_revolve_invalid_curve_shape():
    curve = np.array([1.0, 2.0])  # 1D
    with pytest.raises(ValueError, match="curve must be a \\(n, 2\\) array"):
        circular_revolve(curve, segment_resolution=8)

def test_circular_revolve_invalid_segment_resolution():
    curve = np.array([
        [1.0, 0.0],
        [2.0, 1.0],
    ])
    with pytest.raises(ValueError, match="segment_resolution must be at least 1"):
        circular_revolve(curve, segment_resolution=0)

# ----------------------------- #
# Randomized property test      #
# ----------------------------- #

@pytest.mark.parametrize("segments", [4, 8, 16])
def test_circular_revolve_randomized(segments):
    np.random.seed(42)
    curve = np.random.rand(5, 2)  # Random 5-point curve
    mesh = circular_revolve(curve, segments)
    assert mesh.shape == (segments * (5-1), 4, 3)
    assert np.isfinite(mesh).all()
