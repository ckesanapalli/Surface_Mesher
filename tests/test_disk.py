import re
import pytest
import numpy as np
from surfmesh import (
    disk_mesher_radial,
    circumference_edges,
    disk_mesher_square_centered,
)

# ---------------------------------- #
# disk_mesher_radial Tests             #
# ---------------------------------- #

def test_disk_mesher_radial_basic_shape():
    mesh = disk_mesher_radial(1.0, 5, 8)
    expected_faces = 5 * 8
    assert isinstance(mesh, np.ndarray)
    assert mesh.shape == (expected_faces, 4, 2)

def test_disk_mesher_radial_output_dtype():
    mesh = disk_mesher_radial(1.0, 3, 3)
    assert np.issubdtype(mesh.dtype, np.floating)

def test_disk_mesher_radial_radius_limit():
    mesh = disk_mesher_radial(1.0, 10, 10)
    r = np.linalg.norm(mesh, axis=-1)
    assert np.all(r <= 1.0 + 1e-6)

@pytest.mark.parametrize("radial_resolution, segment_resolution", [
    (1, 1),
    (2, 4),
    (5, 10),
])
def test_disk_mesher_radial_various_resolutions(radial_resolution, segment_resolution):
    mesh = disk_mesher_radial(1.0, radial_resolution, segment_resolution)
    expected_faces = radial_resolution * segment_resolution
    assert mesh.shape == (expected_faces, 4, 2)

def test_disk_mesher_radial_zero_radius():
    mesh = disk_mesher_radial(0.0, 5, 8)
    assert np.allclose(mesh, 0.0)

def test_disk_mesher_radial_negative_inputs():
    with pytest.raises(ValueError, match="Invalid radius"):
        disk_mesher_radial(-1.0, 5, 5)
    with pytest.raises(ValueError, match="Invalid resolution"):
        disk_mesher_radial(1.0, -1, 5)
    with pytest.raises(ValueError, match="Invalid resolution"):
        disk_mesher_radial(1.0, 5, -3)

# ---------------------------------- #
# circumference_edges Tests          #
# ---------------------------------- #

def test_circumference_edges_basic():
    circumference = circumference_edges(1.0, 12)
    assert circumference.shape == (2, 12)

def test_circumference_edges_values():
    circ = circumference_edges(1.0, 4, start_angle=0)
    expected = np.array(
        [[ 1. , -0.5, -0.5,  1. ],
       [ 0. ,  0.9, -0.9, -0. ]]
    )
    np.testing.assert_allclose(circ.round(1), expected)

def test_circumference_edges_negative_radius():
    with pytest.raises(ValueError, match="Invalid radius"):
        circumference_edges(-1.0, 12)

def test_circumference_edges_invalid_segment_resolution():
    with pytest.raises(ValueError, match="Invalid segment_resolution"):
        circumference_edges(1.0, 0)

# ---------------------------------- #
# disk_mesher_square_centered Tests    #
# ---------------------------------- #

def test_disk_mesher_square_centered_basic_shape():
    mesh = disk_mesher_square_centered(1.0, 5, 5)
    assert mesh.shape[1:] == (4, 2)
    assert mesh.ndim == 3

def test_disk_mesher_square_centered_increasing_size():
    mesh_small = disk_mesher_square_centered(1.0, 2, 2)
    mesh_large = disk_mesher_square_centered(1.0, 5, 5)
    assert mesh_large.shape[0] > mesh_small.shape[0]

def test_disk_mesher_square_centered_invalid_inputs():
    with pytest.raises(ValueError, match="radius must be positive"):
        disk_mesher_square_centered(0.0, 5, 5)
    with pytest.raises(ValueError, match="square_resolution must be at least 1"):
        disk_mesher_square_centered(1.0, 0, 5)
    with pytest.raises(ValueError, match="radial_resolution must be at least 1"):
        disk_mesher_square_centered(1.0, 5, 0)
    with pytest.raises(ValueError, match="square_side_radius_ratio must be in"):
        disk_mesher_square_centered(1.0, 5, 5, square_side_radius_ratio=0)
    with pytest.raises(ValueError, match="square_side_radius_ratio must be in"):
        disk_mesher_square_centered(1.0, 5, 5, square_side_radius_ratio=1.5)

def test_disk_mesher_square_centered_with_rotation():
    mesh_rotated = disk_mesher_square_centered(1.0, 5, 5, square_disk_rotation=np.pi/8)
    mesh_non_rotated = disk_mesher_square_centered(1.0, 5, 5, square_disk_rotation=0)
    assert mesh_rotated.shape == mesh_non_rotated.shape

