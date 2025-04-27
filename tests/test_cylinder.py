import pytest
import numpy as np
from surfmesh import (
    cylinder_mesher_radial,
    cylinder_mesher_square_centered,
)

# ---------------------- #
# cylinder_mesher_radial Tests
# ---------------------- #

def test_cylinder_mesher_radial_basic_shape():
    mesh = cylinder_mesher_radial(1.0, 2.0, 5, 8, 6)
    assert isinstance(mesh, np.ndarray)
    assert mesh.ndim == 3
    assert mesh.shape[1:] == (4, 3)

def test_cylinder_mesher_radial_faces_count():
    radial_res, segment_res, height_res = 5, 8, 6
    expected_disk_faces = radial_res * segment_res
    expected_lateral_faces = (height_res) * segment_res
    expected_total_faces = 2 * expected_disk_faces + expected_lateral_faces
    mesh = cylinder_mesher_radial(1.0, 2.0, radial_res, segment_res, height_res)
    assert mesh.shape[0] == expected_total_faces

def test_cylinder_mesher_radial_coordinate_bounds():
    radius = 1.0
    height = 2.0
    mesh = cylinder_mesher_radial(radius, height, 4, 8, 5)
    xy = mesh[..., :2]
    r = np.linalg.norm(xy, axis=-1)
    assert np.all(r <= radius + 1e-6)
    z = mesh[..., 2]
    assert np.all(z >= -height/2 - 1e-6)
    assert np.all(z <= height/2 + 1e-6)

@pytest.mark.parametrize("radial_res, segment_res, height_res", [
    (2, 2, 2),
    (4, 8, 5),
    (6, 12, 10),
])
def test_cylinder_mesher_radial_various_resolutions(radial_res, segment_res, height_res):
    mesh = cylinder_mesher_radial(1.0, 2.0, radial_res, segment_res, height_res)
    assert mesh.shape[1:] == (4, 3)

# ---------------------- #
# cylinder_mesher_square_centered Tests
# ---------------------- #

def test_cylinder_mesher_square_centered_basic_shape():
    mesh = cylinder_mesher_square_centered(1.0, 2.0, 5, 4, 8)
    assert isinstance(mesh, np.ndarray)
    assert mesh.ndim == 3
    assert mesh.shape[1:] == (4, 3)

def test_cylinder_mesher_square_centered_faces_count():
    radial_res, half_square_res, height_res = 4, 3, 5
    square_res = 2 * half_square_res
    expected_disk_faces = (square_res * square_res) + (square_res * 4 * radial_res)
    expected_lateral_faces = height_res * (square_res * 4)
    expected_total_faces = 2 * expected_disk_faces + expected_lateral_faces
    mesh = cylinder_mesher_square_centered(1.0, 2.0, radial_res, half_square_res, height_res)
    assert mesh.shape[0] == expected_total_faces

def test_cylinder_mesher_square_centered_coordinate_bounds():
    radius = 1.0
    height = 2.0
    mesh = cylinder_mesher_square_centered(radius, height, 5, 4, 8)
    xy = mesh[..., :2]
    r = np.linalg.norm(xy, axis=-1)
    assert np.all(r <= radius + 1e-6)
    z = mesh[..., 2]
    assert np.all(z >= -height/2 - 1e-6)
    assert np.all(z <= height/2 + 1e-6)

@pytest.mark.parametrize("radial_res, half_square_res, height_res", [
    (2, 2, 2),
    (4, 3, 5),
    (6, 5, 8),
])
def test_cylinder_mesher_square_centered_various_resolutions(radial_res, half_square_res, height_res):
    mesh = cylinder_mesher_square_centered(1.0, 2.0, radial_res, half_square_res, height_res)
    assert mesh.shape[1:] == (4, 3)

# ---------------------- #
# Error Handling Tests
# ---------------------- #

@pytest.mark.parametrize("radius", [-1.0, 0.0])
def test_cylinder_mesher_radial_invalid_radius(radius):
    with pytest.raises(ValueError, match="radius must be positive"):
        cylinder_mesher_radial(radius, 2.0, 5, 8, 6)

@pytest.mark.parametrize("radius", [-1.0, 0.0])
def test_cylinder_mesher_square_centered_invalid_radius(radius):
    with pytest.raises(ValueError, match="radius must be positive"):
        cylinder_mesher_square_centered(radius, 2.0, 5, 4, 8)

@pytest.mark.parametrize("height", [-1.0, 0.0])
def test_cylinder_mesher_radial_invalid_height(height):
    with pytest.raises(ValueError, match="height must be positive"):
        cylinder_mesher_radial(1.0, height, 5, 8, 6)

@pytest.mark.parametrize("height", [-1.0, 0.0])
def test_cylinder_mesher_square_centered_invalid_height(height):
    with pytest.raises(ValueError, match="height must be positive"):
        cylinder_mesher_square_centered(1.0, height, 5, 4, 8)

@pytest.mark.parametrize("bad_res", [0, -1])
def test_cylinder_mesher_radial_invalid_resolutions(bad_res):
    with pytest.raises(ValueError):
        cylinder_mesher_radial(1.0, 2.0, bad_res, 8, 6)
    with pytest.raises(ValueError):
        cylinder_mesher_radial(1.0, 2.0, 5, bad_res, 6)
    with pytest.raises(ValueError):
        cylinder_mesher_radial(1.0, 2.0, 5, 8, bad_res)

@pytest.mark.parametrize("bad_res", [0, -1])
def test_cylinder_mesher_square_centered_invalid_resolutions(bad_res):
    with pytest.raises(ValueError):
        cylinder_mesher_square_centered(1.0, 2.0, bad_res, 4, 8)
    with pytest.raises(ValueError):
        cylinder_mesher_square_centered(1.0, 2.0, 5, bad_res, 8)
    with pytest.raises(ValueError):
        cylinder_mesher_square_centered(1.0, 2.0, 5, 4, bad_res)
