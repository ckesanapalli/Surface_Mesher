import pytest
import numpy as np
from surface_mesher import circle_mesh_radial

# ---------------------------------- #
# Basic shape and type tests         #
# ---------------------------------- #

def test_circle_mesh_radial_basic_shape():
    radius = 1.0
    radial_resolution = 5
    segment_resolution = 8
    mesh = circle_mesh_radial(radius, radial_resolution, segment_resolution)
    assert isinstance(mesh, np.ndarray)
    expected_faces = radial_resolution * segment_resolution
    assert mesh.shape == (expected_faces, 4, 2)

def test_circle_mesh_radial_output_dtype():
    mesh = circle_mesh_radial(1.0, 3, 3)
    assert np.issubdtype(mesh.dtype, np.floating)

# ---------------------------------- #
# Geometric correctness tests        #
# ---------------------------------- #

def test_circle_mesh_radial_radius_limit():
    radius = 1.0
    mesh = circle_mesh_radial(radius, 10, 10)
    distances = np.linalg.norm(mesh, axis=-1)
    assert np.all(distances <= radius + 1e-6)  # Allow small numerical error

def test_circle_mesh_radial_known_points():
    radius = 1.0
    mesh = circle_mesh_radial(radius, radial_resolution=1, segment_resolution=4).round(6)

    # Check corner points manually
    points = mesh[-1, :, :]
    expected_points = np.array([
        [0, 0],
        [0, -radius],
        [radius, 0],
        [0, 0],
    ])
    np.testing.assert_almost_equal(points, expected_points, decimal=5)

# ---------------------------------- #
# Edge case tests                    #
# ---------------------------------- #

def test_circle_mesh_radial_zero_radius():
    mesh = circle_mesh_radial(0.0, 5, 8)
    assert np.allclose(mesh, 0.0)

def test_circle_mesh_radial_minimal_resolution():
    mesh = circle_mesh_radial(1.0, 1, 1)
    assert mesh.shape == (1, 4, 2)
    # Should create a minimal mesh

# ---------------------------------- #
# Parametrized random tests          #
# ---------------------------------- #

@pytest.mark.parametrize("radial_resolution, segment_resolution", [
    (2, 2),
    (5, 5),
    (10, 20),
])
def test_circle_mesh_various_resolutions(radial_resolution, segment_resolution):
    mesh = circle_mesh_radial(1.0, radial_resolution, segment_resolution)
    expected_faces = radial_resolution * segment_resolution
    assert mesh.shape == (expected_faces, 4, 2)
# ---------------------------------- #
# Negative and invalid input tests   #
# ---------------------------------- #

def test_circle_mesh_radial_negative_radius():
    with pytest.raises(ValueError):
        circle_mesh_radial(-1.0, 5, 5)

def test_circle_mesh_radial_invalid_resolution():
    with pytest.raises(ValueError):
        circle_mesh_radial(1.0, -2, 5)

    with pytest.raises(ValueError):
        circle_mesh_radial(1.0, 5, -3)

# If you want the function to handle invalid inputs better, you need to add manual input checks in the function itself.
# Currently your function does not have guards against negative numbers — could be good idea to add!

def test_circle_faces_within_bounds():
    radius = 1.0
    mesh = circle_mesh_radial(radius, 10, 10)
    assert np.all(np.abs(mesh[..., 0]) <= radius + 1e-6)
    assert np.all(np.abs(mesh[..., 1]) <= radius + 1e-6)