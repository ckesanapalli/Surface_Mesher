import re
import pytest
import numpy as np
from surface_mesher.cuboid import (
    convert_2d_face_to_3d,
    quad_faces_from_edges,
    cuboid_mesh,
    cuboid_mesh_with_resolution,
)

# --------------------------- #
# convert_2d_face_to_3d Tests  #
# --------------------------- #

def test_convert_2d_face_to_3d_basic():
    quad_2d = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]])
    result = convert_2d_face_to_3d(quad_2d, fixed_axis=2, fixed_value=5.0)
    expected = np.array([[[0, 0, 5], [1, 0, 5], [1, 1, 5], [0, 1, 5]]])
    np.testing.assert_array_equal(result, expected)

def test_convert_2d_face_to_3d_different_axes():
    quad_2d = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]])
    for axis in [0, 1, 2]:
        result = convert_2d_face_to_3d(quad_2d, fixed_axis=axis, fixed_value=3.5)
        assert result.shape == (1, 4, 3)
        assert np.all(result[:, :, axis] == 3.5)

def test_invalid_axis():
    quad_2d = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]])
    with pytest.raises(ValueError, match=re.escape("fixed_axis must be 0 (x), 1 (y), or 2 (z). Got 3.")):
        convert_2d_face_to_3d(quad_2d, fixed_axis=3, fixed_value=5.0)

# --------------------------- #
# quad_faces_from_edges Tests #
# --------------------------- #

def test_quad_faces_from_edges_basic():
    u = np.array([0, 1])
    v = np.array([0, 1])
    result = quad_faces_from_edges(u, v)
    expected = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]])
    np.testing.assert_array_equal(result, expected)

def test_quad_faces_from_edges_zero_area():
    u = np.array([0])
    v = np.array([0])
    result = quad_faces_from_edges(u, v)
    assert result.shape == (0, 4, 2)

def test_quad_faces_from_edges_rectangular_grid():
    u = np.array([0, 1, 2])
    v = np.array([0, 1])
    result = quad_faces_from_edges(u, v)
    assert result.shape == (2, 4, 2)

# --------------------------- #
# cuboid_mesh Tests           #
# --------------------------- #

def test_cuboid_mesh_basic():
    x = [0.0, 1.0]
    y = [0.0, 1.0]
    z = [0.0, 1.0]
    faces = cuboid_mesh(x, y, z)
    assert faces.shape == (6, 4, 3)

def test_cuboid_mesh_multiple_cells():
    x = [0.0, 1.0, 2.0]
    y = [0.0, 0.5, 1.0]
    z = [0.0, 0.5, 1.0]
    faces = cuboid_mesh(x, y, z)
    assert faces.shape == (24, 4, 3)

@pytest.mark.parametrize("container_type", [list, tuple, np.array])
def test_cuboid_mesh_accepts_arraylike(container_type):
    x = container_type([0.0, 1.0, 2.0])
    y = container_type([0.0, 1.0])
    z = container_type([0.0, 0.5, 1.0])
    faces = cuboid_mesh(x, y, z)
    assert faces.shape == (16, 4, 3)

def test_cuboid_mesh_invalid_dimensions():
    x = np.array([[0.0, 1.0]])  # Not 1D
    y = [0.0, 1.0]
    z = [0.0, 1.0]
    with pytest.raises(ValueError, match=re.escape("x_coords must be 1D, got shape (1, 2).")):
        cuboid_mesh(x, y, z)

def test_cuboid_mesh_too_few_values():
    x = [0.0]
    y = [0.0, 1.0]
    z = [0.0, 1.0]
    with pytest.raises(ValueError, match="x_coords must have at least 2 points, got 1."):
        cuboid_mesh(x, y, z)

def test_cuboid_mesh_non_strictly_increasing():
    x = [0.0, 1.0, 0.5]
    y = [0.0, 1.0]
    z = [0.0, 1.0]
    with pytest.raises(ValueError, match="x_coords must be strictly increasing."):
        cuboid_mesh(x, y, z)

def test_cuboid_mesh_quad_shape_and_ccw():
    x = [0.0, 1.0]
    y = [0.0, 1.0]
    z = [0.0, 1.0]
    faces = cuboid_mesh(x, y, z)
    assert faces.shape[1:] == (4, 3)

def test_cuboid_mesh_large_axis():
    x = np.linspace(0, 1, 10)
    y = [0.0, 1.0]
    z = [0.0, 1.0]
    faces = cuboid_mesh(x, y, z)
    assert faces.shape[0] == 38

def test_cuboid_mesh_degenerate_case_single_face_each():
    x = [0.0, 1.0]
    y = [0.0, 1.0]
    z = [0.0, 1.0]
    faces = cuboid_mesh(x, y, z)
    assert faces.shape[0] == 6
    for quad in faces:
        assert quad.shape == (4, 3)

# --------------------------- #
# cuboid_mesh_with_resolution Tests #
# --------------------------- #

def test_cuboid_mesh_with_resolution_scalar():
    faces = cuboid_mesh_with_resolution(2.0, 1.0, 1.0, origin=(0.0, 0.0, 0.0), resolution=2)
    assert faces.shape == (24, 4, 3)

def test_cuboid_mesh_with_resolution_arraylike():
    faces = cuboid_mesh_with_resolution(2.0, 1.0, 1.0, origin=(0.0, 0.0, 0.0), resolution=[2, 1, 2])
    assert isinstance(faces, np.ndarray)
    assert faces.shape[1:] == (4, 3)

def test_cuboid_mesh_with_resolution_invalid_shape():
    with pytest.raises(ValueError, match="resolution must be a single int or an array-like of three ints."):
        cuboid_mesh_with_resolution(2.0, 1.0, 1.0, resolution=[2, 2])

def test_cuboid_mesh_with_resolution_nonpositive():
    with pytest.raises(ValueError, match="resolution must contain only positive values."):
        cuboid_mesh_with_resolution(2.0, 1.0, 1.0, resolution=[2, 0, 2])
