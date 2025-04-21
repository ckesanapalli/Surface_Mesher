import pytest
import numpy as np
from surface_mesher import (
    cuboid_mesh,
    quad_faces_from_grid,
    cuboid_mesh_with_resolution,
)


# --------------------------- #
# generate_face_quads Tests   #
# --------------------------- #

def test_generate_face_quads_basic():
    u = np.array([0, 1])
    v = np.array([0, 1])
    result = quad_faces_from_grid(u, v, fixed_axis=2, fixed_value=5.0)
    expected = np.array([[[0, 0, 5], [1, 0, 5], [1, 1, 5], [0, 1, 5]]])
    np.testing.assert_array_equal(result, expected)


def test_generate_face_quads_valid_fixed_axes():
    u = np.array([0, 1])
    v = np.array([0, 1])
    for axis in [0, 1, 2]:
        result = quad_faces_from_grid(u, v, fixed_axis=axis, fixed_value=7.0)
        assert result.shape == (1, 4, 3)
        assert np.all(result[:, :, axis] == 7.0)


def test_generate_face_quads_zero_area():
    u = np.array([0])
    v = np.array([0])
    result = quad_faces_from_grid(u, v, 0, 0)
    assert result.shape == (0, 4, 3)


def test_generate_face_quads_rectangular_grid():
    u = np.array([0, 1, 2])
    v = np.array([0, 1])
    result = quad_faces_from_grid(u, v, 2, 3)
    assert result.shape == (2, 4, 3)


# --------------------------- #
# cuboid_mesh Tests #
# --------------------------- #

def test_valid_mesh_basic():
    x = [0.0, 1.0]
    y = [0.0, 1.0]
    z = [0.0, 1.0]
    faces = cuboid_mesh(x, y, z)
    assert faces.shape == (6, 4, 3)
    assert np.isclose(faces[:, :, 2].min(), 0.0)
    assert np.isclose(faces[:, :, 2].max(), 1.0)


def test_valid_mesh_multiple_cells():
    x = [0.0, 1.0, 2.0]
    y = [0.0, 0.5, 1.0]
    z = [0.0, 0.5, 1.0]
    faces = cuboid_mesh(x, y, z)
    assert faces.shape == (24, 4, 3)


@pytest.mark.parametrize("container_type", [list, tuple, np.array])
def test_accepts_all_arraylike_types(container_type):
    x = container_type([0.0, 1.0, 2.0])
    y = container_type([0.0, 1.0])
    z = container_type([0.0, 0.5, 1.0])
    faces = cuboid_mesh(x, y, z)
    assert faces.shape == (16, 4, 3)


def test_fails_non_1d_array():
    x = np.array([[0.0, 1.0]])  # Not 1D
    y = [0.0, 1.0]
    z = [0.0, 1.0]
    with pytest.raises(ValueError, match="x_coords must be a 1D array."):
        cuboid_mesh(x, y, z)


def test_fails_too_few_values():
    x = [0.0]
    y = [0.0, 1.0]
    z = [0.0, 1.0]
    with pytest.raises(ValueError, match="x_coords must contain at least two values"):
        cuboid_mesh(x, y, z)


def test_fails_non_strictly_increasing():
    x = [0.0, 1.0, 0.5]
    y = [0.0, 1.0]
    z = [0.0, 1.0]
    with pytest.raises(ValueError, match="x_coords must be strictly increasing."):
        cuboid_mesh(x, y, z)


def test_quad_structure_and_ccw_order():
    x = [0.0, 1.0]
    y = [0.0, 1.0]
    z = [0.0, 1.0]
    faces = cuboid_mesh(x, y, z)
    assert faces.shape[1:] == (4, 3)

    z_faces = [f for f in faces if np.allclose(f[:, 2], 0.0)]
    assert len(z_faces) > 0
    for quad in z_faces:
        assert np.allclose(quad[:, 2], quad[0, 2])


def test_large_but_valid_axis():
    x = np.linspace(0, 1, 10)
    y = [0.0, 1.0]
    z = [0.0, 1.0]
    faces = cuboid_mesh(x, y, z)
    assert faces.shape[0] == 38


def test_degenerate_case_single_face_each():
    x = [0.0, 1.0]
    y = [0.0, 1.0]
    z = [0.0, 1.0]
    faces = cuboid_mesh(x, y, z)
    assert faces.shape[0] == 6
    for quad in faces:
        assert quad.shape == (4, 3)


# --------------------------- #
# cuboid_mesh_with_resolution #
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
