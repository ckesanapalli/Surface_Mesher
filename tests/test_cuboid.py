import pytest
import numpy as np
from surface_mesher.cuboid import (
    generate_cuboid_surface,
    generate_axis_coords,
    generate_face_quads,
    Cuboid,
)


def test_generate_axis_coords():
    # Basic test cases
    coords = generate_axis_coords(0.0, 10.0, 3.0)
    assert np.allclose(coords, [0.0, 3.33333333, 6.66666667, 10.0])

    coords = generate_axis_coords(0.0, 10.0, 2.0)
    assert np.array_equal(coords, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])

    # Edge case: step exactly matches length
    coords = generate_axis_coords(0.0, 5.0, 5.0)
    assert np.array_equal(coords, [0.0, 5.0])

    # Edge case: zero step (should raise ValueError)
    with pytest.raises(ValueError, match="step must be a positive number."):
        generate_axis_coords(0.0, 10.0, 0.0)

    # Edge case: negative length (should raise ValueError)
    with pytest.raises(ValueError, match="length must be a positive number."):
        generate_axis_coords(0.0, -10.0, 2.0)


def test_generate_face_quads_basic():
    u = np.array([0, 1])
    v = np.array([0, 1])
    result = generate_face_quads(u, v, fixed_axis=2, fixed_value=5.0)

    expected = np.array([[[0, 0, 5], [1, 0, 5], [1, 1, 5], [0, 1, 5]]])
    np.testing.assert_array_equal(result, expected)


def test_generate_face_quads_valid_fixed_axes():
    u = np.array([0, 1])
    v = np.array([0, 1])
    for axis in [0, 1, 2]:
        result = generate_face_quads(u, v, fixed_axis=axis, fixed_value=7.0)
        assert result.shape == (1, 4, 3)
        assert np.all(result[:, :, axis] == 7.0)


def test_generate_face_quads_zero_area():
    u = np.array([0])
    v = np.array([0])
    result = generate_face_quads(u, v, 0, 0)
    assert result.shape == (0, 4, 3)


def test_generate_face_quads_rectangular_grid():
    u = np.array([0, 1, 2])
    v = np.array([0, 1])
    result = generate_face_quads(u, v, 2, 3)
    assert result.shape == (2, 4, 3)


# ---------------------- #
# VALID MESH GENERATION  #
# ---------------------- #


def test_valid_mesh_basic():
    x = [0.0, 1.0]
    y = [0.0, 1.0]
    z = [0.0, 1.0]
    faces = generate_cuboid_surface(x, y, z)

    assert faces.shape == (6, 4, 3)
    assert np.isclose(faces[:, :, 2].min(), 0.0)
    assert np.isclose(faces[:, :, 2].max(), 1.0)


def test_valid_mesh_multiple_cells():
    x = [0.0, 1.0, 2.0]
    y = [0.0, 0.5, 1.0]
    z = [0.0, 0.5, 1.0]
    faces = generate_cuboid_surface(x, y, z)

    # Expect:
    # XY faces: (2 x 2) x 2 = 8
    # XZ faces: (2 x 2) x 2 = 8
    # YZ faces: (2 x 2) x 2 = 8
    assert faces.shape == (24, 4, 3)


# ---------------------- #
# INPUT TYPE FLEXIBILITY #
# ---------------------- #


@pytest.mark.parametrize("container_type", [list, tuple, np.array])
def test_accepts_all_arraylike_types(container_type):
    x = container_type([0.0, 1.0, 2.0])
    y = container_type([0.0, 1.0])
    z = container_type([0.0, 0.5, 1.0])
    faces = generate_cuboid_surface(x, y, z)
    assert faces.shape == (16, 4, 3)


# --------------------------- #
# INVALID: COORD VALIDATION   #
# --------------------------- #


def test_fails_non_1d_array():
    x = np.array([[0.0, 1.0]])  # Not 1D
    y = [0.0, 1.0]
    z = [0.0, 1.0]
    with pytest.raises(ValueError, match="x_coords must be a 1D array."):
        generate_cuboid_surface(x, y, z)


def test_fails_too_few_values():
    x = [0.0]
    y = [0.0, 1.0]
    z = [0.0, 1.0]
    with pytest.raises(ValueError, match="x_coords must contain at least two values"):
        generate_cuboid_surface(x, y, z)


def test_fails_non_strictly_increasing():
    x = [0.0, 1.0, 0.5]  # Not strictly increasing
    y = [0.0, 1.0]
    z = [0.0, 1.0]
    with pytest.raises(ValueError, match="x_coords must be strictly increasing."):
        generate_cuboid_surface(x, y, z)


# ----------------------------- #
# GEOMETRY & VERTEX STRUCTURE   #
# ----------------------------- #


def test_quad_structure_and_ccw_order():
    x = [0.0, 1.0]
    y = [0.0, 1.0]
    z = [0.0, 1.0]
    faces = generate_cuboid_surface(x, y, z)

    # Check each face has 4 vertices in 3D
    assert faces.shape[1:] == (4, 3)

    # Check that quads lie on expected planes
    z_faces = [f for f in faces if np.allclose(f[:, 2], 0.0)]
    assert len(z_faces) > 0
    for quad in z_faces:
        assert np.allclose(quad[:, 2], quad[0, 2])


# ----------------------------- #
# EDGE CASES                    #
# ----------------------------- #


def test_large_but_valid_axis():
    x = np.linspace(0, 1, 10)  # 9 segments
    y = [0.0, 1.0]  # 1 segment
    z = [0.0, 1.0]  # 1 segment
    faces = generate_cuboid_surface(x, y, z)
    assert faces.shape[0] == 38  # Correct: 18 + 18 + 2


def test_degenerate_case_single_face_each():
    x = [0.0, 1.0]
    y = [0.0, 1.0]
    z = [0.0, 1.0]
    faces = generate_cuboid_surface(x, y, z)
    assert faces.shape[0] == 6  # One quad per face
    for quad in faces:
        assert quad.shape == (4, 3)


def test_create_mesh_from_edge_sizes_valid_float():
    cuboid = Cuboid(length=2.0, width=1.0, height=1.0)
    faces = cuboid.create_mesh_from_edge_sizes(0.5)
    assert isinstance(faces, np.ndarray)
    assert faces.shape == (40, 4, 3)


def test_create_mesh_from_edge_sizes_valid_arraylike():
    cuboid = Cuboid(length=2.0, width=1.0, height=1.0)
    faces = cuboid.create_mesh_from_edge_sizes([0.5, 0.5, 0.5])
    assert isinstance(faces, np.ndarray)
    assert faces.shape == (40, 4, 3)


def test_create_mesh_from_edge_sizes_invalid_shape():
    cuboid = Cuboid(length=2.0, width=1.0, height=1.0)
    with pytest.raises(
        ValueError,
        match="mesh_sizes must be a single float or an array of three floats.",
    ):
        cuboid.create_mesh_from_edge_sizes([0.5, 0.5])


def test_create_mesh_from_edge_sizes_nonpositive():
    cuboid = Cuboid(length=2.0, width=1.0, height=1.0)
    with pytest.raises(ValueError, match="mesh_sizes must be positive values."):
        cuboid.create_mesh_from_edge_sizes([0.5, -0.5, 0.5])


def test_create_mesh_from_edge_sizes_too_large():
    cuboid = Cuboid(length=2.0, width=1.0, height=1.0)
    with pytest.raises(
        ValueError,
        match="mesh_sizes must be less than or equal to the cuboid dimensions.",
    ):
        cuboid.create_mesh_from_edge_sizes([3.0, 0.5, 0.5])


def test_create_mesh_with_resolution_scalar():
    cuboid = Cuboid(length=2.0, width=1.0, height=1.0)
    faces = cuboid.create_mesh_with_resolution(2)
    assert faces.shape == (24, 4, 3)


def test_create_mesh_with_resolution_arraylike():
    cuboid = Cuboid(length=2.0, width=1.0, height=1.0)
    faces = cuboid.create_mesh_with_resolution([2, 1, 2])
    assert isinstance(faces, np.ndarray)
    assert faces.shape[1:] == (4, 3)


def test_create_mesh_with_resolution_invalid_shape():
    cuboid = Cuboid(length=2.0, width=1.0, height=1.0)
    with pytest.raises(
        ValueError, match="divisions must be a single int or an array of three ints."
    ):
        cuboid.create_mesh_with_resolution([2, 2])


def test_create_mesh_with_resolution_nonpositive():
    cuboid = Cuboid(length=2.0, width=1.0, height=1.0)
    with pytest.raises(ValueError, match="divisions must be positive values."):
        cuboid.create_mesh_with_resolution([2, 0, 2])
