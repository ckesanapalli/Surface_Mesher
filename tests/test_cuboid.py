import re
import pytest
import numpy as np
from surface_mesher import cuboid_mesh, cuboid_mesh_with_resolution

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
