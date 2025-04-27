import re
import pytest
import numpy as np
from surfmesh import (
    convert_2d_face_to_3d,
    quad_faces_from_edges,
    mesh_between_edges,
    rectangle_perimeter,
)

# --------------------------- #
# convert_2d_face_to_3d Tests  #
# --------------------------- #

def test_convert_2d_face_to_3d_basic():
    quad_2d = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]])
    result = convert_2d_face_to_3d(quad_2d, axis=2, offset=5.0)
    expected = np.array([[[0, 0, 5], [1, 0, 5], [1, 1, 5], [0, 1, 5]]])
    np.testing.assert_array_equal(result, expected)

def test_convert_2d_face_to_3d_all_axes():
    quad_2d = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]])
    for axis in [0, 1, 2]:
        result = convert_2d_face_to_3d(quad_2d, axis=axis, offset=3.5)
        assert result.shape == (1, 4, 3)
        assert np.allclose(result[:, :, axis], 3.5)

def test_convert_2d_face_to_3d_invalid_axis():
    quad_2d = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]])
    with pytest.raises(ValueError, match=re.escape("fixed_axis must be 0 (x), 1 (y), or 2 (z). Got 3.")):
        convert_2d_face_to_3d(quad_2d, axis=3, offset=0)

# --------------------------- #
# quad_faces_from_edges Tests #
# --------------------------- #

def test_quad_faces_from_edges_basic():
    u = np.array([0, 1])
    v = np.array([0, 1])
    result = quad_faces_from_edges(u, v)
    expected = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]])
    np.testing.assert_array_equal(result, expected)

def test_quad_faces_from_edges_rectangular_grid():
    u = np.array([0, 1, 2])
    v = np.array([0, 1])
    result = quad_faces_from_edges(u, v)
    assert result.shape == (2, 4, 2)

def test_quad_faces_from_edges_zero_area():
    u = np.array([0])
    v = np.array([0])
    result = quad_faces_from_edges(u, v)
    assert result.shape == (0, 4, 2)

# --------------------------- #
# mesh_between_edges Tests    #
# --------------------------- #

def test_mesh_between_edges_basic():
    edge_start = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T
    edge_end = np.array([[0, 0.5], [0.5, 0], [1, 0.5], [0.5, 1]]).T
    edges = np.stack([edge_start, edge_end])
    result = mesh_between_edges(edges, radial_resolution=3)
    assert result.shape == (4 * 2 + 1, 4, 2)  # (n_segments * (radial_resolution-1), 4, 2)

def test_mesh_between_edges_invalid_shape():
    edge_start = np.array([0, 0, 0])
    edge_end = np.array([0, 0.5, 0])
    edges = np.stack([edge_start, edge_end], axis=1)  # wrong shape (2, 2, 2) → should be (2, N, 2)
    with pytest.raises(ValueError, match=re.escape("edges must have shape (2, axis_count, vertex_count)")):
        mesh_between_edges(edges, radial_resolution=3)

def test_mesh_between_edges_invalid_resolution():
    edge_start = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    edge_end = np.array([[0, 0.5], [0.5, 0], [1, 0.5], [0.5, 1]])
    edges = np.stack([edge_start, edge_end])
    with pytest.raises(ValueError, match="radial_resolution must be at least 1"):
        mesh_between_edges(edges, radial_resolution=0)

# --------------------------- #
# rectangle_perimeter Tests   #
# --------------------------- #

def test_rectangle_perimeter_basic():
    length_edge = np.array([0.0, 1.0])
    width_edge = np.array([0.0, 1.0])
    result = rectangle_perimeter(length_edge, width_edge)
    assert result.shape == (2, 5)

def test_rectangle_perimeter_invalid_dimensions():
    length_edge = np.array([[0, 1]])
    width_edge = np.array([0, 1])
    with pytest.raises(ValueError, match="length_edge and width_edge must have the same shape."):
        rectangle_perimeter(length_edge, width_edge)

def test_rectangle_perimeter_too_few_points():
    length_edge = np.array([0.0])
    width_edge = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="length_edge and width_edge must have at least 2 points"):
        rectangle_perimeter(length_edge, width_edge)