# tests/test_common.py

import numpy as np
import pytest
from surfmesh import extract_vertices_faces

def test_extract_vertices_faces_quad_mesh():
    """Test basic extraction for a quad mesh."""
    mesh = np.array([
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    ])
    vertices, faces = extract_vertices_faces(mesh)

    assert vertices.shape == (8, 3)
    assert faces.shape == (2, 4)
    assert np.all(faces.max() < len(vertices))


def test_extract_vertices_faces_triangle_mesh():
    """Test extraction for a triangle mesh (3 vertices per face)."""
    mesh = np.array([
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[0, 0, 1], [1, 0, 1], [0, 1, 1]]
    ])
    vertices, faces = extract_vertices_faces(mesh)

    assert vertices.shape == (6, 3)
    assert faces.shape == (2, 3)
    assert np.all(faces.max() < len(vertices))


def test_extract_vertices_faces_duplicate_faces():
    """Test extraction where multiple faces share vertices."""
    mesh = np.array([
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0]]
    ])
    vertices, faces = extract_vertices_faces(mesh)

    assert vertices.shape[1] == 3
    assert faces.shape == (2, 4)
    assert np.all(faces.max() < len(vertices))


def test_extract_vertices_faces_invalid_input_shape():
    """Test error is raised for invalid input shape."""
    mesh = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ])  # 2D instead of 3D

    with pytest.raises(ValueError, match="Expected mesh to have 3 dimensions"):
        extract_vertices_faces(mesh)


def test_extract_vertices_faces_high_dimensional_mesh():
    """Test on a 5D mesh: large structured grid."""
    mesh = np.random.rand(10, 4, 3)
    vertices, faces = extract_vertices_faces(mesh)

    assert vertices.shape[1] == 3
    assert faces.shape == (10, 4)
    assert np.all(faces.max() < len(vertices))


def test_extract_vertices_faces_preserves_structure():
    """Test the faces correctly index back to vertices."""
    mesh = np.array([
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]  # same face repeated
    ])
    vertices, faces = extract_vertices_faces(mesh)

    # Faces should be identical
    assert np.array_equal(faces[0], faces[1])


@pytest.mark.parametrize("mesh_shape", [(5, 4, 3), (20, 3, 3), (1, 4, 2)])
def test_extract_vertices_faces_various_shapes(mesh_shape):
    """Parametrized test for different mesh shapes."""
    mesh = np.random.random(mesh_shape)
    vertices, faces = extract_vertices_faces(mesh)

    assert vertices.ndim == 2
    assert faces.shape[0] == mesh_shape[0]
    assert np.all(faces.max() < len(vertices))

    np.testing.assert_array_equal(mesh, vertices[faces])
