import numpy as np
import pytest

from surfmesh.sphere import sphere_mesher_from_projection, sphere_mesher_from_radial

# ------------------------------ #
# sphere_mesher_from_projection tests
# ------------------------------ #


def test_sphere_projection_basic() -> None:
    radius = 1.0
    resolution = 10
    mesh = sphere_mesher_from_projection(radius, resolution)
    assert isinstance(mesh, np.ndarray)
    assert mesh.ndim == 3
    assert mesh.shape[1:] == (4, 3)
    np.testing.assert_allclose(np.linalg.norm(mesh, axis=2), radius)


def test_sphere_projection_invalid_radius() -> None:
    with pytest.raises(ValueError, match="Radius must be positive"):
        sphere_mesher_from_projection(0, 10)
    with pytest.raises(ValueError, match="Radius must be positive"):
        sphere_mesher_from_projection(-1, 10)


def test_sphere_projection_invalid_resolution() -> None:
    with pytest.raises(ValueError, match="resolution must be at least 1"):
        sphere_mesher_from_projection(1.0, 0)
    with pytest.raises(ValueError, match="resolution must be at least 1"):
        sphere_mesher_from_projection(1.0, -5)


def test_sphere_projection_shape_scaling() -> None:
    radius = 3.5
    resolution = 5
    mesh = sphere_mesher_from_projection(radius, resolution)
    np.testing.assert_allclose(np.linalg.norm(mesh, axis=2), radius)

# ------------------------------ #
# sphere_mesher_from_radial tests
# ------------------------------ #


def test_sphere_radial_basic() -> None:
    radius = 1.0
    radial_resolution = 10
    segment_resolution = 10
    mesh = sphere_mesher_from_radial(radius, radial_resolution, segment_resolution)
    assert isinstance(mesh, np.ndarray)
    assert mesh.ndim == 3
    assert mesh.shape == (radial_resolution * segment_resolution, 4, 3)


def test_sphere_radial_partial_angle() -> None:
    radius = 1.0
    radial_resolution = 5
    segment_resolution = 8
    mesh = sphere_mesher_from_radial(radius, radial_resolution, segment_resolution, start_angle=0, end_angle=np.pi)
    assert mesh.shape == (radial_resolution * segment_resolution, 4, 3)


def test_sphere_radial_invalid_radius() -> None:
    with pytest.raises(ValueError, match="Radius must be positive"):
        sphere_mesher_from_radial(0, 10, 10)


def test_sphere_radial_invalid_radial_resolution() -> None:
    with pytest.raises(ValueError, match="radial_resolution must be at least 1"):
        sphere_mesher_from_radial(1.0, 0, 10)


def test_sphere_radial_invalid_segment_resolution() -> None:
    with pytest.raises(ValueError, match="segment_resolution must be at least 1"):
        sphere_mesher_from_radial(1.0, 10, 0)


@pytest.mark.parametrize("radius", [0.5, 1.0, 10.0])
def test_sphere_radial_radius_variations(radius) -> None:
    mesh = sphere_mesher_from_radial(radius, 5, 8)
    norms = np.linalg.norm(mesh, axis=2)
    np.testing.assert_allclose(norms, radius)

# ------------------------------ #
# randomized stress test
# ------------------------------ #


def test_sphere_projection_randomized() -> None:
    np.random.seed(0)
    radius = np.random.uniform(0.5, 5.0)
    resolution = np.random.randint(3, 20)
    mesh = sphere_mesher_from_projection(radius, resolution)
    assert mesh.shape[1:] == (4, 3)
    assert np.isfinite(mesh).all()


def test_sphere_radial_randomized() -> None:
    np.random.seed(1)
    radius = np.random.uniform(0.5, 5.0)
    radial_res = np.random.randint(3, 15)
    segment_res = np.random.randint(5, 20)
    mesh = sphere_mesher_from_radial(radius, radial_res, segment_res)
    assert mesh.shape == (radial_res * segment_res, 4, 3)
    assert np.isfinite(mesh).all()
