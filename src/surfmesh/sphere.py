import numpy as np

from .cuboid import cuboid_mesher_with_resolution
from .revolve import circular_revolve


def sphere_mesher_from_projection(radius: float, resolution: int) -> np.ndarray:
    """
    Generate a quadrilateral mesh approximating a sphere using cube projection.

    The method starts by creating a subdivided cube mesh, then normalizes and scales
    it onto the sphere surface.

    Parameters
    ----------
    radius : float
        Radius of the resulting sphere.
    resolution : int
        Number of divisions along each cube edge before projecting onto the sphere.

    Returns
    -------
    sphere_mesh : np.ndarray
        Array of shape (n_faces, 4, 3), where each face is a quad with 3D coordinates.

    Examples
    --------
    >>> from surfmesh import sphere_mesher_from_projection
    >>> mesh = sphere_mesher_from_projection(radius=1.0, resolution=10)
    >>> mesh.shape
    (600, 4, 3)

    """
    if radius <= 0:
        msg = f"Radius must be positive. Got {radius}"
        raise ValueError(msg)

    if resolution < 1:
        msg = f"resolution must be at least 1. Got {resolution}"
        raise ValueError(msg)

    cube_mesh = cuboid_mesher_with_resolution(radius, radius, radius, resolution=resolution)

    # Normalize each vertex to lie on the sphere
    norms = np.linalg.norm(cube_mesh, axis=2, keepdims=True)
    return radius * cube_mesh / norms


def sphere_mesher_from_radial(
    radius: float,
    radial_resolution: int,
    segment_resolution: int,
    start_angle: float = 0,
    end_angle: float = 2 * np.pi,
) -> np.ndarray:
    """
    Generate a mesh approximating a sphere using radial divisions.

    The method uses spherical coordinates to create the mesh.

    Parameters
    ----------
    radius : float
        Radius of the resulting sphere.
    radial_resolution : int
        Number of divisions along the radial direction (latitude).
    segment_resolution : int
        Number of divisions around the circumference (longitude).

    Returns
    -------
    sphere_mesh : np.ndarray
        Array of shape (n_faces, 4, 3), in 3D coordinates.

    Examples
    --------
    >>> from surfmesh import sphere_mesher_from_radial
    >>> mesh = sphere_mesher_from_radial(radius=1.0, radial_resolution=10, segment_resolution=10)
    >>> mesh.shape
    (100, 4, 3)

    """
    if radius <= 0:
        msg = f"Radius must be positive. Got {radius}"
        raise ValueError(msg)

    if radial_resolution < 1:
        msg = f"radial_resolution must be at least 1. Got {radial_resolution}"
        raise ValueError(msg)

    if segment_resolution < 1:
        msg = f"segment_resolution must be at least 1. Got {segment_resolution}"
        raise ValueError(msg)

    angles = np.linspace(-np.pi / 2, np.pi / 2, radial_resolution + 1)
    curve = radius * np.array([np.cos(angles), np.sin(angles)]).T
    return circular_revolve(curve, segment_resolution, start_angle=start_angle, end_angle=end_angle)
