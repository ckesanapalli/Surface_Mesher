import numpy as np

from .cuboid import quad_faces_from_edges


def circle_mesh_radial(radius: float, radial_resolution: int, segment_resolution: int) -> np.ndarray:
    """
    Generate a 2D circular mesh with curvilinear quadrilateral faces.

    The mesh is constructed using radial and angular divisions, and
    each cell is approximately a curved quadrilateral.

    Parameters
    ----------
    radius : float
        Radius of the circle.
    radial_resolution : int
        Number of divisions along the radial (center-to-edge) direction.
    segment_resolution : int
        Number of divisions around the angular (circular) direction.

    Returns
    -------
    circle_2d_mesh : ndarray of shape (radial_resolution + 1, segment_resolution + 1, 2)
        A 3D NumPy array containing the (x, y) coordinates of each node in the mesh.

    Examples
    --------
    >>> from surface_mesher import circle_mesh_radial
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.patches import Polygon
    >>> from matplotlib.collections import PatchCollection
    >>> # Parameters
    >>> radial_resolution = 12
    >>> segment_resolution = 12
    >>> radius = 1
    >>> # Create the circle mesh
    >>> circle_2d_mesh = circle_mesh_radial(radius, radial_resolution, segment_resolution).round(6)
    >>> print(circle_2d_mesh.shape)
    (144, 4, 2)
    """
    if radius < 0:
        msg = f"Invalid radius: {radius}. Radius must be non-negative."
        raise ValueError(msg)
    if radial_resolution < 1 or segment_resolution < 1:
        msg = f"Invalid resolution: {radial_resolution}, {segment_resolution}. Resolutions must be positive integers."
        raise ValueError(msg)

    radial_divisions = np.linspace(0.0, radius, radial_resolution + 1)
    angular_divisions = np.linspace(0.0, 2.0 * np.pi, segment_resolution + 1)

    polar_faces = quad_faces_from_edges(radial_divisions, angular_divisions)  # shape: (N, 4, 2)

    r = polar_faces[..., 0]
    theta = polar_faces[..., 1]

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return np.stack((r * cos_theta, r * sin_theta), axis=-1)  # (N, 4, 2)
