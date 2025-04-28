import numpy as np

from .edge import mesh_between_edges, quad_faces_from_edges, rectangle_perimeter


def circumference_edges(radius: float, segment_resolution: int, start_angle: float = 0, counter_clockwise: bool = True) -> np.ndarray:
    """
    Generate the circumference of a circle in 2D space.

    Parameters
    ----------
    radius : float
        Radius of the circle.
    segment_resolution : int
        Number of segments to divide the circle into.
    start_angle : float
        Starting angle for the circumference.
    counter_clockwise : bool, optional
        If True, the angles are generated in a counter-clockwise direction. Default is True.

    Returns
    -------
    circumference : ndarray of shape (2, ...)
        Coordinates of the points on the circumference of the circle.

    Examples
    --------
    >>> from surfmesh import circumference_edges
    >>> # Parameters
    >>> radius = 1.0
    >>> segment_resolution = 12
    >>> start_angle = 0.0
    >>> # Generate the circumference
    >>> circumference = circumference_edges(radius, segment_resolution, start_angle)
    >>> print(circumference.shape)
    (2, 12)
    """
    if radius < 0:
        msg = f"Invalid radius: {radius}. Radius must be non-negative."
        raise ValueError(msg)
    if segment_resolution < 1:
        msg = f"Invalid segment_resolution: {segment_resolution}. Must be a positive integer."
        raise ValueError(msg)

    angles = start_angle + np.linspace(0, 2 * np.pi, segment_resolution * counter_clockwise)
    return np.array([np.cos(angles), np.sin(angles)]) * radius


def disk_mesher_radial(radius: float, radial_resolution: int, segment_resolution: int) -> np.ndarray:
    """
    Generate a 2D circular mesh with curvilinear quadrilateral faces.

    The mesh is constructed using radial and angular divisions, and
    each cell is approximately a curved quadrilateral.

    Parameters
    ----------
    radius : float
        Radius of the disk.
    radial_resolution : int
        Number of divisions along the radial (center-to-edge) direction.
    segment_resolution : int
        Number of divisions around the angular (circular) direction.

    Returns
    -------
    disk_2d_mesh : ndarray of shape (radial_resolution + 1, segment_resolution + 1, 2)
        A 3D NumPy array containing the (x, y) coordinates of each node in the mesh.

    Examples
    --------
    >>> from surfmesh import disk_mesher_radial
    >>> # Parameters
    >>> radial_resolution = 12
    >>> segment_resolution = 12
    >>> radius = 1
    >>> # Create the disk mesh
    >>> disk_2d_mesh = disk_mesher_radial(radius, radial_resolution, segment_resolution).round(6)
    >>> print(disk_2d_mesh.shape)
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


def disk_mesher_square_centered(radius: float, square_resolution: int, radial_resolution: int, square_side_radius_ratio: float = 1, square_disk_rotation: float = 0) -> np.ndarray:
    """
    Generate a quadrilateral mesh for a filled disk using a square core and radial interpolation.

    Parameters
    ----------
    radius : float
        Radius of the disk. Must be positive.
    square_resolution : int
        Number of subdivisions along one side of the square core.
    radial_resolution : int
        Number of radial layers from square to circle.
    square_side_radius_ratio : float
        Ratio of the square radius to the disk radius.
    square_disk_rotation : float, optional
        Circumference rotation angle with respect to central square. Default is 0.
    Returns
    -------
    disk_mesh : ndarray of shape (N, 4, 2)
        Quadrilateral faces covering the disk.

    Examples
    --------
    >>> from surfmesh import disk_mesher_square_centered
    >>> square_resolution = 10
    >>> radial_resolution = 10
    >>> radius = 1
    >>> square_radius_ratio = 0.8
    >>> square_disk_rotation = 0
    >>> disk_mesh = disk_mesher_square_centered(radius, square_resolution, radial_resolution, square_radius_ratio, square_disk_rotation)
    >>> print(disk_mesh.shape)
    (500, 4, 2)

    """
    if radius <= 0:
        msg = f"radius must be positive, got {radius}."
        raise ValueError(msg)
    if square_resolution < 1:
        msg = f"square_resolution must be at least 1, got {square_resolution}."
        raise ValueError(msg)
    if radial_resolution < 1:
        msg = f"radial_resolution must be at least 1, got {radial_resolution}."
        raise ValueError(msg)
    if not (0 < square_side_radius_ratio <= 1):
        msg = f"square_side_radius_ratio must be in (0, 1], got {square_side_radius_ratio}."
        raise ValueError(msg)
    square_side = radius * square_side_radius_ratio
    side_coords = np.linspace(-square_side / 2, square_side / 2, square_resolution + 1)
    square_mesh = quad_faces_from_edges(side_coords, side_coords)

    square_boundary = rectangle_perimeter(side_coords, side_coords)

    segment_resolution = square_resolution * 4 + 1
    circumference = circumference_edges(radius, segment_resolution, start_angle=np.pi / 4 + square_disk_rotation, counter_clockwise=True)

    # Radial interpolation mesh. np.flip is align the face normal with the square mesh.
    radial_edges = np.flip(np.stack([square_boundary, circumference]), axis=1)
    radial_mesh = mesh_between_edges(radial_edges, radial_resolution)

    # Final mesh
    return np.vstack([square_mesh, radial_mesh])
