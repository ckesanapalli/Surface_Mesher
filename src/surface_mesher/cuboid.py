"""This file contains the class and functions to create a cuboid surface mesh."""

import numpy as np
from numpy.typing import ArrayLike


def quad_faces_from_grid(u_coords: ArrayLike, v_coords: ArrayLike, fixed_axis: int, fixed_value: float) -> np.ndarray:
    """
    Generate quadrilateral faces on a grid where one axis is fixed,
    with counter-clockwise vertex ordering.

    Parameters
    ----------
    u_coords : ArrayLike
        Coordinates along the first varying axis (horizontal direction).
    v_coords : ArrayLike
        Coordinates along the second varying axis (vertical direction).
    fixed_axis : int
        The index of the fixed axis (0 = x, 1 = y, 2 = z).
    fixed_value : float
        The fixed coordinate value for the constant axis.

    Returns
    -------
    quads_3d : np.ndarray
        Shape (N, 4, 3) array of 3D quad vertices, with each face ordered counter-clockwise.

    Examples
    --------
    >>> import numpy as np
    >>> from surface_mesher.cuboid import quad_faces_from_grid
    >>> u = np.array([0.0, 1.0])
    >>> v = np.array([0.0, 1.0])
    >>> quads = quad_faces_from_grid(u, v, fixed_axis=2, fixed_value=0.0)
    >>> print(quads.shape)
    (1, 4, 3)
    >>> print(quads[0])
    [[0. 0. 0.]
     [1. 0. 0.]
     [1. 1. 0.]
     [0. 1. 0.]]
    """

    uu, vv = np.meshgrid(u_coords, v_coords, indexing="ij")

    # Vertices ordered counter-clockwise:
    # p0 = bottom-left
    # p1 = bottom-right
    # p2 = top-right
    # p3 = top-left
    p0 = np.stack([uu[:-1, :-1], vv[:-1, :-1]], axis=-1).reshape(-1, 2)
    p1 = np.stack([uu[1:, :-1], vv[1:, :-1]], axis=-1).reshape(-1, 2)
    p2 = np.stack([uu[1:, 1:], vv[1:, 1:]], axis=-1).reshape(-1, 2)
    p3 = np.stack([uu[:-1, 1:], vv[:-1, 1:]], axis=-1).reshape(-1, 2)

    # Stack into quads: [p0, p1, p2, p3]
    quad_2d = np.stack([p0, p1, p2, p3], axis=1)

    # Insert into 3D
    quads_3d = np.zeros((quad_2d.shape[0], 4, 3))
    axes = [0, 1, 2]
    axes.remove(fixed_axis)

    quads_3d[:, :, axes[0]] = quad_2d[:, :, 0]
    quads_3d[:, :, axes[1]] = quad_2d[:, :, 1]
    quads_3d[:, :, fixed_axis] = fixed_value

    return quads_3d


def cuboid_mesh(x_coords: ArrayLike, y_coords: ArrayLike, z_coords: ArrayLike) -> np.ndarray:
    """
    Generate a full cuboid surface mesh using explicit coordinate arrays along each axis.

    This function creates quadrilateral surface faces based on 1D coordinate arrays for the
    x, y, and z axes. The surface mesh includes all six sides of the cuboid spanned by
    the given coordinates. Vertex order for each quad is counter-clockwise.

    Parameters
    ----------
    x_coords : ArrayLike of float
        1D strictly increasing array of x-axis positions for vertical planes (YZ-facing).
    y_coords : ArrayLike of float
        1D strictly increasing array of y-axis positions for horizontal planes (XZ-facing).
    z_coords : ArrayLike of float
        1D strictly increasing array of z-axis positions for depth planes (XY-facing).

    Returns
    -------
    np.ndarray
        Array of shape (N, 4, 3), where N is the number of quadrilateral faces.
        Each face is defined by four 3D points in counter-clockwise order.

    Raises
    ------
    ValueError
        If any coordinate array is not 1D, has fewer than 2 elements,
        or is not strictly increasing.

    Examples
    --------
    >>> x = [0.0, 1.0, 2.0]
    >>> y = [0.0, 1.0]
    >>> z = [0.0, 0.5, 1.0]
    >>> faces = cuboid_mesh(x, y, z)
    >>> print(faces.shape)  # 6 faces total from the cuboid
    (16, 4, 3)
    """

    # Convert to NumPy arrays and validate
    x_coords = np.asarray(x_coords, dtype=float)
    y_coords = np.asarray(y_coords, dtype=float)
    z_coords = np.asarray(z_coords, dtype=float)

    for name, coords in zip(
        ("x_coords", "y_coords", "z_coords"),
        (x_coords, y_coords, z_coords),
        strict=False,
    ):
        if coords.ndim != 1:
            scalar_error_msg = f"{name} must be a 1D array."
            raise ValueError(scalar_error_msg)
        if coords.size < 2:
            size_error_msg = f"{name} must contain at least two values to form quads."
            raise ValueError(size_error_msg)
        if not np.all(np.diff(coords) > 0):
            increasing_error_msg = f"{name} must be strictly increasing."
            raise ValueError(increasing_error_msg)

    faces = [
        # XY planes (bottom and top)
        quad_faces_from_grid(x_coords, y_coords, fixed_axis=2, fixed_value=z_coords[0]),
        quad_faces_from_grid(x_coords, y_coords, fixed_axis=2, fixed_value=z_coords[-1]),
        # XZ planes (front and back)
        quad_faces_from_grid(x_coords, z_coords, fixed_axis=1, fixed_value=y_coords[0]),
        quad_faces_from_grid(x_coords, z_coords, fixed_axis=1, fixed_value=y_coords[-1]),
        # YZ planes (left and right)
        quad_faces_from_grid(y_coords, z_coords, fixed_axis=0, fixed_value=x_coords[0]),
        quad_faces_from_grid(y_coords, z_coords, fixed_axis=0, fixed_value=x_coords[-1]),
    ]

    return np.concatenate(faces, axis=0)


def cuboid_mesh_with_resolution(
    length: float, width: float, height: float, origin: tuple[float, float, float] = (0.0, 0.0, 0.0), resolution: int | tuple[int, int, int] = (1, 1, 1)
) -> np.ndarray:
    """
    Generate a 3D surface mesh of a cuboid with quadrilateral faces based on resolution.

    Parameters
    ----------
    length : float
        Length of the cuboid along the x-axis.
    width : float
        Width of the cuboid along the y-axis.
    height : float
        Height of the cuboid along the z-axis.
    origin : tuple of 3 floats, optional
        Center point of the cuboid in 3D space. Default is (0.0, 0.0, 0.0).
    resolution : int or tuple of 3 ints
        Number of subdivisions along each axis. If a single int is provided,
        it's used for all axes.

    Returns
    -------
    np.ndarray
        Surface mesh of shape (N, 4, 3), where N is the number of quad faces.

    Examples
    --------
    >>> from surface_mesher import cuboid_mesh_with_resolution
    >>> mesh = cuboid_mesh_with_resolution(2.0, 1.0, 1.0, resolution=2)
    >>> mesh.shape
    (24, 4, 3)

    >>> mesh = cuboid_mesh_with_resolution(2.0, 1.0, 1.0, resolution=[2, 1, 2])
    >>> mesh.shape[1:]
    (4, 3)
    """
    resolution = np.array(resolution, dtype=int)

    if resolution.ndim == 0:
        resolution = np.full(3, resolution)
    elif resolution.shape != (3,):
        msg = "resolution must be a single int or an array-like of three ints."
        raise ValueError(msg)
    if np.any(resolution <= 0):
        msg = "resolution must contain only positive values."
        raise ValueError(msg)

    res_x, res_y, res_z = resolution
    ox, oy, oz = origin

    x_coords = np.linspace(-length / 2.0 + ox, length / 2.0 + ox, res_x + 1)
    y_coords = np.linspace(-width / 2.0 + oy, width / 2.0 + oy, res_y + 1)
    z_coords = np.linspace(-height / 2.0 + oz, height / 2.0 + oz, res_z + 1)

    return cuboid_mesh(x_coords, y_coords, z_coords)
