"""This file contains the class and functions to create a cuboid surface mesh."""

import numpy as np
from numpy.typing import ArrayLike


def convert_2d_face_to_3d(quad_2d_mesh: np.ndarray, fixed_axis: int, fixed_value: float) -> np.ndarray:
    """
    Convert a 2D quadrilateral mesh to a 3D mesh by adding a fixed coordinate.
    """
    face_count = quad_2d_mesh.shape[0]
    quads_3d_mesh = np.empty((face_count, 4, 3), dtype=float)

    match fixed_axis:
        case 0:
            quads_3d_mesh[:, :, 0] = fixed_value
            quads_3d_mesh[:, :, 1:] = quad_2d_mesh
        case 1:
            quads_3d_mesh[:, :, 1] = fixed_value
            quads_3d_mesh[:, :, 0] = quad_2d_mesh[:, :, 0]
            quads_3d_mesh[:, :, 2] = quad_2d_mesh[:, :, 1]
        case 2:
            quads_3d_mesh[:, :, 2] = fixed_value
            quads_3d_mesh[:, :, :2] = quad_2d_mesh
        case _:
            axis_error_msg = f"fixed_axis must be 0 (x), 1 (y), or 2 (z). Got {fixed_axis}."
            raise ValueError(axis_error_msg)

    return quads_3d_mesh


def quad_faces_from_edges(u_coords: ArrayLike, v_coords: ArrayLike) -> np.ndarray:
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
    np.ndarray
        Shape (N, 4, 2) array of 2D quad vertices, with each face ordered counter-clockwise.

    Examples
    --------
    >>> import numpy as np
    >>> from surface_mesher.cuboid import quad_faces_from_edges
    >>> u = np.array([0.0, 1.0])
    >>> v = np.array([0.0, 1.0])
    >>> quads = quad_faces_from_edges(u, v)
    >>> print(quads.shape)
    (1, 4, 2)
    >>> print(quads[0])
    [[0. 0.]
     [1. 0.]
     [1. 1.]
     [0. 1.]]
    """

    # Meshgrid + quad generation
    uu, vv = np.meshgrid(u_coords, v_coords, indexing="ij")

    corners = [
        (uu[:-1, :-1], vv[:-1, :-1]),  # bottom-left
        (uu[1:, :-1], vv[1:, :-1]),  # bottom-right
        (uu[1:, 1:], vv[1:, 1:]),  # top-right
        (uu[:-1, 1:], vv[:-1, 1:]),  # top-left
    ]

    return np.stack([np.stack([x, y], axis=-1).reshape(-1, 2) for x, y in corners], axis=1)


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
    coords = [np.asarray(c, dtype=float) for c in (x_coords, y_coords, z_coords)]

    for name, arr in zip(("x", "y", "z"), coords, strict=False):
        if arr.ndim != 1:
            msg = f"{name}_coords must be 1D, got shape {arr.shape}."
            raise ValueError(msg)
        if arr.size < 2:
            msg = f"{name}_coords must have at least 2 points, got {arr.size}."
            raise ValueError(msg)
        if not np.all(np.diff(arr) > 0):
            msg = f"{name}_coords must be strictly increasing."
            raise ValueError(msg)

    x, y, z = coords

    xy = quad_faces_from_edges(x, y)
    yz = quad_faces_from_edges(y, z)
    zx = quad_faces_from_edges(z, x)

    xf0, xf1 = x[0], x[-1]
    yf0, yf1 = y[0], y[-1]
    zf0, zf1 = z[0], z[-1]

    return np.concatenate(
        [
            convert_2d_face_to_3d(xy, fixed_axis=2, fixed_value=zf0),
            convert_2d_face_to_3d(xy, fixed_axis=2, fixed_value=zf1),
            convert_2d_face_to_3d(yz, fixed_axis=0, fixed_value=xf0),
            convert_2d_face_to_3d(yz, fixed_axis=0, fixed_value=xf1),
            convert_2d_face_to_3d(zx, fixed_axis=1, fixed_value=yf0),
            convert_2d_face_to_3d(zx, fixed_axis=1, fixed_value=yf1),
        ],
        axis=0,
    )


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

    x_edge_point = ox + length / 2.0
    y_edge_point = oy + width / 2.0
    z_edge_point = oz + height / 2.0

    x_coords = np.linspace(-x_edge_point, x_edge_point, res_x + 1)
    y_coords = np.linspace(-y_edge_point, y_edge_point, res_y + 1)
    z_coords = np.linspace(-z_edge_point, z_edge_point, res_z + 1)

    return cuboid_mesh(x_coords, y_coords, z_coords)
