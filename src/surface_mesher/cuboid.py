"""This file contains the class and functions to create a cuboid surface mesh."""

from dataclasses import dataclass, field

import numpy as np

from numpy.typing import ArrayLike

def generate_face_quads(u_coords: ArrayLike,
                        v_coords: ArrayLike,
                        fixed_axis: int,
                        fixed_value: float) -> np.ndarray:
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
    >>> from surface_mesher.cuboid import generate_face_quads
    >>> u = np.array([0.0, 1.0])
    >>> v = np.array([0.0, 1.0])
    >>> quads = generate_face_quads(u, v, fixed_axis=2, fixed_value=0.0)
    >>> print(quads.shape)
    (1, 4, 3)
    >>> print(quads[0])
    [[0. 0. 0.]
     [1. 0. 0.]
     [1. 1. 0.]
     [0. 1. 0.]]
    """
    uu, vv = np.meshgrid(u_coords, v_coords, indexing='ij')

    # Vertices ordered counter-clockwise:
    # p0 = bottom-left
    # p1 = bottom-right
    # p2 = top-right
    # p3 = top-left
    p0 = np.stack([uu[:-1, :-1], vv[:-1, :-1]], axis=-1).reshape(-1, 2)
    p1 = np.stack([uu[1:, :-1],  vv[1:, :-1]], axis=-1).reshape(-1, 2)
    p2 = np.stack([uu[1:, 1:],   vv[1:, 1:]], axis=-1).reshape(-1, 2)
    p3 = np.stack([uu[:-1, 1:],  vv[:-1, 1:]], axis=-1).reshape(-1, 2)

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

def generate_axis_coords(start: float, length: float, step: float) -> np.ndarray:
    """
    Generate 1D coordinate array from `start` to `start + length` using a step size,
    adjusting the step to fit evenly based on the nearest number of intervals.

    Parameters
    ----------
    start : float
        The starting coordinate.
    length : float
        The total length of the interval.
    step : float
        Approximate spacing between coordinates (will be adjusted slightly).

    Returns
    -------
    coords : np.ndarray
        1D array of coordinates from start to start + length (inclusive).

    Examples
    --------
    >>> generate_axis_coords(0.0, 10.0, 3.0)
    array([ 0.,  3.33333333,  6.66666667, 10.])
    >>> generate_axis_coords(0.0, 10.0, 2.0)
    array([ 0.,  2.,  4.,  6.,  8., 10.])
    """
    if step <= 0:
        raise ValueError("step must be a positive number.")
    if length <= 0:
        raise ValueError("length must be a positive number.")

    num_steps = max(1, np.round(length / step).astype(int))
    coords = np.linspace(start, start + length, num_steps + 1)
    return coords


def generate_cuboid_surface(
    x_coords: ArrayLike, 
    y_coords: ArrayLike,
    z_coords: ArrayLike
) -> np.ndarray:
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
    faces : np.ndarray
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
    >>> faces = create_mesh_from_coordinate_arrays(x, y, z)
    >>> print(faces.shape)
    (16, 4, 3)  # 6 faces total from the cuboid
    """
    # Convert to NumPy arrays and validate
    x_coords = np.asarray(x_coords, dtype=float)
    y_coords = np.asarray(y_coords, dtype=float)
    z_coords = np.asarray(z_coords, dtype=float)

    for name, coords in zip(("x_coords", "y_coords", "z_coords"), (x_coords, y_coords, z_coords)):
        if coords.ndim != 1:
            raise ValueError(f"{name} must be a 1D array.")
        if coords.size < 2:
            raise ValueError(f"{name} must contain at least two values to form quads.")
        if not np.all(np.diff(coords) > 0):
            raise ValueError(f"{name} must be strictly increasing.")

    faces = []

    # XY planes (bottom and top)
    faces.append(generate_face_quads(x_coords, y_coords, fixed_axis=2, fixed_value=z_coords[0]))
    faces.append(generate_face_quads(x_coords, y_coords, fixed_axis=2, fixed_value=z_coords[-1]))

    # XZ planes (front and back)
    faces.append(generate_face_quads(x_coords, z_coords, fixed_axis=1, fixed_value=y_coords[0]))
    faces.append(generate_face_quads(x_coords, z_coords, fixed_axis=1, fixed_value=y_coords[-1]))

    # YZ planes (left and right)
    faces.append(generate_face_quads(y_coords, z_coords, fixed_axis=0, fixed_value=x_coords[0]))
    faces.append(generate_face_quads(y_coords, z_coords, fixed_axis=0, fixed_value=x_coords[-1]))

    return np.concatenate(faces, axis=0)


@dataclass
class Cuboid:
    """
    Class to create a cuboid surface mesh.

    Attributes
    ----------
    length : float
        The length of the cuboid.
    width : float
        The width of the cuboid.
    height : float
        The height of the cuboid.
    origin : np.ndarray
        The origin point of the cuboid in 3D space.
    """
    length: float
    width: float
    height: float
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def create_mesh_from_edge_sizes(self, mesh_sizes: float | ArrayLike) -> np.ndarray:
        """
        Create a 3D surface mesh of the cuboid using specified edge sizes along each axis.

        Parameters
        ----------
        mesh_sizes : float | Arraylike of 3 floats
            Edge size(s) for mesh generation. Can be a single float or a sequence of three values
            for the x, y, and z directions.

        Returns
        -------
        faces : np.ndarray
            An array of shape (N, 4, 3), each representing a quadrilateral face of the cuboid.

        Examples
        --------
        >>> from surface_mesher.cuboid import Cuboid
        >>> cuboid = Cuboid(length=2.0, width=1.0, height=1.0)
        >>> faces = cuboid.create_mesh_from_edge_sizes(0.5)
        >>> print(faces.shape)
        (24, 4, 3)  # Number of quads: 6 faces * 2x1 divisions each
        """
        mesh_sizes = np.array(mesh_sizes, dtype=float)

        # Normalize input to (3,) shape
        if mesh_sizes.ndim == 0:
            mesh_sizes = np.full(3, mesh_sizes)
        elif mesh_sizes.shape != (3,):
            raise ValueError("mesh_sizes must be a single float or an array of three floats.")

        if np.any(mesh_sizes <= 0):
            raise ValueError("mesh_sizes must be positive values.")

        if np.any(mesh_sizes > np.array([self.length, self.width, self.height])):
            raise ValueError("mesh_sizes must be less than or equal to the cuboid dimensions.")

        # Generate coordinate arrays based on origin and dimensions
        ox, oy, oz = self.origin
        x_coords = generate_axis_coords(-self.length/2.0, self.length, mesh_sizes[0]) + ox
        y_coords = generate_axis_coords(-self.width/2.0, self.width, mesh_sizes[1]) + oy
        z_coords = generate_axis_coords(-self.height/2.0, self.height, mesh_sizes[2]) + oz

        # Use external utility to generate surface mesh
        return generate_cuboid_surface(x_coords, y_coords, z_coords)

    def create_mesh_with_resolution(self, resolution: int | ArrayLike) -> np.ndarray:
        """
        Create a 3D mesh of the cuboid as quads (N, 4, 3), where each face is a quadrilateral.

        Parameters
        ----------
        resolution : int | Arraylike of 3 integers
            The number of divisions along each axis. If a single int is given, it is used as the resolution.

        Returns
        -------
        faces : np.ndarray
            An array of shape (N, 4, 3), where each item is a quad defined by 4 3D points.
        """
        resolution = np.array(resolution, dtype=int)

        # Normalize to (3,) shape
        if resolution.ndim == 0:
            resolution = np.full(3, resolution)
        elif len(resolution) != 3:
            raise ValueError("divisions must be a single int or an array of three ints.")

        if np.any(resolution <= 0):
            raise ValueError("divisions must be positive values.")

        edge_sizes = np.array([self.length, self.width, self.height]) / resolution

        return self.create_mesh_from_edge_sizes(edge_sizes)
