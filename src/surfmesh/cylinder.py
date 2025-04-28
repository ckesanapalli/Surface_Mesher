"""
Created on Sun Apr 27 16:45:52 2025

@author: kccho
"""

import numpy as np

from .disk import disk_mesher_radial, disk_mesher_square_centered
from .edge import convert_2d_face_to_3d
from .revolve import circular_revolve


def cylinder_mesher_radial(radius: float, height: float, radial_resolution: int, segment_resolution: int, height_resolution: int) -> np.ndarray:
    """
    Generate a triangular mesh for a closed cylinder.

    Parameters
    ----------
    radius : float
        Radius of the cylinder.
    height : float
        Height of the cylinder.
    radial_resolution : int
        Number of radial divisions for the disks.
    segment_resolution : int
        Number of segments around the circumference.
    height_resolution : int
        Number of divisions along the height.

    Returns
    -------
    mesh : ndarray of shape (n_faces, 4, 3)
        Stack of 3D faces representing the bottom disk, top disk, and lateral surface.

    Examples
    --------
    >>> radius = 1.0
    >>> height = 2.0
    >>> radial_res = 8
    >>> segment_res = 16
    >>> height_res = 10
    >>> mesh = cylinder_mesher_radial(radius, height, radial_res, segment_res, height_res)
    >>> mesh.shape
    (416, 4, 3)

    This example generates a cylinder mesh with 8 radial divisions for the disks,
    16 angular segments around the side, and 10 divisions along the height.
    """
    if radius <= 0:
        msg = f"radius must be positive. Got {radius}."
        raise ValueError(msg)
    if height <= 0:
        msg = f"height must be positive. Got {height}."
        raise ValueError(msg)
    if radial_resolution < 1:
        msg = f"radial_resolution must be at least 1. Got {radial_resolution}."
        raise ValueError(msg)
    if segment_resolution < 1:
        msg = f"segment_resolution must be at least 1. Got {segment_resolution}."
        raise ValueError(msg)
    if height_resolution < 1:
        msg = f"height_resolution must be at least 1 to create lateral faces. Got {height_resolution}."
        raise ValueError(msg)

    disk_mesh = disk_mesher_radial(radius, radial_resolution, segment_resolution)

    # flip the disk mesh to create the top disk
    disk_mesh_flip = np.flip(disk_mesh, axis=1)

    top_disk_mesh = convert_2d_face_to_3d(disk_mesh, axis=2, offset=height / 2)
    bottom_disk_mesh = convert_2d_face_to_3d(disk_mesh_flip, axis=2, offset=-height / 2)

    height_coords = 0.5 * height * np.linspace(-1, 1, height_resolution + 1)
    radial_coords = radius * np.ones_like(height_coords)
    lateral_curve = np.array([radial_coords, height_coords]).T
    lateral_mesh = circular_revolve(lateral_curve, segment_resolution)

    return np.vstack([
        bottom_disk_mesh,
        top_disk_mesh,
        lateral_mesh,
    ])


def cylinder_mesher_square_centered(radius: float, height: float, radial_resolution: int, half_square_resolution: int, height_resolution: int) -> np.ndarray:
    """
    Generate a mesh of a closed cylinder where the caps are based on a square-centered pattern.

    Parameters
    ----------
    radius : float
        Radius of the cylinder.
    height : float
        Height of the cylinder.
    radial_resolution : int
        Number of radial divisions from center to radius for the disks.
    half_square_resolution : int
        Number of divisions along half the square edge; full square resolution will be double.
    height_resolution : int
        Number of divisions along the height of the cylinder (plus one for full coverage).

    Returns
    -------
    mesh : ndarray of shape (n_faces, 4, 3)
        Stack of 3D faces representing the bottom disk, top disk, and lateral surface.

    Examples
    --------
    >>> radius = 1.0
    >>> height = 2.0
    >>> radial_res = 5
    >>> half_square_res = 4
    >>> height_res = 8
    >>> mesh = cylinder_mesher_square_centered(radius, height, radial_res, half_square_res, height_res)
    >>> mesh.shape
    (704, 4, 3)

    This generates a cylinder with square-centered disk caps and a smoothly revolved side wall.
    """
    if radius <= 0:
        msg = f"radius must be positive. Got {radius}."
        raise ValueError(msg)
    if height <= 0:
        msg = f"height must be positive. Got {height}."
        raise ValueError(msg)
    if radial_resolution < 1:
        msg = f"radial_resolution must be at least 1. Got {radial_resolution}."
        raise ValueError(msg)
    if half_square_resolution < 1:
        msg = f"half_square_resolution must be at least 1. Got {half_square_resolution}."
        raise ValueError(msg)
    if height_resolution < 1:
        msg = f"height_resolution must be at least 1 to create lateral faces. Got {height_resolution}."
        raise ValueError(msg)

    square_resolution = 2 * half_square_resolution

    disk_mesh = disk_mesher_square_centered(radius, square_resolution, radial_resolution)

    # flip the disk mesh to create the top disk
    disk_mesh_flip = np.flip(disk_mesh, axis=1)

    top_disk_mesh = convert_2d_face_to_3d(disk_mesh, axis=2, offset=height / 2)
    bottom_disk_mesh = convert_2d_face_to_3d(disk_mesh_flip, axis=2, offset=-height / 2)

    segment_resolution = square_resolution * 4

    height_coords = 0.5 * height * np.linspace(-1, 1, height_resolution + 1)
    radial_coords = radius * np.ones_like(height_coords)
    lateral_curve = np.array([radial_coords, height_coords]).T
    lateral_mesh = circular_revolve(lateral_curve, segment_resolution)

    return np.vstack([
        bottom_disk_mesh,
        top_disk_mesh,
        lateral_mesh,
    ])
