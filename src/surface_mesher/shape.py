from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class Shape(ABC):
    """
    Abstract base class for 3D shapes.
    """

    @abstractmethod
    def create_mesh_from_edge_sizes(self, mesh_sizes: float | ArrayLike) -> np.ndarray:
        """Generate surface mesh based on edge sizes."""
        ...  # pragma: no cover

    @abstractmethod
    def create_mesh_with_resolution(self, resolution: int | ArrayLike) -> np.ndarray:
        """Generate surface mesh based on number of divisions (resolution)."""
        ...  # pragma: no cover
