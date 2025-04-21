import pytest
import numpy as np
from surface_mesher import Shape


def test_shape_instantiation_fails():
    with pytest.raises(TypeError):
        Shape()  # Cannot instantiate abstract class


def test_shape_subclass_without_implementation_fails():
    class IncompleteShape(Shape):
        ... # pragma: no cover

    with pytest.raises(TypeError):
        IncompleteShape()  # Still abstract, no methods implemented


def test_shape_subclass_with_methods_can_instantiate():
    class DummyShape(Shape):
        def create_mesh_from_edge_sizes(self, mesh_sizes):
            return np.array([[1, 2, 3]])

        def create_mesh_with_resolution(self, resolution):
            return np.array([[4, 5, 6]])

    shape = DummyShape()
    assert isinstance(shape.create_mesh_from_edge_sizes(1), np.ndarray)
    assert isinstance(shape.create_mesh_with_resolution(1), np.ndarray)
