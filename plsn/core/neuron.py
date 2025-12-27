"""Neuron class with spatial position for connectivity."""

from dataclasses import dataclass
from typing import Self

import numpy as np


@dataclass
class Neuron:
    """A neuron with a spatial position for determining connectivity.

    Attributes:
        position: d-dimensional position vector in space.
    """
    position: np.ndarray

    @property
    def dimensions(self) -> int:
        """Number of spatial dimensions."""
        return len(self.position)

    def distance_to(self, other: Self, ord: int = 1) -> float:
        """Compute distance to another neuron.

        Args:
            other: Another neuron to compute distance to.
            ord: Order of the norm (1 for L1/Manhattan, 2 for L2/Euclidean).

        Returns:
            Distance between this neuron and the other.
        """
        return float(np.linalg.norm(self.position - other.position, ord=ord))

    def __repr__(self) -> str:
        pos_str = np.array2string(self.position, precision=2, separator=', ')
        return f"Neuron(pos={pos_str})"
