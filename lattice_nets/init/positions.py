"""Position initialization strategies for neurons in the network."""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PositionInitializer(Protocol):
    """Protocol for position initialization strategies."""
    
    def initialize(self, num_neurons: int, dimensions: int) -> np.ndarray:
        """Generate positions for neurons.
        
        Args:
            num_neurons: Number of neurons to place.
            dimensions: Dimensionality of the space.
            
        Returns:
            Array of shape (num_neurons, dimensions) with positions.
        """
        ...


class LatticePositionInitializer:
    """Initialize neurons on an evenly-spaced lattice grid.
    
    Neurons are distributed in a grid pattern with adjacent neurons
    approximately `spacing` units apart (default: 1.0). This makes
    distribution parameters (like sigma) absolute distances rather
    than relative to space size.
    
    If num_neurons doesn't form a perfect hypercube, extra positions
    are added along the first dimensions.
    
    Args:
        spacing: Distance between adjacent neurons. Default is 1.0,
                 meaning adjacent neurons are 1 unit apart.
    """
    
    def __init__(self, spacing: float = 1.0) -> None:
        self.spacing = spacing
    
    def initialize(self, num_neurons: int, dimensions: int) -> np.ndarray:
        """Generate lattice positions.
        
        Args:
            num_neurons: Number of neurons to place.
            dimensions: Dimensionality of the space.
            
        Returns:
            Array of shape (num_neurons, dimensions) with positions.
        """
        # Compute points per dimension to get at least num_neurons
        points_per_dim = int(np.ceil(num_neurons ** (1 / dimensions)))
        
        # Use the specified spacing (default 1.0 for unit spacing)
        step = self.spacing
        
        # Generate grid coordinates
        coords = [np.arange(points_per_dim) * step for _ in range(dimensions)]
        grids = np.meshgrid(*coords, indexing='ij')
        
        # Stack and reshape to (total_points, dimensions)
        positions = np.stack([g.ravel() for g in grids], axis=1)
        
        # Take only the first num_neurons positions
        return positions[:num_neurons].astype(np.float64)


class RandomPositionInitializer:
    """Initialize neurons at random positions.
    
    Positions are uniformly distributed within a hypercube. By default,
    the size of the hypercube is computed to achieve approximately unit
    density (neurons ~1 unit apart on average), matching the behavior
    of LatticePositionInitializer.
    
    Args:
        low: Lower bound for each dimension. Default: 0.0
        size: Size of the hypercube in each dimension. If None (default),
              computed automatically to achieve ~1 unit spacing between
              neurons on average.
        seed: Random seed for reproducibility.
    """
    
    def __init__(
        self,
        low: float = 0.0,
        size: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.low = low
        self.size = size  # None means auto-compute based on num_neurons
        self.rng = np.random.default_rng(seed)
    
    def initialize(self, num_neurons: int, dimensions: int) -> np.ndarray:
        """Generate random positions.
        
        Args:
            num_neurons: Number of neurons to place.
            dimensions: Dimensionality of the space.
            
        Returns:
            Array of shape (num_neurons, dimensions) with positions.
        """
        if self.size is not None:
            size = self.size
        else:
            # Compute size to achieve approximately unit spacing
            # For a lattice of N neurons in d dimensions, we'd have
            # ~N^(1/d) neurons per dimension, so size should be ~N^(1/d) - 1
            # to match a lattice with spacing=1.0
            points_per_dim = int(np.ceil(num_neurons ** (1 / dimensions)))
            size = max(points_per_dim - 1, 1)
        
        return self.rng.uniform(
            self.low, self.low + size, size=(num_neurons, dimensions)
        ).astype(np.float64)
