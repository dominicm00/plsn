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
    
    Neurons are distributed in a hypercube from 0 to 1 in each dimension,
    arranged in a grid pattern. If num_neurons doesn't form a perfect
    hypercube, extra positions are added along the first dimensions.
    
    Args:
        spacing: Distance between adjacent neurons. If None, computed
                 automatically based on num_neurons.
    """
    
    def __init__(self, spacing: float | None = None) -> None:
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
        
        if self.spacing is not None:
            step = self.spacing
        else:
            step = 1.0 / max(points_per_dim - 1, 1)
        
        # Generate grid coordinates
        coords = [np.arange(points_per_dim) * step for _ in range(dimensions)]
        grids = np.meshgrid(*coords, indexing='ij')
        
        # Stack and reshape to (total_points, dimensions)
        positions = np.stack([g.ravel() for g in grids], axis=1)
        
        # Take only the first num_neurons positions
        return positions[:num_neurons].astype(np.float64)


class RandomPositionInitializer:
    """Initialize neurons at random positions.
    
    Positions are uniformly distributed within a hypercube.
    
    Args:
        low: Lower bound for each dimension.
        high: Upper bound for each dimension.
        seed: Random seed for reproducibility.
    """
    
    def __init__(
        self,
        low: float = 0.0,
        high: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.low = low
        self.high = high
        self.rng = np.random.default_rng(seed)
    
    def initialize(self, num_neurons: int, dimensions: int) -> np.ndarray:
        """Generate random positions.
        
        Args:
            num_neurons: Number of neurons to place.
            dimensions: Dimensionality of the space.
            
        Returns:
            Array of shape (num_neurons, dimensions) with positions.
        """
        return self.rng.uniform(
            self.low, self.high, size=(num_neurons, dimensions)
        ).astype(np.float64)
