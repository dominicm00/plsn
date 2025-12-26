"""Neuron class with spatial position and band-based processing."""

from dataclasses import dataclass, field
from typing import Self

import numpy as np


@dataclass
class Neuron:
    """A neuron with a spatial position and internal band mixing.
    
    Attributes:
        position: d-dimensional position vector in space.
        is_global: Whether this is a global neuron (vs local).
        num_bands: Number of signal bands this neuron processes.
        band_weights: Internal B×B weight matrix for mixing bands.
                     Initialized to identity (bands pass through unchanged).
    """
    position: np.ndarray
    is_global: bool = False
    num_bands: int = 1
    band_weights: np.ndarray = field(init=False)
    
    def __post_init__(self) -> None:
        """Initialize band mixing weights to identity matrix."""
        self.band_weights = np.eye(self.num_bands, dtype=np.float64)
    
    @property
    def dimensions(self) -> int:
        """Number of spatial dimensions."""
        return len(self.position)
    
    def process_bands(self, inputs: np.ndarray) -> np.ndarray:
        """Mix input bands through internal weights.
        
        Args:
            inputs: Array of shape (num_bands,) with input signal per band.
            
        Returns:
            Array of shape (num_bands,) with mixed output signals.
        """
        if inputs.shape != (self.num_bands,):
            raise ValueError(
                f"Expected input shape ({self.num_bands},), got {inputs.shape}"
            )
        return self.band_weights @ inputs
    
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
        neuron_type = "global" if self.is_global else "local"
        return f"Neuron(pos={pos_str}, {neuron_type}, bands={self.num_bands})"
