"""LatticeNetwork class - collection of neurons with weighted connections."""

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from lattice_nets.core.neuron import Neuron


@dataclass
class LatticeNetwork:
    """A network of neurons connected via a weight matrix.
    
    The weight matrix is N×N where N is the number of neurons.
    - None values indicate no connection (cannot be adjusted during learning)
    - Float values indicate connection strength (can be adjusted)
    
    Attributes:
        neurons: List of neurons in the network.
        weights: N×N connection matrix (dtype=object to allow None).
        num_bands: Number of bands per neuron.
    """
    neurons: list[Neuron] = field(default_factory=list)
    weights: np.ndarray = field(init=False)
    num_bands: int = 1
    
    def __post_init__(self) -> None:
        """Initialize weight matrix with all None values."""
        n = len(self.neurons)
        self.weights = np.full((n, n), None, dtype=object)
    
    @property
    def num_neurons(self) -> int:
        """Number of neurons in the network."""
        return len(self.neurons)
    
    @property
    def dimensions(self) -> int:
        """Spatial dimensions of the network (from first neuron)."""
        if not self.neurons:
            return 0
        return self.neurons[0].dimensions
    
    def connect(self, from_idx: int, to_idx: int, weight: float = 0.0) -> None:
        """Create a connection between two neurons.
        
        Args:
            from_idx: Index of source neuron.
            to_idx: Index of target neuron.
            weight: Initial connection weight.
        """
        self.weights[from_idx, to_idx] = weight
    
    def disconnect(self, from_idx: int, to_idx: int) -> None:
        """Remove connection between two neurons (set to None)."""
        self.weights[from_idx, to_idx] = None
    
    def is_connected(self, from_idx: int, to_idx: int) -> bool:
        """Check if two neurons are connected."""
        return self.weights[from_idx, to_idx] is not None
    
    def get_connections(self, neuron_idx: int) -> list[tuple[int, float]]:
        """Get all incoming connections to a neuron.
        
        Returns:
            List of (source_idx, weight) tuples.
        """
        connections = []
        for i in range(self.num_neurons):
            w = self.weights[i, neuron_idx]
            if w is not None:
                connections.append((i, w))
        return connections
    
    def get_outgoing(self, neuron_idx: int) -> list[tuple[int, float]]:
        """Get all outgoing connections from a neuron.
        
        Returns:
            List of (target_idx, weight) tuples.
        """
        connections = []
        for j in range(self.num_neurons):
            w = self.weights[neuron_idx, j]
            if w is not None:
                connections.append((j, w))
        return connections
    
    def connection_count(self) -> int:
        """Count total number of connections in the network."""
        return int(np.sum(self.weights != None))  # noqa: E711
    
    def max_distance(self, ord: int = 1) -> float:
        """Compute maximum pairwise distance between any two neurons.
        
        Args:
            ord: Order of the norm (1 for L1, 2 for L2).
        """
        if len(self.neurons) < 2:
            return 0.0
        
        max_dist = 0.0
        for i, n1 in enumerate(self.neurons):
            for n2 in self.neurons[i + 1:]:
                dist = n1.distance_to(n2, ord=ord)
                if dist > max_dist:
                    max_dist = dist
        return max_dist
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through the network.
        
        Args:
            inputs: Array of shape (num_neurons, num_bands) with input per neuron/band.
            
        Returns:
            Array of shape (num_neurons, num_bands) with output signals.
            
        Note:
            This is a single-step forward pass. For recurrent processing,
            call this method multiple times.
        """
        n = self.num_neurons
        b = self.num_bands
        
        if inputs.shape != (n, b):
            raise ValueError(f"Expected input shape ({n}, {b}), got {inputs.shape}")
        
        outputs = np.zeros((n, b), dtype=np.float64)
        
        for j, neuron in enumerate(self.neurons):
            # Aggregate inputs from connected neurons (per band)
            band_inputs = np.zeros(b, dtype=np.float64)
            
            for i in range(n):
                w = self.weights[i, j]
                if w is not None:
                    # Each band is summed separately
                    band_inputs += w * inputs[i]
            
            # Mix bands within the neuron
            outputs[j] = neuron.process_bands(band_inputs)
        
        return outputs
    
    def __iter__(self) -> Iterator[Neuron]:
        """Iterate over neurons."""
        return iter(self.neurons)
    
    def __len__(self) -> int:
        return len(self.neurons)
    
    def __repr__(self) -> str:
        return (
            f"LatticeNetwork(neurons={self.num_neurons}, "
            f"dims={self.dimensions}, "
            f"bands={self.num_bands}, "
            f"connections={self.connection_count()})"
        )
