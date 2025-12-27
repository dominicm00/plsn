"""LatticeNetwork class - collection of neurons with weighted connections."""

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from scipy.sparse import lil_array, csr_array

from plsn.core.neuron import Neuron


@dataclass
class LatticeNetwork:
    """A network of neurons connected via a weight matrix.

    The weight matrix is N×N where N is the number of neurons.
    Uses scipy sparse arrays for memory efficiency with sparse connectivity.

    Connection structure is stored separately from weights:
    - connections: Boolean sparse matrix indicating which connections exist
    - weights: Float sparse matrix storing connection weights

    This allows zero-weight connections that can still be updated during learning.

    Attributes:
        neurons: List of neurons in the network.
        connections: N×N sparse boolean matrix (True = connection exists).
        weights: N×N sparse float matrix (connection weights).
        num_bands: Number of bands per neuron.
    """
    neurons: list[Neuron] = field(default_factory=list)
    connections: lil_array = field(init=False)
    weights: lil_array = field(init=False)
    num_bands: int = 1

    def __post_init__(self) -> None:
        """Initialize empty sparse matrices."""
        n = len(self.neurons)
        self.connections = lil_array((n, n), dtype=np.bool_)
        self.weights = lil_array((n, n), dtype=np.float32)
    
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
        self.connections[from_idx, to_idx] = True
        self.weights[from_idx, to_idx] = weight

    def disconnect(self, from_idx: int, to_idx: int) -> None:
        """Remove connection between two neurons."""
        # Remove from connections matrix
        row = self.connections.rows[from_idx]
        data = self.connections.data[from_idx]
        if to_idx in row:
            idx = row.index(to_idx)
            row.pop(idx)
            data.pop(idx)
        # Remove from weights matrix
        row = self.weights.rows[from_idx]
        data = self.weights.data[from_idx]
        if to_idx in row:
            idx = row.index(to_idx)
            row.pop(idx)
            data.pop(idx)

    def is_connected(self, from_idx: int, to_idx: int) -> bool:
        """Check if two neurons are connected."""
        return to_idx in self.connections.rows[from_idx]
    
    def get_connections(self, neuron_idx: int) -> list[tuple[int, float]]:
        """Get all incoming connections to a neuron.

        Returns:
            List of (source_idx, weight) tuples.
        """
        # Incoming connections: look at column neuron_idx across all rows
        result = []
        for i in range(self.num_neurons):
            if neuron_idx in self.connections.rows[i]:
                weight = self.weights[i, neuron_idx]
                result.append((i, weight))
        return result

    def get_outgoing(self, neuron_idx: int) -> list[tuple[int, float]]:
        """Get all outgoing connections from a neuron.

        Returns:
            List of (target_idx, weight) tuples.
        """
        # Outgoing connections: directly from the connections row
        targets = self.connections.rows[neuron_idx]
        return [(j, self.weights[neuron_idx, j]) for j in targets]
    
    def connection_count(self) -> int:
        """Count total number of connections in the network."""
        return self.connections.nnz
    
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

        # Convert to CSR for efficient matrix multiplication
        # weights is (n, n), inputs is (n, b)
        # We need weights.T @ inputs for each neuron j to get weighted sum of inputs
        weights_csr = csr_array(self.weights)
        weighted_inputs = weights_csr.T @ inputs  # (n, b)

        outputs = np.zeros((n, b), dtype=np.float32)
        for j, neuron in enumerate(self.neurons):
            # Mix bands within the neuron
            outputs[j] = neuron.process_bands(weighted_inputs[j])

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
