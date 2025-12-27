"""LatticeNetwork class - collection of neurons with weighted connections."""

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from scipy.sparse import lil_array, csr_array

from plsn.core.neuron import Neuron


@dataclass
class NeuronRanges:
    """Index ranges for different neuron types in the network.
    
    Attributes:
        input_start: Start index of input neurons (inclusive).
        input_end: End index of input neurons (exclusive).
        model_start: Start index of model neurons (inclusive).
        model_end: End index of model neurons (exclusive).
    """
    input_start: int
    input_end: int
    model_start: int
    model_end: int
    
    @property
    def input_slice(self) -> slice:
        """Slice for input neurons."""
        return slice(self.input_start, self.input_end)

    @property
    def input_range(self) -> range:
        """Range for input neurons."""
        return range(self.input_start, self.input_end)
    
    @property
    def model_slice(self) -> slice:
        """Slice for model neurons."""
        return slice(self.model_start, self.model_end)

    @property
    def model_range(self) -> range:
        """Range for model neurons."""
        return range(self.model_start, self.model_end)
    
    @property
    def num_inputs(self) -> int:
        return self.input_end - self.input_start
    
    @property
    def num_model(self) -> int:
        return self.model_end - self.model_start


@dataclass
class LatticeNetwork:
    """A network of neurons connected via a weight matrix.

    The weight matrix is N×N where N is the number of neurons.
    Uses scipy sparse arrays for memory efficiency with sparse connectivity.

    Connection structure is stored separately from weights:
    - connections: Boolean sparse matrix indicating which connections exist
    - weights: Float sparse matrix storing connection weights

    This allows zero-weight connections that can still be updated during learning.

    State is stored internally. Input neurons can have their state set via
    set_input(). The forward() method updates state in-place.

    Attributes:
        neurons: List of neurons in the network.
        connections: N×N sparse boolean matrix (True = connection exists).
        weights: N×N sparse float matrix (connection weights).
        num_bands: Number of bands per neuron.
        ranges: Index ranges for input and model neurons.
        state: (num_neurons, num_bands) array of neuron activations.
    """
    neurons: list[Neuron] = field(default_factory=list)
    connections: lil_array = field(init=False)
    weights: lil_array = field(init=False)
    num_bands: int = 1
    ranges: NeuronRanges = field(default_factory=lambda: NeuronRanges(0, 0, 0, 0))

    def __post_init__(self) -> None:
        """Initialize empty sparse matrices and state."""
        n = len(self.neurons)
        self.connections = lil_array((n, n), dtype=np.bool_)
        self.weights = lil_array((n, n), dtype=np.float32)
        self.state = np.zeros((n, self.num_bands), dtype=np.float32)
        if self.ranges.model_end == 0 and n > 0:
            self.ranges = NeuronRanges(0, 0, 0, n)
        
        assert self.ranges.input_start >= 0, "input_start must be non-negative"
        assert self.ranges.model_start >= 0, "model_start must be non-negative"
        assert self.ranges.num_inputs + self.ranges.num_model == n, "total number of neurons must match"
    
    @property
    def num_inputs(self) -> int:
        """Number of input neurons (for backwards compatibility)."""
        return self.ranges.num_inputs
    
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
    
    def ensure_weights_lil(self) -> None:
        """Convert weights to LIL format if not already LIL.
        
        LIL format is efficient for element-wise assignments and modifications.
        """
        if not isinstance(self.weights, lil_array):
            self.weights = self.weights.tolil()
    
    def ensure_weights_csr(self) -> None:
        """Convert weights to CSR format if not already CSR.
        
        CSR format is efficient for matrix operations and computations.
        """
        if not isinstance(self.weights, csr_array):
            self.weights = self.weights.tocsr()
    
    def connect(self, from_idx: int, to_idx: int, weight: float = 0.0) -> None:
        """Create a connection between two neurons.

        Args:
            from_idx: Index of source neuron.
            to_idx: Index of target neuron.
            weight: Initial connection weight.
        
        Raises:
            AssertionError: If to_idx targets an input neuron.
        """
        assert not (self.ranges.input_start <= to_idx < self.ranges.input_end), (
            f"Cannot connect to input neuron {to_idx}. "
            f"Input neurons are in range [{self.ranges.input_start}, {self.ranges.input_end})"
        )
        self.ensure_weights_lil()
        self.connections[from_idx, to_idx] = True
        self.weights[from_idx, to_idx] = weight

    def disconnect(self, from_idx: int, to_idx: int) -> None:
        """Remove connection between two neurons."""
        self.ensure_weights_lil()
        self.connections[from_idx, to_idx] = False
        self.weights[from_idx, to_idx] = 0.0

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
    
    def set_input(self, values: np.ndarray) -> None:
        """Set state for input neurons.

        The input values are replicated across all bands for each input neuron.

        Args:
            values: Array of shape (num_inputs,) with one value per input neuron.

        Raises:
            ValueError: If values shape doesn't match num_inputs.
        """
        if values.shape != (self.num_inputs,):
            raise ValueError(
                f"Expected input shape ({self.num_inputs},), got {values.shape}"
            )
        self.state[self.ranges.input_slice] = np.tile(
            values[:, np.newaxis], (1, self.num_bands)
        ).astype(np.float32, copy=False)
    
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
    
    def forward(self) -> None:
        """Forward pass through the network.

        Updates the internal state in-place. Input neuron state is preserved
        (not overwritten by incoming connections).

        Note:
            This is a single-step forward pass. For recurrent processing,
            call this method multiple times.
        """
        self.ensure_weights_csr()
        weighted_inputs = self.weights.T @ self.state  # (n, b)

        band_weights = np.stack(
            [neuron.band_weights for neuron in self.neurons],
            axis=0,
        ).astype(np.float32, copy=False)  # (n, b, b)

        self.state = np.einsum("nij,nj->ni", band_weights, weighted_inputs).astype(
            np.float32, copy=False
        )

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
