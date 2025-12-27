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
        output_start: Start index of output neurons (inclusive).
        output_end: End index of output neurons (exclusive).
    """
    input_start: int
    input_end: int
    model_start: int
    model_end: int
    output_start: int = 0
    output_end: int = 0

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
    def output_slice(self) -> slice:
        """Slice for output neurons."""
        return slice(self.output_start, self.output_end)

    @property
    def output_range(self) -> range:
        """Range for output neurons."""
        return range(self.output_start, self.output_end)

    @property
    def num_inputs(self) -> int:
        return self.input_end - self.input_start

    @property
    def num_model(self) -> int:
        return self.model_end - self.model_start

    @property
    def num_outputs(self) -> int:
        return self.output_end - self.output_start


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
    set_input(). The forward() method updates state in-place and returns the
    output neuron values.

    Attributes:
        neurons: List of neurons in the network.
        connections: N×N sparse boolean matrix (True = connection exists).
        weights: N×N sparse float matrix (connection weights).
        num_bands: Number of bands per neuron.
        ranges: Index ranges for input, model, and output neurons.
        state: (num_neurons, num_bands) array of neuron activations.
        input_values: (num_inputs,) array of raw input values.
        output_weights: (num_outputs, num_bands) learnable weights for mixing output bands.
    """
    neurons: list[Neuron] = field(default_factory=list)
    connections: lil_array = field(init=False)
    weights: lil_array = field(init=False)
    state: np.ndarray = field(init=False)
    input_values: np.ndarray = field(init=False)
    num_bands: int = 1
    ranges: NeuronRanges = field(default_factory=lambda: NeuronRanges(0, 0, 0, 0))

    def __post_init__(self) -> None:
        """Initialize empty sparse matrices, state, and output weights."""
        n = len(self.neurons)
        self.connections = lil_array((n, n), dtype=np.bool_)
        self.weights = lil_array((n, n), dtype=np.float32)
        self.state = np.zeros((n, self.num_bands), dtype=np.float32)
        if self.ranges.model_end == 0 and n > 0:
            self.ranges = NeuronRanges(0, 0, 0, n)
        
        self.input_values = np.zeros(self.ranges.num_inputs, dtype=np.float32)

        assert self.ranges.input_start >= 0, "input_start must be non-negative"
        assert self.ranges.model_start >= 0, "model_start must be non-negative"
        assert self.ranges.output_start >= 0, "output_start must be non-negative"
        total = self.ranges.num_inputs + self.ranges.num_model + self.ranges.num_outputs
        assert total == n, f"total number of neurons must match: {total} != {n}"

        self.output_weights = np.ones(
            (self.ranges.num_outputs, self.num_bands), dtype=np.float32
        )
    
    @property
    def num_inputs(self) -> int:
        """Number of input neurons (for backwards compatibility)."""
        return self.ranges.num_inputs

    @property
    def num_outputs(self) -> int:
        """Number of output neurons."""
        return self.ranges.num_outputs

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
            AssertionError: If from_idx is an output neuron.
        """
        assert not (self.ranges.input_start <= to_idx < self.ranges.input_end), (
            f"Cannot connect to input neuron {to_idx}. "
            f"Input neurons are in range [{self.ranges.input_start}, {self.ranges.input_end})"
        )
        assert not (self.ranges.output_start <= from_idx < self.ranges.output_end), (
            f"Cannot connect from output neuron {from_idx}. "
            f"Output neurons are in range [{self.ranges.output_start}, {self.ranges.output_end})"
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
        """Set raw input values and update state for input neurons.

        Args:
            values: Array of shape (num_inputs,) with one value per input neuron.

        Raises:
            ValueError: If values shape doesn't match num_inputs.
        """
        if values.shape != (self.num_inputs,):
            raise ValueError(
                f"Expected input shape ({self.num_inputs},), got {values.shape}"
            )
        self.input_values[:] = values.astype(np.float32, copy=False)
        self._sync_input_state()

    def _sync_input_state(self) -> None:
        """Sync internal state for input neurons from stored input_values."""
        if self.num_inputs > 0:
            self.state[self.ranges.input_slice] = np.tile(
                self.input_values[:, np.newaxis], (1, self.num_bands)
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
    
    def forward(self) -> np.ndarray:
        """Forward pass through the network.

        Updates the internal state in-place. Input neuron state is set
        based on stored input_values.

        Returns:
            Output vector of shape (num_outputs,) with mixed band values for
            each output neuron. Returns empty array if no output neurons.

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

        # Set input state from stored values
        self._sync_input_state()

        # Mix output neuron bands down to scalar outputs
        output_state = self.state[self.ranges.output_slice]  # (num_outputs, num_bands)
        output = (output_state * self.output_weights).sum(axis=1)  # (num_outputs,)
        return output.astype(np.float32, copy=False)

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
