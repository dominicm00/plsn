"""LatticeNetwork class - collection of neurons with weighted connections."""

from dataclasses import dataclass, field
from typing import Iterator, Callable


import numpy as np
from scipy.sparse import lil_array, csr_array, coo_array

from plsn.core.neuron import Neuron
from plsn.core.learning import HebbianConfig




def relu(x: np.float32) -> np.float32:
    """ReLU activation function."""
    return np.maximum(0.0, x)


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
        band_weights: (N, B, B) array of band mixing weights per neuron.
        ranges: Index ranges for input, model, and output neurons.
        state: (num_neurons, num_bands) array of neuron activations.
        input_values: (num_inputs,) array of raw input values.
        output_weights: (num_outputs, num_bands) learnable weights for mixing output bands.
        learning_config: Configuration for Hebbian learning (None = no learning).
        learning_enabled: Whether to apply learning during forward pass.
        bcm_theta: (num_neurons,) BCM sliding threshold per neuron.
    """
    neurons: list[Neuron] = field(default_factory=list)
    connections: lil_array = field(init=False)
    weights: lil_array = field(init=False)
    state: np.ndarray = field(init=False)
    input_values: np.ndarray = field(init=False)
    band_weights: np.ndarray = field(init=False)
    num_bands: int = 1
    ranges: NeuronRanges = field(default_factory=lambda: NeuronRanges(0, 0, 0, 0))
    activation: "Callable[[float], float]" = field(default=relu)
    learning_config: HebbianConfig | None = None
    learning_enabled: bool = False
    bcm_theta: np.ndarray = field(init=False)
    _pre_state: np.ndarray = field(init=False)
    _post_state: np.ndarray = field(init=False)
    _weighted_inputs: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Initialize empty sparse matrices, state, and output weights."""
        if self.activation is None:
            self.activation = relu
        self.activation = np.vectorize(self.activation, otypes=[np.float32])

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

        # Initialize band weights as identity matrices for each neuron
        self.band_weights = np.stack(
            [np.eye(self.num_bands, dtype=np.float32) for _ in range(n)],
            axis=0,
        )  # (n, b, b)

        # Initialize BCM threshold for Hebbian learning
        if self.learning_config is not None:
            self.bcm_theta = np.full(
                n, self.learning_config.theta_init, dtype=np.float32
            )
        else:
            self.bcm_theta = np.zeros(n, dtype=np.float32)

        # Initialize learning state buffers
        self._pre_state = np.zeros((n, self.num_bands), dtype=np.float32)
        self._post_state = np.zeros((n, self.num_bands), dtype=np.float32)
        self._weighted_inputs = np.zeros((n, self.num_bands), dtype=np.float32)

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

    def _apply_hebbian_learning(
        self,
        pre_state: np.ndarray,
        post_state: np.ndarray,
        weighted_inputs: np.ndarray,
        reward: float = 1.0,
    ) -> None:
        """Apply Oja-BCM Hebbian learning to connection weights and band weights.

        Combined rule: dw = eta * reward * y * (y - theta) * (x - y * w)

        Where:
            - y: postsynaptic (output) activation
            - x: presynaptic (input) activation
            - theta: BCM sliding threshold
            - w: current weight
            - reward: scalar that scales learning (negative = depression)

        Args:
            pre_state: Neuron state before forward pass, shape (num_neurons, num_bands).
            post_state: Neuron state after activation, shape (num_neurons, num_bands).
            weighted_inputs: Input to band mixing, shape (num_neurons, num_bands).
            reward: Scalar reward signal that scales learning. Positive values
                strengthen learning, negative values cause depression (anti-learning).
        """
        if self.learning_config is None:
            return

        cfg = self.learning_config

        # === Update BCM thresholds ===
        # Average across bands to get scalar activation per neuron
        y_avg = post_state.mean(axis=1)  # (num_neurons,)
        y_squared = y_avg ** 2
        self.bcm_theta += (y_squared - self.bcm_theta) / cfg.bcm_tau

        # === Learn connection weights ===
        # Using vectorized sparse operations
        # x_j: presynaptic (source) activations, y_i: postsynaptic (target) activations
        x = pre_state.mean(axis=1).astype(np.float32)  # (n,)
        y = y_avg.astype(np.float32)  # (n,)
        bcm_term = y - self.bcm_theta  # (n,)

        # Convert to COO for efficient updates
        coo = self.weights.tocoo()
        rows, cols, data = coo.row, coo.col, coo.data.copy()

        if len(data) > 0:
            # rows[k] = j (source), cols[k] = i (target), data[k] = w_ji
            j = rows  # source indices
            i = cols  # target indices
            w = data  # current weights

            y_i = y[i]
            x_j = x[j]
            bcm_i = bcm_term[i]

            # Oja-BCM: dw = eta * reward * y_i * bcm_i * (x_j - y_i * w)
            oja_term = x_j - y_i * w
            dw = cfg.learning_rate * reward * y_i * bcm_i * oja_term

            # Weight decay
            if cfg.weight_decay > 0:
                dw -= cfg.weight_decay * w

            # Apply update and clip
            new_weights = np.clip(w + dw, cfg.min_weight, cfg.max_weight)

            # Reconstruct sparse matrix
            self.weights = coo_array(
                (new_weights.astype(np.float32), (rows, cols)),
                shape=self.weights.shape,
                dtype=np.float32,
            ).tocsr()

        # === Learn band weights (vectorized) ===
        # Band weights: (n, b, b) where band_weights[n, i, j] mixes input band j to output band i

        # Create mask for non-input neurons (only they learn)
        mask = np.ones(self.num_neurons, dtype=bool)
        mask[self.ranges.input_slice] = False

        # Extract tensors for learning neurons
        y_bands = post_state[mask]  # (N', B) - output activations
        x_bands = weighted_inputs[mask]  # (N', B) - inputs to band mixer
        theta = self.bcm_theta[mask, None]  # (N', 1) - BCM threshold
        w = self.band_weights[mask]  # (N', B, B) - current band weights

        # Compute BCM factor per output band: y * (y - theta)
        bcm = y_bands * (y_bands - theta)  # (N', B)

        # Reshape for broadcasting over (N', B_out, B_in)
        y_exp = y_bands[:, :, None]  # (N', B, 1)
        bcm_exp = bcm[:, :, None]  # (N', B, 1)
        x_exp = x_bands[:, None, :]  # (N', 1, B)

        # Oja-BCM rule: dw = eta * reward * y * bcm * (x - y * w)
        oja_term = x_exp - y_exp * w  # (N', B, B)
        dw = cfg.band_learning_rate * reward * y_exp * bcm_exp * oja_term

        # Weight decay
        if cfg.weight_decay > 0:
            dw -= cfg.weight_decay * w

        # Apply update and clip
        self.band_weights[mask] = np.clip(
            w + dw, cfg.min_weight, cfg.max_weight
        ).astype(np.float32)

    def forward(self) -> np.ndarray:
        """Forward pass through the network.

        Updates the internal state in-place. Input neuron state is set
        based on stored input_values. If learning is enabled, stores
        state information for later use by reward().

        Returns:
            Output vector of shape (num_outputs,) with mixed band values for
            each output neuron. Returns empty array if no output neurons.

        Note:
            This is a single-step forward pass. For recurrent processing,
            call this method multiple times. To apply learning, call
            reward() after forward() with a reward signal.
        """
        # Store pre-activation state for learning
        if self.learning_enabled and self.learning_config is not None:
            self._pre_state = self.state.copy()

        self.ensure_weights_csr()
        weighted_inputs = self.weights.T @ self.state  # (n, b)

        self.state = np.einsum(
            "nij,nj->ni", self.band_weights, weighted_inputs
        ).astype(np.float32, copy=False)

        # Apply activation function
        self.state = self.activation(self.state)

        # Set input state from stored values
        self._sync_input_state()

        # Store post-activation state and weighted inputs for learning
        if self.learning_enabled and self.learning_config is not None:
            self._post_state = self.state.copy()
            self._weighted_inputs = weighted_inputs.copy()

        # Mix output neuron bands down to scalar outputs
        output_state = self.state[self.ranges.output_slice]  # (num_outputs, num_bands)
        output = (output_state * self.output_weights).sum(axis=1)  # (num_outputs,)
        return output.astype(np.float32, copy=False)

    def reward(self, reward: float) -> None:
        """Apply reward-modulated Hebbian learning based on the last forward pass.

        This method should be called after forward() to apply learning scaled
        by the reward signal. Positive rewards strengthen the learned associations,
        while negative rewards cause depression (weakening of associations).

        Args:
            reward: Scalar reward signal. Positive values increase learning,
                negative values cause depression (anti-learning), and zero
                results in no weight changes.

        Note:
            This method has no effect if learning_enabled is False or if
            learning_config is None.
        """
        if not self.learning_enabled or self.learning_config is None:
            return

        self._apply_hebbian_learning(
            self._pre_state,
            self._post_state,
            self._weighted_inputs,
            reward=reward,
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
