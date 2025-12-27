"""Connection initialization strategies based on neuron positions."""

from typing import Protocol, runtime_checkable

import numpy as np

from plsn.core.network import LatticeNetwork
from plsn.init.distributions import (
    DistanceDistribution,
    LinearDistanceDistribution,
    WeightDistribution,
    ConstantWeightDistribution,
)


@runtime_checkable
class ConnectionInitializer(Protocol):
    """Protocol for connection initialization strategies."""

    def initialize(self, network: LatticeNetwork, rng: np.random.Generator) -> None:
        """Initialize connections in the network.

        This method modifies the network's weight matrix in-place,
        setting connections based on the strategy.

        Args:
            network: The network to initialize connections for.
            rng: Random number generator to use for stochastic initialization.
        """
        ...


class DistanceBasedInitializer:
    """Initialize connections based on distance between neurons.

    Uses L1 (Manhattan) distance by default. The probability of connection
    is determined by a DistanceDistribution that maps normalized distance
    to probability.

    Args:
        distribution: Probability distribution for distance-based connection.
        initial_weight: Initial weight for created connections.
        self_connections: Whether neurons can connect to themselves.
        bidirectional: If True, connections are symmetric (i→j implies j→i).
        ord: Order of the norm (1 for L1, 2 for L2).
    """

    def __init__(
        self,
        distribution: DistanceDistribution | None = None,
        initial_weight: float = 0.0,
        self_connections: bool = False,
        bidirectional: bool = False,
        ord: int = 1,
    ) -> None:
        self.distribution = distribution or LinearDistanceDistribution()
        self.initial_weight = initial_weight
        self.self_connections = self_connections
        self.bidirectional = bidirectional
        self.ord = ord

    def initialize(self, network: LatticeNetwork, rng: np.random.Generator) -> None:
        """Create connections based on distance probabilities.

        For each pair of neurons, computes the probability of connection
        based on their distance, then randomly creates the connection
        with that probability.

        Note: Distances are passed directly to the distribution (not normalized).
        With the default positioning where neurons are ~1 unit apart, distribution
        parameters (like sigma) represent absolute distances.
        """
        n = network.num_neurons
        if n == 0:
            return

        # For each pair, compute probability and potentially connect
        for i in range(n):
            start_j = 0 if self.bidirectional else i
            for j in range(start_j, n):
                if i == j and not self.self_connections:
                    continue

                # Compute distance (not normalized - distributions work with absolute distances)
                dist = network.neurons[i].distance_to(
                    network.neurons[j], ord=self.ord
                )

                # Get probability and roll the dice
                prob = self.distribution.probability(dist)

                if rng.random() < prob:
                    network.connect(i, j, self.initial_weight)
                    if self.bidirectional and i != j:
                        network.connect(j, i, self.initial_weight)


class FullyConnectedInitializer:
    """Connect all neurons to all other neurons.

    Args:
        initial_weight: Initial weight for all connections.
        self_connections: Whether neurons can connect to themselves.
    """

    def __init__(
        self,
        initial_weight: float = 0.0,
        self_connections: bool = False,
    ) -> None:
        self.initial_weight = initial_weight
        self.self_connections = self_connections

    def initialize(self, network: LatticeNetwork, rng: np.random.Generator) -> None:
        """Create all possible connections."""
        n = network.num_neurons
        for i in range(n):
            for j in range(n):
                if i == j and not self.self_connections:
                    continue
                network.connect(i, j, self.initial_weight)


class GlobalInitializer:
    """Connect neurons with uniform probability regardless of distance.

    This creates random "long-range" connections that ignore spatial structure,
    useful for adding small-world properties to locally-connected networks.

    Args:
        probability: Probability of creating each possible connection.
        initial_weight: Initial weight for created connections.
        self_connections: Whether neurons can connect to themselves.
        bidirectional: If True, connections are symmetric (i→j implies j→i).
    """

    def __init__(
        self,
        probability: float = 0.01,
        initial_weight: float = 0.0,
        self_connections: bool = False,
        bidirectional: bool = False,
    ) -> None:
        self.probability = probability
        self.initial_weight = initial_weight
        self.self_connections = self_connections
        self.bidirectional = bidirectional

    def initialize(self, network: LatticeNetwork, rng: np.random.Generator) -> None:
        """Create connections with uniform probability."""
        n = network.num_neurons
        if n == 0:
            return

        for i in range(n):
            start_j = 0 if self.bidirectional else i
            for j in range(start_j, n):
                if i == j and not self.self_connections:
                    continue

                if rng.random() < self.probability:
                    network.connect(i, j, self.initial_weight)
                    if self.bidirectional and i != j:
                        network.connect(j, i, self.initial_weight)


class WeightInitializer:
    """Initialize weights of existing connections.

    Iterates through all existing connections in the network and sets their
    weights by sampling from the provided WeightDistribution.

    Args:
        distribution: Distribution to sample weights from.
    """

    def __init__(
        self,
        distribution: WeightDistribution | None = None,
    ) -> None:
        self.distribution = distribution or ConstantWeightDistribution(1.0)

    def initialize(self, network: LatticeNetwork, rng: np.random.Generator) -> None:
        """Initialize weights for existing connections."""
        if network.num_neurons == 0:
            return

        # Iterate over all existing connections
        rows = network.connections.rows
        for i, cols in enumerate(rows):
            for j in cols:
                # Sample weight and assign
                # Note: modifying lil_array this way is efficient enough for this purpose
                weight = self.distribution.sample(rng)
                network.weights[i, j] = weight

