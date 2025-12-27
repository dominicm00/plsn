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
    """Protocol for connection initialization strategies.
    
    Connection initializers receive source and target neuron ranges to control
    which neurons to connect. This allows the same initializer to be used for
    input->model, model->model, or any custom connectivity pattern.
    """

    def initialize(
        self,
        network: LatticeNetwork,
        rng: np.random.Generator,
        source_range: range,
        target_range: range,
    ) -> None:
        """Initialize connections in the network.

        This method modifies the network's weight matrix in-place,
        setting connections based on the strategy.

        Args:
            network: The network to initialize connections for.
            rng: Random number generator to use for stochastic initialization.
            source_range: Range of source neurons (connections go FROM these).
            target_range: Range of target neurons (connections go TO these).
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

    def initialize(
        self,
        network: LatticeNetwork,
        rng: np.random.Generator,
        source_range: range,
        target_range: range,
    ) -> None:
        """Create connections based on distance probabilities.

        Connects neurons from source_range to target_range based on distance.

        Note: Distances are passed directly to the distribution (not normalized).
        With the default positioning where neurons are ~1 unit apart, distribution
        parameters (like sigma) represent absolute distances.
        """
        if network.num_neurons == 0:
            return

        for i in source_range:
            for j in target_range:
                if i == j and not self.self_connections:
                    continue

                dist = network.neurons[i].distance_to(
                    network.neurons[j], ord=self.ord
                )

                prob = self.distribution.probability(dist)

                if rng.random() < prob:
                    network.connect(i, j, self.initial_weight)
                    if self.bidirectional and i != j:
                        network.connect(j, i, self.initial_weight)


class FullyConnectedInitializer:
    """Connect all source neurons to all target neurons.

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

    def initialize(
        self,
        network: LatticeNetwork,
        rng: np.random.Generator,
        source_range: range,
        target_range: range,
    ) -> None:
        """Create all possible connections from source to target range."""
        for i in source_range:
            for j in target_range:
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

    def initialize(
        self,
        network: LatticeNetwork,
        rng: np.random.Generator,
        source_range: range,
        target_range: range,
    ) -> None:
        """Create connections with uniform probability from source to target."""
        if network.num_neurons == 0:
            return

        for i in source_range:
            for j in target_range:
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

    def initialize(
        self,
        network: LatticeNetwork,
        rng: np.random.Generator,
        source_range: range,
        target_range: range,
    ) -> None:
        """Initialize weights for existing connections within the given ranges."""
        if network.num_neurons == 0:
            return

        rows = network.connections.rows
        for i in source_range:
            for j in rows[i]:
                if j in target_range:
                    weight = self.distribution.sample(rng)
                    network.weights[i, j] = weight

