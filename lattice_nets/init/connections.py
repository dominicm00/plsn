"""Connection initialization strategies based on neuron positions."""

from typing import Protocol, runtime_checkable

import numpy as np

from lattice_nets.core.network import LatticeNetwork
from lattice_nets.init.distributions import DistanceDistribution, LinearDistribution


@runtime_checkable
class ConnectionInitializer(Protocol):
    """Protocol for connection initialization strategies."""
    
    def initialize(self, network: LatticeNetwork) -> None:
        """Initialize connections in the network.
        
        This method modifies the network's weight matrix in-place,
        setting connections based on the strategy.
        
        Args:
            network: The network to initialize connections for.
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
        seed: Random seed for reproducibility.
    """
    
    def __init__(
        self,
        distribution: DistanceDistribution | None = None,
        initial_weight: float = 0.0,
        self_connections: bool = False,
        bidirectional: bool = False,
        ord: int = 1,
        seed: int | None = None,
    ) -> None:
        self.distribution = distribution or LinearDistribution()
        self.initial_weight = initial_weight
        self.self_connections = self_connections
        self.bidirectional = bidirectional
        self.ord = ord
        self.rng = np.random.default_rng(seed)
    
    def initialize(self, network: LatticeNetwork) -> None:
        """Create connections based on distance probabilities.
        
        For each pair of neurons, computes the probability of connection
        based on their distance (normalized by max network distance),
        then randomly creates the connection with that probability.
        """
        n = network.num_neurons
        if n == 0:
            return
        
        # Compute max distance for normalization
        max_dist = network.max_distance(ord=self.ord)
        if max_dist == 0:
            max_dist = 1.0  # Avoid division by zero
        
        # For each pair, compute probability and potentially connect
        for i in range(n):
            start_j = 0 if self.bidirectional else i
            for j in range(start_j, n):
                if i == j and not self.self_connections:
                    continue
                
                # Compute normalized distance
                dist = network.neurons[i].distance_to(
                    network.neurons[j], ord=self.ord
                )
                normalized_dist = dist / max_dist
                
                # Get probability and roll the dice
                prob = self.distribution.probability(normalized_dist)
                
                if self.rng.random() < prob:
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
    
    def initialize(self, network: LatticeNetwork) -> None:
        """Create all possible connections."""
        n = network.num_neurons
        for i in range(n):
            for j in range(n):
                if i == j and not self.self_connections:
                    continue
                network.connect(i, j, self.initial_weight)
