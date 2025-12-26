"""NetworkBuilder - fluent API for constructing lattice networks."""

from typing import Self

import numpy as np

from plsn.core.neuron import Neuron
from plsn.core.network import LatticeNetwork
from plsn.init.positions import (
    PositionInitializer,
    LatticePositionInitializer,
)
from plsn.init.connections import ConnectionInitializer

class NetworkBuilder:
    """Fluent builder for constructing LatticeNetwork instances.
    
    Example:
        network = (NetworkBuilder()
            .with_dimensions(2)
            .with_neurons(16)
            .with_bands(4)
            .with_position_initializer(LatticePositionInitializer())
            .with_connection_initializer(
                DistanceBasedInitializer(LinearDistribution())
            )
            .build())
    """
    
    def __init__(self) -> None:
        self._num_neurons: int = 16
        self._dimensions: int = 2
        self._num_bands: int = 1
        self._global_ratio: float = 0.0  # Ratio of global neurons
        self._position_init: PositionInitializer = LatticePositionInitializer()
        self._connection_init: ConnectionInitializer | None = None
        self._seed: int | None = None
    
    def with_neurons(self, count: int) -> Self:
        """Set the number of neurons in the network."""
        self._num_neurons = count
        return self
    
    def with_dimensions(self, dims: int) -> Self:
        """Set the dimensionality of the neuron position space."""
        self._dimensions = dims
        return self
    
    def with_bands(self, num_bands: int) -> Self:
        """Set the number of bands per neuron."""
        self._num_bands = num_bands
        return self
    
    def with_global_ratio(self, ratio: float) -> Self:
        """Set the ratio of global vs local neurons (0 to 1)."""
        self._global_ratio = np.clip(ratio, 0.0, 1.0)
        return self
    
    def with_position_initializer(self, init: PositionInitializer) -> Self:
        """Set the position initialization strategy."""
        self._position_init = init
        return self
    
    def with_connection_initializer(self, init: ConnectionInitializer) -> Self:
        """Set the connection initialization strategy."""
        self._connection_init = init
        return self
    
    def with_seed(self, seed: int) -> Self:
        """Set random seed for reproducibility."""
        self._seed = seed
        return self
    
    def build(self) -> LatticeNetwork:
        """Construct the network with all configured parameters.
        
        Returns:
            A fully initialized LatticeNetwork.
        """
        rng = np.random.default_rng(self._seed)
        
        # Generate positions
        positions = self._position_init.initialize(
            self._num_neurons, self._dimensions
        )
        
        # Determine which neurons are global
        num_global = int(self._global_ratio * self._num_neurons)
        global_indices = set(
            rng.choice(self._num_neurons, size=num_global, replace=False)
        )
        
        # Create neurons
        neurons = []
        for i, pos in enumerate(positions):
            neuron = Neuron(
                position=pos,
                is_global=(i in global_indices),
                num_bands=self._num_bands,
            )
            neurons.append(neuron)
        
        # Create network
        network = LatticeNetwork(neurons=neurons, num_bands=self._num_bands)
        
        # Initialize connections
        if self._connection_init is not None:
            self._connection_init.initialize(network)
        
        return network
