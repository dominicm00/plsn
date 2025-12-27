"""NetworkBuilder - fluent API for constructing lattice networks."""

from dataclasses import dataclass, field
from typing import Self

import numpy as np

from plsn.core.neuron import Neuron
from plsn.core.network import LatticeNetwork, NeuronRanges
from plsn.init.positions import (
    PositionInitializer,
    LatticePositionInitializer,
)
from plsn.init.connections import ConnectionInitializer


@dataclass
class LayerConfig:
    """Configuration for a layer of neurons (model or input).
    
    Attributes:
        num_neurons: Number of neurons in this layer.
        position_init: Position initialization strategy.
        connection_inits: Connection initialization strategies.
    """
    num_neurons: int = 16
    position_init: PositionInitializer = field(default_factory=LatticePositionInitializer)
    connection_inits: list[ConnectionInitializer] = field(default_factory=list)


class NetworkBuilder:
    """Fluent builder for constructing LatticeNetwork instances.

    Example:
        network = (NetworkBuilder()
            .with_dimensions(2)
            .with_bands(4)
            .with_model_initializer(
                neurons=16,
                position=LatticePositionInitializer(),
                connections=[DistanceBasedInitializer(LinearDistanceDistribution())]
            )
            .with_input_initializer(
                neurons=4,
                position=LatticePositionInitializer(),
                connections=[FullyConnectedInitializer()]
            )
            .build())

    Input neurons are placed first in the network (indices 0 to num_inputs-1),
    followed by model neurons (indices num_inputs to num_neurons-1).
    
    Connection initializers are applied to the full network:
    - Input connection initializers typically create input->model connections
    - Model connection initializers typically create model->model connections
    """
    
    def __init__(self) -> None:
        self._dimensions: int = 2
        self._num_bands: int = 1
        self._model_config: LayerConfig | None = None
        self._input_config: LayerConfig | None = None
        self._seed: int | None = None
    
    def with_dimensions(self, dims: int) -> Self:
        """Set the dimensionality of the neuron position space."""
        self._dimensions = dims
        return self
    
    def with_bands(self, num_bands: int) -> Self:
        """Set the number of bands per neuron."""
        self._num_bands = num_bands
        return self
    
    def with_model_initializer(
        self,
        neurons: int = 16,
        position: PositionInitializer | None = None,
        connections: list[ConnectionInitializer] | None = None,
    ) -> Self:
        """Configure model neuron initialization.
        
        Args:
            neurons: Number of model neurons.
            position: Position initialization strategy.
            connections: Connection initialization strategies for model-to-model connections.
        """
        self._model_config = LayerConfig(
            num_neurons=neurons,
            position_init=position or LatticePositionInitializer(),
            connection_inits=connections or [],
        )
        return self
    
    def with_input_initializer(
        self,
        neurons: int,
        position: PositionInitializer | None = None,
        connections: list[ConnectionInitializer] | None = None,
    ) -> Self:
        """Configure input neuron initialization.
        
        Input neurons have fixed values and their outputs connect to model neurons.
        
        Args:
            neurons: Number of input neurons.
            position: Position initialization strategy.
            connections: Connection initialization strategies for input-to-model connections.
        """
        self._input_config = LayerConfig(
            num_neurons=neurons,
            position_init=position or LatticePositionInitializer(),
            connection_inits=connections or [],
        )
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
        if self._model_config is None:
            self._model_config = LayerConfig()
        
        rng = np.random.default_rng(self._seed)
        
        num_inputs = self._input_config.num_neurons if self._input_config else 0
        num_model = self._model_config.num_neurons
        
        neurons: list[Neuron] = []
        
        if self._input_config:
            pos_rng = rng.spawn(1)[0]
            input_positions = self._input_config.position_init.initialize(
                self._input_config.num_neurons, self._dimensions, pos_rng
            )
            for pos in input_positions:
                neurons.append(Neuron(position=pos, num_bands=self._num_bands))
        
        pos_rng = rng.spawn(1)[0]
        model_positions = self._model_config.position_init.initialize(
            num_model, self._dimensions, pos_rng
        )
        for pos in model_positions:
            neurons.append(Neuron(position=pos, num_bands=self._num_bands))
        
        ranges = NeuronRanges(
            input_start=0,
            input_end=num_inputs,
            model_start=num_inputs,
            model_end=num_inputs + num_model,
        )
        
        network = LatticeNetwork(
            neurons=neurons,
            num_bands=self._num_bands,
            ranges=ranges,
        )
        
        conn_rngs = rng.spawn(len(self._model_config.connection_inits))
        for init, conn_rng in zip(self._model_config.connection_inits, conn_rngs):
            init.initialize(network, conn_rng, ranges.model_range, ranges.model_range)

        if self._input_config:
            input_conn_rngs = rng.spawn(len(self._input_config.connection_inits))
            for init, conn_rng in zip(self._input_config.connection_inits, input_conn_rngs):
                init.initialize(network, conn_rng, ranges.input_range, ranges.model_range)
        
        return network
