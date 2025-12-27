"""NetworkBuilder - fluent API for constructing lattice networks."""

from dataclasses import dataclass, field
from typing import Self, Callable

import numpy as np

from plsn.core.neuron import Neuron
from plsn.core.network import LatticeNetwork, NeuronRanges
from plsn.core.learning import HebbianConfig
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
            .with_output_initializer(
                neurons=2,
                position=LatticePositionInitializer(),
                connections=[FullyConnectedInitializer()]
            )
            .build())

    Neuron layout: [inputs | model | outputs]
    - Input neurons: indices 0 to num_inputs-1
    - Model neurons: indices num_inputs to num_inputs+num_model-1
    - Output neurons: indices num_inputs+num_model to total-1

    Connection initializers are applied to the full network:
    - Input connection initializers create input->model connections
    - Model connection initializers create model->model connections
    - Output connection initializers create model->output connections
    """
    
    def __init__(self) -> None:
        self._dimensions: int = 2
        self._num_bands: int = 1
        self._model_config: LayerConfig | None = None
        self._input_config: LayerConfig | None = None
        self._output_config: LayerConfig | None = None
        self._seed: int | None = None
        self._activation: Callable[[np.float32], np.float32] | None = None
        self._learning_config: HebbianConfig | None = None
    
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

    def with_output_initializer(
        self,
        neurons: int,
        position: PositionInitializer | None = None,
        connections: list[ConnectionInitializer] | None = None,
    ) -> Self:
        """Configure output neuron initialization.

        Output neurons receive connections from model neurons and produce the
        network's output. Each output neuron has learnable weights that mix its
        bands down to a single scalar value.

        Args:
            neurons: Number of output neurons.
            position: Position initialization strategy.
            connections: Connection initialization strategies for model-to-output connections.
        """
        self._output_config = LayerConfig(
            num_neurons=neurons,
            position_init=position or LatticePositionInitializer(),
            connection_inits=connections or [],
        )
        return self

    def with_seed(self, seed: int) -> Self:
        """Set random seed for reproducibility."""
        self._seed = seed
        return self

    def with_activation(
        self, activation: Callable[[np.float32], np.float32]
    ) -> Self:
        """Set the activation function for the network.

        Args:
            activation: A callable that takes a float and returns a float.
                       The network will map this function over the state element-wise.
        """
        self._activation = activation
        return self

    def with_learning(
        self,
        config: HebbianConfig | None = None,
        learning_rate: float = 0.01,
        band_learning_rate: float = 0.01,
        bcm_tau: float = 100.0,
        theta_init: float = 0.5,
    ) -> Self:
        """Configure Hebbian learning for the network.

        Uses Oja-BCM combined rule for both connection weights and band weights.

        Args:
            config: Full HebbianConfig object (overrides other args if provided).
            learning_rate: Learning rate for connection weights.
            band_learning_rate: Learning rate for band mixing weights.
            bcm_tau: BCM threshold time constant.
            theta_init: Initial BCM threshold.
        """
        if config is not None:
            self._learning_config = config
        else:
            self._learning_config = HebbianConfig(
                learning_rate=learning_rate,
                band_learning_rate=band_learning_rate,
                bcm_tau=bcm_tau,
                theta_init=theta_init,
            )
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
        num_outputs = self._output_config.num_neurons if self._output_config else 0

        neurons: list[Neuron] = []

        # Create input neurons first
        if self._input_config:
            pos_rng = rng.spawn(1)[0]
            input_positions = self._input_config.position_init.initialize(
                self._input_config.num_neurons, self._dimensions, pos_rng
            )
            for pos in input_positions:
                neurons.append(Neuron(position=pos))

        # Create model neurons
        pos_rng = rng.spawn(1)[0]
        model_positions = self._model_config.position_init.initialize(
            num_model, self._dimensions, pos_rng
        )
        for pos in model_positions:
            neurons.append(Neuron(position=pos))

        # Create output neurons last
        if self._output_config:
            pos_rng = rng.spawn(1)[0]
            output_positions = self._output_config.position_init.initialize(
                self._output_config.num_neurons, self._dimensions, pos_rng
            )
            for pos in output_positions:
                neurons.append(Neuron(position=pos))

        ranges = NeuronRanges(
            input_start=0,
            input_end=num_inputs,
            model_start=num_inputs,
            model_end=num_inputs + num_model,
            output_start=num_inputs + num_model,
            output_end=num_inputs + num_model + num_outputs,
        )

        network = LatticeNetwork(
            neurons=neurons,
            num_bands=self._num_bands,
            ranges=ranges,
            activation=self._activation,
            learning_config=self._learning_config,
        )

        # Model-to-model connections
        conn_rngs = rng.spawn(len(self._model_config.connection_inits))
        for init, conn_rng in zip(self._model_config.connection_inits, conn_rngs):
            init.initialize(network, conn_rng, ranges.model_range, ranges.model_range)

        # Input-to-model connections
        if self._input_config:
            input_conn_rngs = rng.spawn(len(self._input_config.connection_inits))
            for init, conn_rng in zip(self._input_config.connection_inits, input_conn_rngs):
                init.initialize(network, conn_rng, ranges.input_range, ranges.model_range)

        # Model-to-output connections
        if self._output_config:
            output_conn_rngs = rng.spawn(len(self._output_config.connection_inits))
            for init, conn_rng in zip(self._output_config.connection_inits, output_conn_rngs):
                init.initialize(network, conn_rng, ranges.model_range, ranges.output_range)

        return network
