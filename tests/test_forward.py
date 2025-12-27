from plsn.init.distributions.distance import StepDistanceDistribution
from plsn.init.connections import DistanceBasedInitializer
from plsn.init.positions import LatticePositionInitializer
import numpy as np
from scipy.sparse import csr_array

from plsn.builder import NetworkBuilder


def _forward_reference(net) -> np.ndarray:
    """Reference implementation matching the pre-vectorization behavior."""
    weights_csr = csr_array(net.weights)
    
    weighted_inputs = weights_csr.T @ net.state  # (n, b)

    out = np.zeros((net.num_neurons, net.num_bands), dtype=np.float32)
    for j, neuron in enumerate(net.neurons):
        out[j] = neuron.process_bands(weighted_inputs[j])
    
    if net.num_inputs > 0:
        out[net.ranges.input_slice] = net.state[net.ranges.input_slice]
    
    return out


def test_forward_matches_reference() -> None:
    net = (
        NetworkBuilder()
        .with_model_initializer(neurons=10)
        .with_dimensions(2)
        .with_bands(4)
        .with_seed(42)
        .build()
    )

    rng = np.random.default_rng(0)
    for i in range(net.num_neurons):
        j = (i + 1) % net.num_neurons
        net.connect(i, j, weight=float(rng.normal()))

    for i, neuron in enumerate(net.neurons):
        local_rng = np.random.default_rng(i + 123)
        neuron.band_weights = local_rng.normal(
            size=(net.num_bands, net.num_bands)
        ).astype(np.float32)

    net.state = rng.normal(size=(net.num_neurons, net.num_bands)).astype(np.float32)
    
    expected = _forward_reference(net)
    net.forward()

    assert net.state.dtype == np.float32
    np.testing.assert_allclose(net.state, expected, rtol=1e-6, atol=1e-6)


def test_forward_with_input_neurons() -> None:
    net = (
        NetworkBuilder()
        .with_model_initializer(
            neurons=6,
            position=LatticePositionInitializer(),
            connections=[DistanceBasedInitializer(StepDistanceDistribution(4))],
        )
        .with_input_initializer(
            neurons=4,
            position=LatticePositionInitializer(),
            connections=[DistanceBasedInitializer(StepDistanceDistribution(4))],
        )
        .with_dimensions(2)
        .with_bands(3)
        .with_seed(42)
        .build()
    )
    
    assert net.num_inputs == 4
    assert net.num_neurons == 10

    input_values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    net.set_input(input_values)
    
    net.forward()
    
    assert net.state.shape == (net.num_neurons, net.num_bands)
    
    for i in range(net.num_inputs):
        expected = np.full(net.num_bands, input_values[i], dtype=np.float32)
        np.testing.assert_allclose(net.state[i], expected)


def test_set_input_validates_shape() -> None:
    net = (
        NetworkBuilder()
        .with_model_initializer(neurons=4)
        .with_input_initializer(neurons=2)
        .with_seed(42)
        .build()
    )

    try:
        net.set_input(np.array([1.0, 2.0, 3.0]))
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Expected input shape (2,)" in str(e)


def test_forward_with_output_neurons() -> None:
    net = (
        NetworkBuilder()
        .with_model_initializer(neurons=4)
        .with_output_initializer(neurons=2)
        .with_dimensions(2)
        .with_bands(3)
        .with_seed(42)
        .build()
    )

    assert net.num_outputs == 2
    assert net.num_neurons == 6
    assert net.output_weights.shape == (2, 3)

    rng = np.random.default_rng(42)
    for i in range(4):
        for j in range(4, 6):
            net.connect(i, j, weight=float(rng.normal()))

    net.state = rng.normal(size=(net.num_neurons, net.num_bands)).astype(np.float32)

    output = net.forward()

    assert output.shape == (2,)
    assert output.dtype == np.float32


def test_output_weights_affect_output() -> None:
    net = (
        NetworkBuilder()
        .with_model_initializer(neurons=2)
        .with_output_initializer(neurons=1)
        .with_dimensions(2)
        .with_bands(3)
        .with_seed(42)
        .build()
    )

    for i in range(2):
        net.connect(i, 2, weight=1.0)

    net.state = np.ones((3, 3), dtype=np.float32)

    output1 = net.forward()

    net.output_weights = np.array([[2.0, 2.0, 2.0]], dtype=np.float32)
    net.state = np.ones((3, 3), dtype=np.float32)

    output2 = net.forward()

    np.testing.assert_allclose(output2, output1 * 2)


def test_forward_returns_empty_without_output_neurons() -> None:
    net = (
        NetworkBuilder()
        .with_model_initializer(neurons=4)
        .with_seed(42)
        .build()
    )

    output = net.forward()

    assert output.shape == (0,)
    assert output.dtype == np.float32


def test_cannot_connect_from_output_neuron() -> None:
    net = (
        NetworkBuilder()
        .with_model_initializer(neurons=2)
        .with_output_initializer(neurons=2)
        .with_seed(42)
        .build()
    )

    try:
        net.connect(2, 0)
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert "Cannot connect from output neuron" in str(e)
