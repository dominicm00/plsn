import numpy as np
from scipy.sparse import csr_array

from plsn.builder import NetworkBuilder


def _forward_reference(net, inputs: np.ndarray) -> np.ndarray:
    """Reference implementation matching the pre-vectorization behavior."""
    weights_csr = csr_array(net.weights)
    weighted_inputs = weights_csr.T @ inputs  # (n, b)

    out = np.zeros((net.num_neurons, net.num_bands), dtype=np.float32)
    for j, neuron in enumerate(net.neurons):
        out[j] = neuron.process_bands(weighted_inputs[j])
    return out


def test_forward_matches_reference() -> None:
    net = NetworkBuilder().with_neurons(10).with_dimensions(2).with_bands(4).build()

    # Ensure we have non-trivial connectivity; builder defaults to no connections.
    rng = np.random.default_rng(0)
    for i in range(net.num_neurons):
        j = (i + 1) % net.num_neurons
        net.connect(i, j, weight=float(rng.normal()))

    # Ensure non-trivial per-neuron band mixing.
    for i, neuron in enumerate(net.neurons):
        local_rng = np.random.default_rng(i + 123)
        neuron.band_weights = local_rng.normal(
            size=(net.num_bands, net.num_bands)
        ).astype(np.float32)

    inputs = rng.normal(size=(net.num_neurons, net.num_bands)).astype(np.float32)

    got = net.forward(inputs)
    want = _forward_reference(net, inputs)

    assert got.dtype == np.float32
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


