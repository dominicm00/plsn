
import numpy as np
from plsn.builder import NetworkBuilder
from plsn.core.network import LatticeNetwork

def custom_activation(x):
    # Identity activation for testing
    return x

def test_default_relu_activation():
    # Build network without specifying activation
    network = (NetworkBuilder()
        .with_dimensions(1)
        .with_model_initializer(neurons=5)
        .build())
    
    assert network.activation.__name__ == 'relu'

    # Test logic
    # Note: relu now takes float
    assert network.activation(-1.0) == 0.0
    assert network.activation(1.0) == 1.0

def test_custom_activation():
    network = (NetworkBuilder()
        .with_dimensions(1)
        .with_model_initializer(neurons=5)
        .with_activation(custom_activation)
        .build())

    # Test that custom activation behavior is preserved (identity function)
    assert network.activation(-1.0) == -1.0
    assert network.activation(5.0) == 5.0

