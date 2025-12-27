"""PLSN (Plastic Local Sparse Network) - A modular neural network architecture with spatial connectivity."""

from plsn.core.neuron import Neuron
from plsn.core.network import LatticeNetwork
from plsn.core.learning import HebbianConfig
from plsn.builder import NetworkBuilder

__all__ = [
    "Neuron",
    "LatticeNetwork",
    "HebbianConfig",
    "NetworkBuilder",
]
