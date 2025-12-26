"""Distance-based probability distributions for connection initialization."""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DistanceDistribution(Protocol):
    """Protocol for distance-to-probability distributions.
    
    Maps distance (in the same units as neuron positioning) to
    connection probability (0 to 1).
    
    With the default positioning where neurons are ~1 unit apart,
    distance values represent intuitive spacing:
    - distance = 1.0 means adjacent neurons in a lattice
    - distance = 2.0 means two steps apart
    """
    
    def probability(self, distance: float) -> float:
        """Compute connection probability for a given distance.
        
        Args:
            distance: Absolute distance between neurons. With default
                      positioning, 1.0 = adjacent neurons.
            
        Returns:
            Probability of connection in [0, 1].
        """
        ...


class LinearDistribution:
    """Linear interpolation of connection probability based on distance.
    
    P(d) = max(0, 1 - d / max_distance)
    
    Probability is 1.0 at distance=0 and decreases linearly to 0
    at max_distance.
    
    Args:
        max_distance: Distance at which probability becomes 0.
                      Default: 3.0 (covers ~3 neuron spacings).
    """
    
    def __init__(self, max_distance: float = 3.0) -> None:
        self.max_distance = max_distance
    
    def probability(self, distance: float) -> float:
        """Linear falloff of probability with distance."""
        return max(0.0, 1.0 - distance / self.max_distance)


class ExponentialDistribution:
    """Exponential decay of connection probability with distance.
    
    P(d) = exp(-d / scale)
    
    Args:
        scale: Characteristic distance scale. Probability drops to ~37%
               at distance=scale. Default: 1.0 (matches neuron spacing).
    """
    
    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale
    
    def probability(self, distance: float) -> float:
        """Exponential decay."""
        return float(np.exp(-distance / self.scale))


class GaussianDistribution:
    """Gaussian (bell-curve) probability distribution.
    
    P(d) = exp(-d² / (2 * sigma²))
    
    Args:
        sigma: Standard deviation of the Gaussian. ~68% of connections
               are within distance=sigma, ~95% within distance=2*sigma.
               Default: 1.5 (covers ~1.5 neuron spacings for most connections).
    """
    
    def __init__(self, sigma: float = 1.5) -> None:
        self.sigma = sigma
    
    def probability(self, distance: float) -> float:
        """Gaussian falloff."""
        return float(np.exp(-distance**2 / (2 * self.sigma**2)))


class StepDistribution:
    """Step function: connect if distance is below threshold.
    
    P(d) = 1 if d < threshold else 0
    
    Args:
        threshold: Maximum distance for connection.
                   Default: 2.0 (connect to neurons within 2 spacings).
    """
    
    def __init__(self, threshold: float = 2.0) -> None:
        self.threshold = threshold
    
    def probability(self, distance: float) -> float:
        """Step function."""
        return 1.0 if distance < self.threshold else 0.0
