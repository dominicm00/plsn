"""Distance-based probability distributions for connection initialization."""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DistanceDistribution(Protocol):
    """Protocol for distance-to-probability distributions.
    
    Maps normalized distance (0 to 1) to connection probability (0 to 1).
    """
    
    def probability(self, normalized_distance: float) -> float:
        """Compute connection probability for a given distance.
        
        Args:
            normalized_distance: Distance normalized to [0, 1] range,
                                 where 0 is closest and 1 is max distance.
            
        Returns:
            Probability of connection in [0, 1].
        """
        ...


class LinearDistribution:
    """Linear interpolation from high probability at distance=0 to low at distance=1.
    
    P(d) = high - (high - low) * d
    
    Default: (0, 1) -> (1, 0) meaning close neurons always connect,
    far neurons never connect.
    
    Args:
        prob_at_zero: Probability when distance is 0 (closest).
        prob_at_max: Probability when distance is max (farthest).
    """
    
    def __init__(
        self,
        prob_at_zero: float = 1.0,
        prob_at_max: float = 0.0,
    ) -> None:
        self.prob_at_zero = prob_at_zero
        self.prob_at_max = prob_at_max
    
    def probability(self, normalized_distance: float) -> float:
        """Linear interpolation between endpoints."""
        d = np.clip(normalized_distance, 0.0, 1.0)
        return self.prob_at_zero - (self.prob_at_zero - self.prob_at_max) * d


class ExponentialDistribution:
    """Exponential decay of connection probability with distance.
    
    P(d) = exp(-decay * d)
    
    Args:
        decay: Decay rate. Higher values = faster falloff.
               decay=3 means P(1) ≈ 0.05
    """
    
    def __init__(self, decay: float = 3.0) -> None:
        self.decay = decay
    
    def probability(self, normalized_distance: float) -> float:
        """Exponential decay."""
        d = np.clip(normalized_distance, 0.0, 1.0)
        return float(np.exp(-self.decay * d))


class GaussianDistribution:
    """Gaussian (bell-curve) probability distribution.
    
    P(d) = exp(-d² / (2 * sigma²))
    
    Args:
        sigma: Standard deviation of the Gaussian.
               sigma=0.3 means ~95% of probability is within d=0.6.
    """
    
    def __init__(self, sigma: float = 0.3) -> None:
        self.sigma = sigma
    
    def probability(self, normalized_distance: float) -> float:
        """Gaussian falloff."""
        d = np.clip(normalized_distance, 0.0, 1.0)
        return float(np.exp(-d**2 / (2 * self.sigma**2)))


class StepDistribution:
    """Step function: connect if distance is below threshold.
    
    P(d) = 1 if d < threshold else 0
    
    Args:
        threshold: Normalized distance threshold for connection.
    """
    
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
    
    def probability(self, normalized_distance: float) -> float:
        """Step function."""
        return 1.0 if normalized_distance < self.threshold else 0.0
