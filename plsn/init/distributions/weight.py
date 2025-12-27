"""Weight initialization distributions."""

from typing import Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class WeightDistribution(Protocol):
    """Protocol for weight initialization distributions."""
    
    def sample(self, rng: np.random.Generator) -> float:
        """Sample a weight from the distribution.
        
        Args:
            rng: Random number generator to use.
            
        Returns:
            Sampled weight value.
        """
        ...

class ConstantWeightDistribution:
    """Always returns a constant weight."""
    
    def __init__(self, value: float = 1.0):
        self.value = value
        
    def sample(self, rng: np.random.Generator) -> float:
        return self.value

class UniformWeightDistribution:
    """Uniform distribution between low and high."""
    
    def __init__(self, low: float = -1.0, high: float = 1.0):
        self.low = low
        self.high = high
        
    def sample(self, rng: np.random.Generator) -> float:
        return rng.uniform(self.low, self.high)

class NormalWeightDistribution:
    """Normal (Gaussian) distribution."""
    
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std
        
    def sample(self, rng: np.random.Generator) -> float:
        return rng.normal(self.mean, self.std)

class LogNormalWeightDistribution:
    """Log-normal distribution."""

    def __init__(self, mean: float = 0.0, sigma: float = 1.0):
        self.mean = mean
        self.sigma = sigma

    def sample(self, rng: np.random.Generator) -> float:
        return rng.lognormal(self.mean, self.sigma)
