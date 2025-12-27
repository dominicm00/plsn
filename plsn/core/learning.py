"""Hebbian learning configuration and rules."""

from dataclasses import dataclass


@dataclass
class HebbianConfig:
    """Configuration for Oja-BCM Hebbian learning.

    The combined Oja-BCM rule is:
        dw = eta * y * (y - theta) * (x - y * w)

    Where:
        - y: postsynaptic (output) activation
        - x: presynaptic (input) activation
        - theta: BCM sliding threshold (adapts toward y²)
        - w: current weight

    BCM provides bidirectional plasticity (LTP when y > theta, LTD when y < theta).
    Oja's normalization term (x - y * w) prevents unbounded weight growth.

    Attributes:
        learning_rate: Learning rate for connection weights. Default 0.01.
        band_learning_rate: Learning rate for band mixing weights. Default 0.01.
        output_learning_rate: Learning rate for output mixing weights. Default 0.01.
        bcm_tau: Time constant for BCM threshold adaptation.
                 Higher values = slower adaptation. Default 100.0.
        theta_init: Initial BCM threshold value. Default 0.5.
        weight_decay: Optional L2 weight decay coefficient. Default 0.0.
        min_weight: Minimum weight value (clipping). Default -inf.
        max_weight: Maximum weight value (clipping). Default +inf.
    """
    learning_rate: float = 0.01
    band_learning_rate: float = 0.01
    output_learning_rate: float = 0.01
    bcm_tau: float = 100.0
    theta_init: float = 0.5
    weight_decay: float = 0.0
    min_weight: float = float('-inf')
    max_weight: float = float('inf')
