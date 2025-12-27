# PLSN (Plastic Local Sparse Network)

A modular neural network architecture where neurons have **spatial positions** and connect based on **distance-dependent probability distributions**.

## Architecture Overview

### Core Concepts

| Concept | Description |
|---------|-------------|
| **Neuron** | Has a d-dimensional position and B bands |
| **Bands** | Separate signal channels that mix only within each neuron via a B×B weight matrix |
| **Weights** | N×N matrix where `None` = no connection (immutable), `float` = adjustable weight |
| **Distance-based connectivity** | Neurons connect probabilistically based on L1 (Manhattan) distance |

### Band Processing

Each neuron has B bands. Between neurons, bands are kept separate (band 1 inputs only sum band 1 outputs from connected neurons). Within each neuron, bands mix through a dense B×B weight matrix.

```
          ┌─────────────────┐
Band 1 ──►│                 │──► Band 1
Band 2 ──►│  B×B weights    │──► Band 2
Band 3 ──►│  (mixing)       │──► Band 3
          └─────────────────┘
```

## Quick Start

```python
from plsn import NetworkBuilder
from plsn.init import DistanceBasedInitializer, LinearDistribution

network = (
    NetworkBuilder()
    .with_dimensions(2)          # 2D space
    .with_neurons(16)            # 4×4 lattice
    .with_bands(4)               # 4 signal bands
    .with_connection_initializer(
        DistanceBasedInitializer(LinearDistribution(), seed=42)
    )
    .build()
)

print(network)
# LatticeNetwork(neurons=16, dims=2, bands=4, connections=...)
```

## Extensibility

The architecture uses Python Protocols for modularity:

### Position Initializers
```python
from plsn.init import LatticePositionInitializer, RandomPositionInitializer

# Even grid distribution (default)
LatticePositionInitializer()

# Uniform random positions
RandomPositionInitializer(low=0.0, high=1.0, seed=42)
```

### Distance Distributions
```python
from plsn.init import (
    LinearDistribution,      # P(d) = 1 - d
    ExponentialDistribution, # P(d) = exp(-λd)
    GaussianDistribution,    # P(d) = exp(-d²/2σ²)
    StepDistribution,        # P(d) = 1 if d < threshold else 0
)
```

### Connection Initializers
```python
from plsn.init import DistanceBasedInitializer

DistanceBasedInitializer(
    distribution=LinearDistribution(),
    initial_weight=0.0,
    self_connections=False,
    bidirectional=False,
    ord=1,  # L1 norm (Manhattan distance)
    seed=42,
)
```

## Project Structure

```
plsn/
├── __init__.py           # Main exports
├── builder.py            # Fluent NetworkBuilder API
├── core/
│   ├── neuron.py         # Neuron with position, bands
│   └── network.py        # LatticeNetwork with N×N weight matrix
└── init/
    ├── positions.py      # Position initializers
    ├── distributions.py  # Distance → probability mappings
    └── connections.py    # Connection initializers
```

## Setup

1. Install `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies:
```bash
uv sync
```

3. Start Jupyter Lab:
```bash
uv run jupyter lab
```

## Google Colab

Colab can’t import local modules unless you either install the project or add the repo to `sys.path`. The most reliable approach is to clone the repo and install it via the following cell:

```python
!git clone https://github.com/dominicm00/plsn.git
%cd plsn
!pip install -e .
```

## Files

- `visualizations.ipynb` – Interactive examples with visualizations
- `pyproject.toml` – Project configuration (managed by `uv`)
