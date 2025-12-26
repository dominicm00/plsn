# AGENTS.md

This file provides guidance for AI agents working on this codebase.

## Project Overview

See [README.md](./README.md) for full architecture documentation.

**TL;DR:** This is a neural network library where neurons have spatial positions and connect based on distance. Key abstractions:

- `Neuron` – position + bands + global/local marker
- `LatticeNetwork` – neurons + N×N weight matrix (None = no connection)
- `NetworkBuilder` – fluent API for construction
- `PositionInitializer` / `DistanceDistribution` / `ConnectionInitializer` – protocols for extensibility

## Key Design Decisions

1. **`None` vs `0` weights** – `None` means no connection exists (cannot be learned). `0.0` means connection exists but has zero weight (can be adjusted during training).

2. **Bands are separate between neurons** – When aggregating inputs, each band sums only from the same band of connected neurons. Mixing happens inside each neuron via a B×B matrix.

3. **L1 (Manhattan) distance by default** – Configurable via `ord` parameter.

## Development Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Start Jupyter
uv run jupyter lab
```

## Code Locations

| What | Where |
|------|-------|
| Neuron class | `lattice_nets/core/neuron.py` |
| Network class | `lattice_nets/core/network.py` |
| Builder | `lattice_nets/builder.py` |
| Initializers | `lattice_nets/init/` |
| Examples | `visualizations.ipynb` |
