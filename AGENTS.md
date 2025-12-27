# AGENTS.md

This file provides guidance for AI agents working on this codebase.

## Project Overview

See [README.md](./README.md) for full architecture documentation.

**TL;DR:** This is a neural network library where neurons have spatial positions and connect based on distance. Key abstractions:

- `Neuron` – position + bands
- `LatticeNetwork` – neurons + N×N weight matrix (None = no connection)
- `NetworkBuilder` – fluent API for construction
- `PositionInitializer` / `DistanceDistribution` / `ConnectionInitializer` – protocols for extensibility

## Development

Use `uv` or the uv-generated `.venv` to run code.

## Code Locations

| What | Where |
|------|-------|
| Neuron class | `plsn/core/neuron.py` |
| Network class | `plsn/core/network.py` |
| Builder | `plsn/builder.py` |
| Initializers | `plsn/init/` |
| Examples | `visualizations.ipynb` |
