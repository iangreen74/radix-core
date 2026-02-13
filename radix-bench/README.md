# Radix-Bench: GPU Scheduler Benchmarking Suite

A comprehensive benchmarking framework for GPU scheduling algorithms with statistical rigor and reproducible results.

## Features

- **8-Scheduler League**: FIFO, SRPT, DRF, EASY, HEFT, BFD, Gavel, Radix v0
- **Statistical Harness**: Bootstrap confidence intervals, effect sizes, win-rate analysis
- **Cost Model**: GPU hourly rates, budget constraints, cost optimization
- **Golden Baselines**: Hash-based reproducibility verification
- **Deterministic Execution**: Reproducible results across runs

## Quick Start

```bash
# Install dependencies
poetry install

# Run smoke test
poetry run python -m radixbench.cli.simulate --config configs/experiments/smoke.yaml

# Run full league analysis
poetry run python -m radixbench.cli.league --config configs/experiments/hetero.yaml
```

## Scheduler Implementations

- **FIFO**: First-In-First-Out baseline
- **SRPT**: Shortest Remaining Processing Time
- **DRF**: Dominant Resource Fairness
- **EASY**: Extensible Argonne Scheduling System (backfilling)
- **HEFT**: Heterogeneous Earliest Finish Time
- **BFD**: Best Fit Decreasing (bin packing)
- **Gavel**: Utility/speedup-aware scheduling
- **Radix v0**: Information-theoretic scheduler with entropy optimization

## Performance Expectations

- **Radix v0**: 25% improvement over FIFO baseline
- **Statistical Validation**: 95% confidence intervals
- **Cost Optimization**: Budget-aware scheduling
- **Reproducible Research**: Golden hash verification
