# Changelog

## 0.1.0 — 2026-02-13

Initial MVP release.

### Added
- **CLI** (`radix-core`): `status`, `submit`, `plan`, `info`, `validate` commands
- **Scheduling**: FIFO, Priority, FairShare, ShortestJobFirst policies
- **Planning**: Greedy and Optimal execution planners with dependency-aware topological sort
- **Placement**: Local single-node and LoadBalanced multi-node strategies
- **Job Graph**: DAG with cycle detection, topological sort, critical path, parallel levels
- **Executors**: ThreadPool (built-in), Ray local-mode, HuggingFace, vLLM (optional)
- **Batching**: Dynamic SLA-aware batcher, memory-efficient microbatcher with tensor size estimation
- **Safety**: Dry-run mode (enforced), $0.00 cost cap, no-deploy mode, network guard, deployment guard
- **Cost Simulator**: Job, batch, schedule, and swarm cost estimation (always $0.00 in dry-run)
- **Configuration**: Pydantic-based with immutable safety settings and env var support
- **Logging**: Structured logging with correlation IDs, safety-aware formatting
- **Utilities**: High-precision timers, SLA monitoring, seeded failure injection
- **CI**: GitHub Actions with lint (ruff) and tests (pytest) across Python 3.9/3.10/3.11
- **Docker**: Dockerfile with safety env vars baked in
- **Tests**: 159 tests covering types, config, scheduling, safety, batching, CLI, and integration
