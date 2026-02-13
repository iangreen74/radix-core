# Changelog

## 0.2.0 — 2026-02-13

Dashboard, CI/CD, and developer experience improvements.

### Added
- **Dashboard**: FastAPI + Jinja2 web dashboard with 4 pages (overview, jobs, metrics, health)
- **Dashboard**: Job submission form with scheduler scoring integration
- **Dashboard**: Auto-refreshing overview with live GPU node and pending job counts
- **Dashboard**: Timeseries charts via Chart.js, Prometheus metrics table
- **Dashboard**: Component health checks with readiness probes
- **Dashboard**: Dockerfile and Helm chart wiring for Kubernetes deployment
- **CI/CD**: Format checking (black + isort)
- **CI/CD**: Type checking (mypy) with strict mode
- **CI/CD**: Security scanning (bandit)
- **CI/CD**: Coverage enforcement (70% threshold) across Python 3.9/3.10/3.11
- **CI/CD**: Integration test stage with CLI smoke tests
- **CI/CD**: Package build and wheel verification
- **CI/CD**: Docker build and container verification for both radix-core and dashboard
- **CI/CD**: Helm chart linting
- **Dev**: `poetry.lock` committed for reproducible installs
- **Dev**: `.python-version` file (3.11)
- **Dev**: `.env.example` with documented environment variables
- **Dev**: Expanded Makefile with `help`, `install-dev`, `test-cov`, `format`, `typecheck`, `security`, `docker-build` targets

### Changed
- Makefile updated with full dev workflow targets
- README expanded with dashboard docs, CI/CD pipeline table, dev setup, and Kubernetes deployment
- `.gitignore` updated to track `poetry.lock`

### Fixed
- Helm chart dashboard deployment command updated to match actual app module path

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
