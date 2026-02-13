# radix-core

GPU Orchestration Platform — scheduling, placement, execution, and governance for GPU workloads.

## Quick Start

```bash
# Install
pip install -e .

# Run CLI
radix-core status
radix-core info
radix-core submit "echo hello" --name my-job
radix-core plan --count 5

# Run tests
pytest tests/

# Lint
ruff check radix_core/ tests/
```

## Architecture

```
radix_core/                    Core Python package
  cli/                         CLI interface (radix-core command)
  scheduler/                   Job scheduling, placement, and dependency DAGs
    policies.py                FIFO, Priority, FairShare, ShortestJobFirst
    planner.py                 Greedy and Optimal execution planners
    placement.py               Local and LoadBalanced placement strategies
    job_graph.py               DAG with topological sort, critical path, parallel levels
  executors/                   Execution backends
    threadpool.py              Thread-based job execution with monitoring
    hf_runner.py               HuggingFace Transformers inference
    vllm_local.py              vLLM high-throughput inference
    ray_local.py               Ray local-mode map/reduce
  batching/                    Batch processing
    dynamic_batcher.py         SLA-aware adaptive batching
    microbatch.py              Memory-efficient microbatching with tensor size estimation
  utils/                       Timers, SLA monitoring, failure injection
  config.py                    Pydantic-based configuration with safety validation
  types.py                     Job, JobResult, ResourceRequirements, SchedulePlan
  dryrun.py                    Dry-run safety guards (all operations simulated)
  cost_simulator.py            Cost estimation (always $0.00 in dry-run)
  errors.py                    Error hierarchy
  logging.py                   Structured logging with correlation IDs

agents/                        Microservices
  dashboard/                   Web dashboard (FastAPI + Jinja2)
  observer/                    Cluster observer (GPU nodes, timeseries)
  scheduler-agent/             Information-theoretic GPU scheduler
  scheduler-webhook/           Kubernetes admission webhook

charts/                        Helm charts for Kubernetes deployment
infra/                         Terraform infrastructure (AWS EKS)
gitops/                        Kubernetes manifests and policy
```

## Dashboard

The web dashboard provides a live view of the GPU orchestration platform:

- **Overview** — GPU nodes, pending jobs, estimated improvement, scheduler status
- **Jobs** — Submit and track GPU jobs, view scheduling scores
- **Metrics** — Timeseries charts, Prometheus scheduler metrics
- **Health** — Component health and readiness checks

```bash
# Run locally
cd agents/dashboard
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080

# Or via Docker
docker build -t radix-dashboard agents/dashboard/
docker run -p 8080:8080 radix-dashboard
```

The dashboard connects to the observer and scheduler-agent services. Configure via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OBSERVER_URL` | `http://radix-observer:8080` | Observer service URL |
| `SCHEDULER_URL` | `http://radix-scheduler-agent:8080` | Scheduler agent URL |
| `DASHBOARD_PORT` | `8080` | Dashboard listen port |

## Safety Model

radix-core is built with a **safety-first** architecture:

- **Dry-run by default** — all operations are simulated; no real resources consumed
- **Cost cap at $0.00** — cannot be overridden; prevents accidental spending
- **No-deploy mode** — deployment operations (kubectl, docker, terraform) are blocked
- **Local-only execution** — Ray runs in local mode; no external cluster connections
- **Network guard** — only localhost connections allowed

These constraints are enforced at the configuration level via Pydantic validators that reject unsafe values.

## CLI Commands

| Command    | Description                                      |
|------------|--------------------------------------------------|
| `status`   | Show safety config, execution config, batching   |
| `submit`   | Submit a job with resource requirements           |
| `plan`     | Create an execution plan for sample jobs          |
| `info`     | Show available components and their status        |
| `validate` | Run all validation checks (config, safety, imports) |

## Components

### Scheduling Policies
- **FIFO** — first-in-first-out by creation time
- **Priority** — weighted scoring (base priority + age + resource efficiency)
- **FairShare** — fair resource allocation across users/projects
- **ShortestJobFirst** — minimize average waiting time

### Execution Backends
- **ThreadPool** — built-in, always available
- **Ray** — optional, local-mode only (`pip install ray[default]`)
- **HuggingFace** — optional (`pip install transformers accelerate`)
- **vLLM** — optional, requires CUDA (`pip install vllm`)

### Batching
- **DynamicBatcher** — adaptive batch sizing with SLA awareness
- **MicrobatchProcessor** — memory-aware batch fragmentation using tensor size estimation

## Development

```bash
# Clone and set up
git clone https://github.com/iangreen74/radix-core.git
cd radix-core
cp .env.example .env
make install-dev

# Run all checks
make lint                 # ruff
make format               # black + isort (check)
make typecheck            # mypy
make security             # bandit
make test                 # pytest
make test-cov             # pytest with coverage

# All available targets
make help
```

## CI/CD Pipeline

The GitHub Actions pipeline runs on every push and PR:

| Stage | What it does |
|-------|-------------|
| **Lint** | `ruff check` on all Python code |
| **Format** | `black --check` and `isort --check` |
| **Type Check** | `mypy` with strict mode |
| **Security** | `bandit` security scanner |
| **Tests** | `pytest` with coverage across Python 3.9/3.10/3.11 |
| **Integration** | Integration tests + CLI smoke tests |
| **Build** | Package build (sdist + wheel) and verification |
| **Docker** | Build and verify radix-core and dashboard containers |
| **Helm Lint** | Validate all Helm charts |

## Docker

```bash
# Core CLI
docker build -t radix-core .
docker run radix-core status
docker run radix-core validate

# Dashboard
docker build -t radix-dashboard agents/dashboard/
docker run -p 8080:8080 radix-dashboard
```

## Kubernetes Deployment

```bash
# Install with Helm
helm install radix charts/radix-core/ -n radix --create-namespace

# Verify
kubectl -n radix get pods
kubectl -n radix port-forward svc/radix-dashboard 8080:8080
```

## License

MIT License
