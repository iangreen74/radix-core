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
radix_core/
  cli/                   CLI interface (radix-core command)
  scheduler/             Job scheduling, placement, and dependency DAGs
    policies.py          FIFO, Priority, FairShare, ShortestJobFirst
    planner.py           Greedy and Optimal execution planners
    placement.py         Local and LoadBalanced placement strategies
    job_graph.py         DAG with topological sort, critical path, parallel levels
  executors/             Execution backends
    threadpool.py        Thread-based job execution with monitoring
    hf_runner.py         HuggingFace Transformers inference
    vllm_local.py        vLLM high-throughput inference
    ray_local.py         Ray local-mode map/reduce
  batching/              Batch processing
    dynamic_batcher.py   SLA-aware adaptive batching
    microbatch.py        Memory-efficient microbatching with tensor size estimation
  utils/                 Timers, SLA monitoring, failure injection
  config.py              Pydantic-based configuration with safety validation
  types.py               Job, JobResult, ResourceRequirements, SchedulePlan
  dryrun.py              Dry-run safety guards (all operations simulated)
  cost_simulator.py      Cost estimation (always $0.00 in dry-run)
  errors.py              Error hierarchy
  logging.py             Structured logging with correlation IDs
```

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
# Install dev dependencies
pip install -e .
pip install pytest ruff

# Run tests
pytest tests/ -v

# Lint
ruff check radix_core/ tests/

# CLI
radix-core validate
```

## Docker

```bash
docker build -t radix-core .
docker run radix-core status
docker run radix-core validate
```

## License

MIT License
