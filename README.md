# radix-core

GPU Orchestration Platform — scheduling, placement, execution, and governance for GPU workloads on Kubernetes.

## Architecture

```
radix_core/              Core library (types, config, scheduling, executors)
  scheduler/             GPU-aware job scheduling and placement
  executors/             HuggingFace, vLLM, Ray execution backends
  batching/              Dynamic microbatching
  cli/                   CLI interface (radix-core command)
agents/                  Standalone microservices
  scheduler-agent/       Information-theoretic GPU scoring (FastAPI)
  scheduler-webhook/     K8s mutating admission webhook
  k8s-agent/             GPU metrics collection via nvidia-smi
  observer/              Observability agent
services/
  cluster_agent/         External cluster GPU workload executor
  gpu_policy/            Per-user/per-cluster GPU governance
infra/
  aws/                   EKS GPU node groups, VPC, NVIDIA drivers
charts/                  Helm charts for K8s deployment
gitops/                  Kueue queues, OPA/Rego policies, demo jobs
benchmarks/              ResNet-50 GPU benchmark
radix-bench/             Scheduler comparison benchmark suite
```

## Quick Start

```bash
# Install
pip install -e .

# Run tests
make test

# Lint
make lint
```

## Key Components

- **Scheduler**: Load-balanced GPU placement with resource tracking (CPU, memory, GPU count)
- **Executors**: HuggingFace (Accelerator + device_map), vLLM (tensor parallelism), Ray (distributed GPU)
- **Kueue Integration**: A100-80GB, A100-40GB, L4-24GB, H100-80GB resource flavors with quotas
- **Scheduler Agent**: Bayesian information-theoretic scoring across job-type x GPU-type combinations
- **GPU Governance**: Per-user job limits, OPA/Rego admission policies, tenant GPU licensing
- **Benchmarking**: ResNet-50 training, multi-scheduler comparison (DRF, HEFT, FIFO, BFD, SRPT, Gavel)

## License

MIT License - See [LICENSE](LICENSE)
