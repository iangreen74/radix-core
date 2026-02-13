# Information-Theoretic GPU Scheduler Agent

FastAPI microservice that implements information-theoretic GPU scheduling using Bayesian learning and active exploration.

## Features

- **Information-Theoretic Scoring**: Uses uncertainty and information gain to balance exploration vs exploitation
- **Bayesian Learning**: Learns performance patterns for (job_type, gpu_type) combinations
- **Interference Detection**: Learns and avoids problematic job co-locations
- **Prometheus Metrics**: Comprehensive monitoring and observability
- **Configurable Parameters**: Runtime tuning via environment variables
- **Persistent Storage**: SQLite database with periodic checkpointing

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python -m uvicorn app.main:app --reload --port 8080

# Test health endpoint
curl http://localhost:8080/healthz
```

### Docker

```bash
# Build image
docker build -t scheduler-agent .

# Run container
docker run -p 8080:8080 scheduler-agent
```

### Kubernetes

```bash
# Deploy via Helm
helm install scheduler-agent charts/scheduler-agent
```

## API Endpoints

### Scoring

**POST /v1/score** - Score a job for GPU assignment

```bash
curl -X POST http://localhost:8080/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "job_type": "train-bert",
    "features": {"gpu_mem_gb": 40, "batch_size": 16},
    "candidate_gpu_types": ["A100-80GB", "L4-24GB"],
    "colocated_job_types": []
  }'
```

### Learning

**POST /v1/observe** - Record runtime observation

```bash
curl -X POST http://localhost:8080/v1/observe \
  -H "Content-Type: application/json" \
  -d '{
    "job_type": "train-bert",
    "gpu_type": "A100-80GB",
    "measured_runtime_s": 73.4,
    "colocated_job_types": []
  }'
```

### Monitoring

**GET /metrics** - Prometheus metrics

Key metrics:
- `scheduler_decisions_total`: Total scheduling decisions
- `scheduler_exploration_ratio`: Current exploration percentage
- `scheduler_avg_uncertainty`: Average uncertainty across job-GPU pairs

## Configuration

Configure via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SCHEDULER_LAMBDA_UNCERTAINTY` | 0.3 | Uncertainty weight (λ) |
| `SCHEDULER_BETA_EXPLORATION` | 0.7 | Information gain weight (β) |
| `SCHEDULER_GAMMA_INTERFERENCE` | 0.0 | Interference penalty weight (γ) |
| `SCHEDULER_EXPLORATION_CAP` | 0.25 | Maximum exploration ratio |
| `SCHEDULER_ENABLE_INTERFERENCE` | false | Enable interference learning |

## Algorithm Details

### Objective Function

```
cost(j,g) = μ[j,g] + λ * sqrt(σ²[j,g]) - β * IG[j,g] + γ * interference_penalty(j,g, batch)
```

- **μ[j,g]**: Expected runtime mean
- **σ²[j,g]**: Runtime variance (uncertainty)
- **IG[j,g]**: Information gain = 0.5 * log(1 + σ²/τ²)
- **interference_penalty**: Learned slowdown from co-location

### Learning

- **Bayesian Updates**: Exponential moving average for mean and variance
- **Exploration Control**: Cap on exploration ratio to prevent thrashing
- **Interference Learning**: Track slowdown patterns between job types

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_exploration_cap.py -v
pytest tests/test_uncertainty_shrinks.py -v
pytest tests/test_interference_penalty.py -v
pytest tests/test_sinkhorn_optional.py -v
```

## Development

### Project Structure

```
app/
├── main.py           # FastAPI application
├── config.py         # Configuration management
├── model.py          # Core scheduling model
├── scoring.py        # Scoring logic and API models
└── sinkhorn.py       # Optional Sinkhorn assignment

tests/
├── test_exploration_cap.py       # Exploration control tests
├── test_uncertainty_shrinks.py   # Learning verification tests
├── test_interference_penalty.py  # Interference learning tests
└── test_sinkhorn_optional.py     # Sinkhorn algorithm tests
```

### Key Classes

- **SchedulerModel**: Core information-theoretic scheduler
- **ScoringService**: API service layer
- **JobFeatures**: Job feature extraction and validation
- **SinkhornSolver**: Optional batched assignment

### Adding New Features

1. **New Job Features**: Add to `JobFeatures` dataclass
2. **New Scoring Terms**: Modify objective function in `score_job`
3. **New Metrics**: Add Prometheus metrics in `main.py`
4. **New Configuration**: Add to `SchedulerConfig` in `config.py`

## Production Deployment

### Helm Configuration

```yaml
# values.yaml
config:
  lambdaUncertainty: 0.3
  betaExploration: 0.7
  enableInterference: true

persistence:
  enabled: true
  size: 10Gi

resources:
  requests:
    cpu: 250m
    memory: 256Mi
  limits:
    cpu: 500m
    memory: 512Mi
```

### Monitoring Setup

1. **ServiceMonitor**: Enabled by default for Prometheus scraping
2. **Dashboards**: Import Grafana dashboards for scheduler metrics
3. **Alerts**: Set up alerts for high exploration ratio or low certainty

### Security

1. **RBAC**: Minimal permissions required (no cluster resources accessed)
2. **Network Policies**: Restrict traffic to webhook and monitoring
3. **Security Context**: Runs as non-root user with read-only filesystem

## Troubleshooting

### Common Issues

1. **High Exploration Ratio**
   - Normal during cold start
   - Check `SCHEDULER_EXPLORATION_CAP` setting
   - Add observations via `/v1/observe`

2. **No Learning**
   - Verify observations are being recorded
   - Check SQLite database persistence
   - Look for errors in logs

3. **Poor Performance**
   - Tune λ, β parameters for your workload
   - Enable interference learning if co-location matters
   - Increase observation retention

### Debug Commands

```bash
# Check current configuration
curl http://localhost:8080/

# Monitor metrics
curl http://localhost:8080/metrics | grep scheduler_

# Check logs
kubectl logs -f deployment/scheduler-agent -n scheduler
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `pytest tests/`
5. Submit a pull request

## License

See [LICENSE](../../LICENSE) file for details.
