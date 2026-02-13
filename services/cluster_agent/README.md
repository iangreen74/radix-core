# Radix Cluster Agent

The Radix Cluster Agent runs on external GPU clusters and executes containerized workloads submitted via the Radix control plane.

## Features

- **Real Docker Execution**: Runs jobs as Docker containers with GPU support
- **Automatic Capabilities Detection**: Detects CPUs, RAM, and GPUs via nvidia-smi
- **Heartbeat & Health Monitoring**: Reports status every 5 minutes
- **Job Polling**: Long-polls for jobs every 30 seconds (configurable)
- **Retry Logic**: Automatic retry with exponential backoff for API calls
- **Secure Authentication**: Token-based auth with SHA-256 hashing

## Requirements

- Docker installed and running
- Docker socket accessible at `/var/run/docker.sock`
- NVIDIA drivers + nvidia-docker2 (for GPU support)
- Outbound HTTPS access to Radix API

## Installation

### Option 1: Docker (Recommended)

```bash
# Pull the agent image
docker pull iangreen74/radix-cluster-agent:latest

# Run the agent
docker run -d \
  --name radix-agent \
  --restart unless-stopped \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /tmp/radix-jobs:/tmp/radix-jobs \
  --gpus all \
  -e RADIX_API_BASE=https://api.vaultscaler.com \
  -e RADIX_CLUSTER_ID=cluster-abc123 \
  -e RADIX_CLUSTER_TOKEN=your_token_here \
  -e RADIX_TENANT_ID=demo-tenant \
  iangreen74/radix-cluster-agent:latest

# Check logs
docker logs -f radix-agent
```

**Note**: Remove `--gpus all` if no GPU is available.

### Option 2: Python (Development)

```bash
cd services/cluster_agent

# Install dependencies
pip install -r requirements.txt

# Run agent
python main.py \
  --api-base https://api.vaultscaler.com \
  --cluster-id cluster-abc123 \
  --cluster-token your_token_here \
  --tenant-id demo-tenant
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `RADIX_API_BASE` | Yes | - | Radix API base URL |
| `RADIX_CLUSTER_ID` | Yes | - | Cluster identifier |
| `RADIX_CLUSTER_TOKEN` | Yes | - | Cluster authentication token |
| `RADIX_TENANT_ID` | No | `demo-tenant` | Tenant identifier |
| `RADIX_POLL_INTERVAL_SECONDS` | No | `30` | Job polling interval |
| `RADIX_JOBS_WORKDIR` | No | `/tmp/radix-jobs` | Job working directory |
| `RADIX_EXECUTION_MODE` | No | `docker` | Execution mode (docker only for now) |

### Command-Line Arguments

All environment variables can also be passed as command-line arguments:

```bash
python main.py \
  --api-base https://api.vaultscaler.com \
  --cluster-id cluster-abc123 \
  --cluster-token your_token_here \
  --tenant-id demo-tenant \
  --poll-interval 30 \
  --jobs-workdir /tmp/radix-jobs \
  --execution-mode docker
```

## How It Works

### 1. Startup

- Detects cluster capabilities (CPUs, RAM, GPUs)
- Sends initial heartbeat to control plane
- Starts main loop

### 2. Main Loop

- Sends heartbeat every 5 minutes
- Polls for jobs every 30 seconds
- Executes jobs when available

### 3. Job Execution

For each job:

1. Creates job directory in `RADIX_JOBS_WORKDIR`
2. Writes `input.json` with job input payload
3. Runs Docker container:
   - Mounts job directory as `/workspace`
   - Passes environment variables
   - Enables GPU access if available
4. Captures stdout, stderr, exit code
5. Reads output from `/workspace/output.json` (or parses stdout)
6. Reports completion to control plane
7. Cleans up job directory

### 4. Output Handling

The agent looks for output in this order:

1. `/workspace/output.json` - if exists and valid JSON
2. Container stdout - if valid JSON
3. Wrapped stdout/stderr - fallback

Example output.json:

```json
{
  "result": "success",
  "output_text": "Generated text here",
  "metrics": {
    "tokens": 100,
    "latency_ms": 250
  }
}
```

## Job Specification

Jobs submitted to the cluster have this structure:

```json
{
  "job_id": "job-abc123",
  "cluster_id": "cluster-xyz789",
  "job_kind": "inference",
  "image": "myregistry/my-model:latest",
  "command": ["python", "inference.py"],
  "args": ["--input", "/workspace/input.json"],
  "env": {
    "MODEL_PATH": "/models/checkpoint.pt"
  },
  "input_payload": {
    "text": "The future of AI is",
    "max_length": 100
  }
}
```

## Security

- **Token Hashing**: Tokens are hashed with SHA-256 on the server
- **Token Transmission**: Tokens are sent via HTTPS only
- **Token Storage**: Never logged or stored in plaintext
- **Container Isolation**: Jobs run in isolated Docker containers
- **No Privileged Access**: Containers run without privileged mode

## Limitations

### Current Implementation

- ✅ Docker execution mode only
- ✅ Single job execution (no parallelism)
- ✅ CPU and GPU support
- ✅ 1-hour job timeout

### Not Yet Implemented

- ❌ Kubernetes-native job submission
- ❌ Slurm scheduler integration
- ❌ Parallel job execution
- ❌ Job priority queues
- ❌ Resource quotas per job
- ❌ Job cancellation from control plane

## Troubleshooting

### Agent won't start

```bash
# Check Docker socket access
docker ps

# Check environment variables
echo $RADIX_API_BASE
echo $RADIX_CLUSTER_ID
echo $RADIX_CLUSTER_TOKEN
```

### Jobs fail with "permission denied"

```bash
# Ensure Docker socket is mounted
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  alpine docker ps
```

### GPU not detected

```bash
# Check nvidia-smi
nvidia-smi

# Check nvidia-docker2
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Jobs timeout

- Default timeout is 1 hour
- Increase timeout in agent code if needed
- Check job logs: `docker logs <container_id>`

## Building the Image

```bash
cd services/cluster_agent

# Build
docker build -t iangreen74/radix-cluster-agent:latest .

# Push (requires Docker Hub login)
docker push iangreen74/radix-cluster-agent:latest
```

## Development

```bash
# Run locally
python main.py \
  --api-base http://localhost:8000 \
  --cluster-id test-cluster \
  --cluster-token test-token

# Run with debug logging
PYTHONUNBUFFERED=1 python main.py ...
```

## License

Same as Radix project.
