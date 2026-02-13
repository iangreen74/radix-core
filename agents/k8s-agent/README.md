# Radix Kubernetes Agent v1

The Radix Kubernetes Agent runs inside your Kubernetes cluster and:
- Polls the Radix control plane for pending jobs
- Executes jobs as shell commands (Phase 1)
- Reports job completion and results
- Pushes cluster metrics (GPU utilization, job counts)

## Quick Start

### 1. Build and Push Docker Image

```bash
cd agents/k8s-agent
docker build -t YOUR_REGISTRY/radix-k8s-agent:latest .
docker push YOUR_REGISTRY/radix-k8s-agent:latest
```

### 2. Update Kubernetes Manifest

Edit `k8s/agent-deployment.yaml`:
- Replace `YOUR_REGISTRY` with your container registry
- Update `CLUSTER_ID` to match your cluster name
- Update `agent-api-key` to match your control plane's `AGENT_API_KEY`

### 3. Deploy to Kubernetes

```bash
kubectl apply -f k8s/agent-deployment.yaml
```

### 4. Verify Deployment

```bash
# Check pod status
kubectl get pods -n radix-system

# View agent logs
kubectl logs -n radix-system -l app=radix-k8s-agent -f
```

## Configuration

Environment variables:
- `API_BASE_URL`: Radix control plane URL (default: https://api.vaultscaler.com)
- `CLUSTER_ID`: Unique cluster identifier (required)
- `AGENT_API_KEY`: Shared secret for authentication (required)
- `POLL_INTERVAL_SECONDS`: Job polling interval (default: 10)
- `METRICS_INTERVAL_SECONDS`: Metrics push interval (default: 30)

## Testing Locally

```bash
export API_BASE_URL="https://api.vaultscaler.com"
export CLUSTER_ID="vault-hub-demo"
export AGENT_API_KEY="radix-agent-key-change-in-production"

python agent.py
```

## Phase 1 Limitations

- Jobs execute as shell commands (not full Kubernetes Jobs yet)
- GPU metrics require nvidia-smi (optional)
- Job/queue counts not yet tracked from Kubernetes API

## Next Steps (Phase 2)

- Launch jobs as Kubernetes Jobs/Pods
- Use Kubernetes API for job/queue metrics
- Add resource requests/limits per job
- Support GPU scheduling and affinity
