# GPU Scheduler Mutating Admission Webhook

Kubernetes mutating admission webhook that integrates with the information-theoretic GPU scheduler to automatically assign optimal priority classes and node affinities to GPU workloads.

## Features

- **Automatic Pod Mutation**: Adds scheduling hints based on scheduler recommendations
- **Priority Class Assignment**: Maps scores to priority classes for Kubernetes scheduler
- **Node Affinity**: Sets node selectors and tolerations for specific GPU types
- **Anti-Affinity**: Prevents problematic job co-locations
- **Graceful Degradation**: Fails open if scheduler agent is unavailable
- **TLS Integration**: Secure webhook communication via cert-manager

## Quick Start

### Prerequisites

- Kubernetes cluster with admission controllers enabled
- cert-manager installed for TLS certificate management
- Scheduler agent service running

### Installation

```bash
# Install via Helm
helm install scheduler-webhook charts/scheduler-webhook \
  --namespace scheduler \
  --set webhook.schedulerUrl=http://scheduler-agent:8080
```

### Enable for Namespace

```bash
# Label namespace to enable webhook
kubectl label namespace gpu-workloads scheduler.radix.ai/enabled=true
```

## How It Works

### Admission Flow

1. **Pod Creation**: User submits Pod/Job with GPU requests
2. **Webhook Intercept**: Kubernetes calls webhook for mutation
3. **Feature Extraction**: Extract job type and features from annotations
4. **Scheduler Call**: Call scheduler agent `/v1/score` endpoint
5. **Mutation**: Apply JSON patches based on recommendations
6. **Scheduling**: Pod gets scheduled with enhanced priority and affinity

### Required Annotations

For pods to be processed by the webhook:

```yaml
apiVersion: v1
kind: Pod
metadata:
  annotations:
    app.kubernetes.io/job-type: "train-bert"  # Required
    gpu.mem.gi: "40"                          # Optional
    ml.batch_size: "16"                       # Optional
    scheduler.radix.ai/tenant: "team-a"       # Optional
spec:
  containers:
  - name: trainer
    resources:
      requests:
        nvidia.com/gpu: "1"
```

### Generated Mutations

The webhook adds these fields to Pods:

```yaml
spec:
  priorityClassName: gpu-high               # Based on score
  nodeSelector:
    gpu.nvidia.com/class: A100-80GB         # GPU type selection
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
  affinity:
    podAntiAffinity:                        # Avoid interference
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchExpressions:
            - key: app.kubernetes.io/job-type
              operator: In
              values: ["train-llama"]
          topologyKey: kubernetes.io/hostname
metadata:
  annotations:
    scheduler.radix.ai/priority-score: "85"
    scheduler.radix.ai/chosen-gpu: "A100-80GB"
    scheduler.radix.ai/decision-timestamp: "1699123456"
```

## Configuration

### Priority Class Mapping

Configure score-to-priority mapping in values.yaml:

```yaml
webhook:
  priorityBuckets:
    - minScore: 90
      priorityClassName: "gpu-ultra"
    - minScore: 70
      priorityClassName: "gpu-high"
    - minScore: 50
      priorityClassName: "gpu-medium"
    - minScore: 0
      priorityClassName: "gpu-default"
```

### Webhook Rules

Configure which resources to mutate:

```yaml
webhook:
  rules:
    - operations: ["CREATE", "UPDATE"]
      apiGroups: [""]
      apiVersions: ["v1"]
      resources: ["pods"]
    - operations: ["CREATE", "UPDATE"]
      apiGroups: ["batch"]
      apiVersions: ["v1"]
      resources: ["jobs"]
```

### Namespace Selection

Enable webhook for specific namespaces:

```yaml
webhook:
  namespaceSelector:
    matchLabels:
      scheduler.radix.ai/enabled: "true"
```

## Testing

### Unit Tests

```bash
# Run webhook tests
cd agents/scheduler-webhook
pytest tests/ -v
```

### Integration Testing

```bash
# Create test namespace
kubectl create namespace test-webhook
kubectl label namespace test-webhook scheduler.radix.ai/enabled=true

# Apply test job
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: test-mutation
  namespace: test-webhook
  annotations:
    app.kubernetes.io/job-type: "train-bert"
    gpu.mem.gi: "40"
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: trainer
        image: busybox
        command: ["sleep", "300"]
        resources:
          requests:
            nvidia.com/gpu: "1"
EOF

# Verify mutation
kubectl get pod -n test-webhook -l job-name=test-mutation -o yaml
```

## API Integration

### Scheduler Agent Calls

The webhook makes HTTP calls to the scheduler agent:

```bash
# Example scoring request
POST http://scheduler-agent:8080/v1/score
{
  "job_type": "train-bert",
  "features": {
    "gpu_mem_gb": 40,
    "batch_size": 16,
    "tenant": "team-a"
  },
  "candidate_gpu_types": ["A100-80GB", "A100-40GB", "L4-24GB"],
  "colocated_job_types": []
}
```

### Error Handling

- **Scheduler Unavailable**: Webhook allows pod creation without mutation
- **Invalid Response**: Logs error and allows pod creation
- **Timeout**: Configurable timeout with graceful fallback

## Security

### TLS Configuration

The webhook requires TLS certificates for secure communication:

```yaml
# Using cert-manager (recommended)
tls:
  certManager:
    enabled: true
    issuer:
      kind: ClusterIssuer
      name: selfsigned-cluster-issuer

# Using existing secret
tls:
  existingSecret: webhook-tls-secret
```

### RBAC

Minimal permissions required:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: scheduler-webhook
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
```

### Network Policies

Restrict webhook traffic:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: scheduler-webhook
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: scheduler-webhook
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: TCP
      port: 8443
  egress:
  - to:
    - podSelector:
        matchLabels:
          app.kubernetes.io/name: scheduler-agent
    ports:
    - protocol: TCP
      port: 8080
```

## Monitoring

### Health Checks

```bash
# Health endpoint
curl -k https://scheduler-webhook:8443/healthz

# Readiness endpoint
curl -k https://scheduler-webhook:8443/readyz
```

### Logs

```bash
# View webhook logs
kubectl logs -f deployment/scheduler-webhook -n scheduler

# Common log patterns
grep "Mutated pod" logs
grep "Scheduler unavailable" logs
grep "ERROR" logs
```

### Metrics

While the webhook doesn't expose Prometheus metrics directly, monitor via:

1. **Kubernetes Events**: Failed admission webhook calls
2. **Scheduler Agent Metrics**: Scoring requests from webhook
3. **Pod Mutations**: Count of successfully mutated pods

## Troubleshooting

### Common Issues

1. **Pods Not Being Mutated**
   - Check namespace has `scheduler.radix.ai/enabled=true` label
   - Verify pod has `app.kubernetes.io/job-type` annotation
   - Ensure webhook is running and healthy

2. **Certificate Issues**
   - Check cert-manager is installed and working
   - Verify certificate is valid: `kubectl describe certificate -n scheduler`
   - Check webhook configuration has correct CA bundle

3. **Scheduler Agent Unreachable**
   - Verify agent service is running: `kubectl get svc scheduler-agent -n scheduler`
   - Check network connectivity between webhook and agent
   - Review webhook timeout configuration

### Debug Commands

```bash
# Check webhook configuration
kubectl get mutatingwebhookconfiguration scheduler-webhook -o yaml

# Verify TLS certificate
kubectl get secret scheduler-webhook-tls -n scheduler -o yaml

# Test webhook manually
kubectl get pod -n test-webhook -o yaml

# Check admission controller logs
kubectl logs -n kube-system kube-apiserver-* | grep admission
```

## Development

### Local Testing

```bash
# Run webhook locally (requires valid TLS certs)
cd agents/scheduler-webhook
python -m uvicorn app.webhook:app --host 0.0.0.0 --port 8443

# Mock scheduler for testing
python -c "
from fastapi import FastAPI
app = FastAPI()
@app.post('/v1/score')
async def mock_score():
    return {'priority_score': 75, 'gpu_selector': {'nodeSelector': {'gpu.nvidia.com/class': 'A100-80GB'}}, 'avoid_co_locate_with': [], 'terms': {'chosen_gpu': 'A100-80GB'}}
" &
```

### Adding Features

1. **New Annotations**: Add parsing in `extract_job_features`
2. **New Mutations**: Extend `create_json_patch` method
3. **New Resources**: Add to webhook rules in Chart values

## Production Deployment

### High Availability

```yaml
# values.yaml
replicaCount: 2

affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchExpressions:
          - key: app.kubernetes.io/name
            operator: In
            values: [scheduler-webhook]
        topologyKey: kubernetes.io/hostname
```

### Resource Limits

```yaml
resources:
  requests:
    cpu: 100m
    memory: 128Mi
  limits:
    cpu: 200m
    memory: 256Mi
```

### Failure Policy

```yaml
webhook:
  failurePolicy: Ignore  # Fail open for safety
  sideEffects: None
  timeoutSeconds: 10
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for webhook functionality
4. Test with actual Kubernetes cluster
5. Submit a pull request

## License

See [LICENSE](../../LICENSE) file for details.
