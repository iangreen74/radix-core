# Radix GPU Benchmark Suite — g5.2xlarge — ResNet50 (Earned Claim)

Generated: 2025-12-26  
Scope: g5.2xlarge (A10G), ResNet50, us-west-2

## Executive summary

- Direct-run "raw box" throughput ceiling measured at **~346.45 images/sec** (batch=128, epochs=1).
- Radix Evidence Pack baseline ("policy-off", batch=128) achieved **343.27 images/sec** (3 trials).
- Batch sweep across {64, 128, 256} identified **batch=256** as best, at **346.61 images/sec** (2 trials).
- Radix Evidence Pack "policy-on" (controller-selected batch=256) achieved **346.69 images/sec** (3 trials).
- Net improvement from policy-on vs policy-off: **~1.0% throughput increase**.
- Radix policy-on performance is effectively at the direct-run ceiling for this workload configuration.

## Configuration

- Instance Type: g5.2xlarge
- Region: us-west-2
- Workload: ResNet50
- Epochs: 1
- Baseline batch (policy-off): 128
- Sweep batches: 64, 128, 256

## Baselines

### Direct Run (Ceiling)

Direct Run executes the workload directly on the GPU instance without Radix orchestration.

- Observed throughput (examples): 346.35, 346.41, 346.58 images/sec  
- Mean (approx): **346.45 images/sec**

### Evidence Pack baseline (Policy-off, batch=128)

Evidence Pack runs the workload through the Radix benchmark harness and produces auditable artifacts.

Throughput (images/sec):
- Mean: **343.27**
- Std Dev: 0.06
- Min/Max: 343.20 / 343.34
- Samples: 3

Duration (seconds):
- Mean: 175.13
- Std Dev: 2.36
- Min/Max: 171.81 / 177.06
- Samples: 3

## Sweep + controller decision

### Sweep results (2 trials per batch)

| Rank | Batch | Mean Throughput (img/s) | Mean Duration (s) | Trials |
|---:|---:|---:|---:|---:|
| 1 | 256 | 346.61 | 248.39 | 2 |
| 2 | 128 | 343.29 | 186.39 | 2 |
| 3 | 64  | 337.32 | 140.77 | 2 |

Controller decision:
- Best batch: **256**
- Rationale: Highest mean throughput across sweep trials

## Policy-on validation (Evidence Pack @ batch=256)

Throughput (images/sec):
- Mean: **346.69**
- Std Dev: 0.01
- Min/Max: 346.67 / 346.70
- Samples: 3

Duration (seconds):
- Mean: 249.59
- Std Dev: 6.70
- Min/Max: 244.55 / 259.06
- Samples: 3

Per-trial:
- trial_1: 346.69 img/s, 244.55 s
- trial_2: 346.70 img/s, 259.06 s
- trial_3: 346.67 img/s, 245.17 s

## Comparisons

### Policy-on vs Policy-off (Radix-run)

- Policy-off throughput: 343.27 img/s  
- Policy-on throughput: 346.69 img/s  
- Absolute delta: +3.42 img/s  
- Relative improvement: **~0.996% (~1.0%)**

### Policy-on vs Direct Run (Ceiling)

- Direct-run ceiling: ~346.45 img/s  
- Policy-on: 346.69 img/s  
- Delta: +0.24 img/s (effectively within noise)

## Earned claim

For this workload configuration on g5.2xlarge, Radix orchestration operates near the raw hardware ceiling, and a controller-selected configuration (batch=256) improves throughput by ~1.0% relative to a fixed baseline (batch=128). This is an earned performance claim backed by reproducible artifacts.

## Caveats

- Throughput is the primary comparator. Duration is not directly comparable between Direct Run and Evidence Pack due to differences in end-to-end measurement scope.
- Epochs=1 is a short benchmark. Future suites may use longer runs (e.g., epochs=50) to reduce measurement noise and improve confidence.

## Next steps

1) Automate this suite with a single orchestrated "benchmark suite runner" workflow that:
   - runs Direct Run baseline (N trials)
   - runs Evidence Pack baseline (N trials)
   - runs sweep
   - applies controller decision
   - runs Evidence Pack policy-on (N trials)
   - generates suite_summary.md and suite_summary.json

2) Expand the action space beyond batch size:
   - placement policy (node/GPU)
   - fairness/preemption policy
   - cost-aware scheduling

3) Begin RIM-1 dataset standardization:
   - unify observation schema across run types
   - persist observations to S3 and maintain a longitudinal dataset

4) Introduce FEP/active-inference controller evolution once datasets are stable.
