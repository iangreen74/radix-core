# ResNet-50 Training Benchmark

This benchmark measures GPU training throughput using a ResNet-50 model on synthetic ImageNet-like data. It's designed for internal performance measurement and case studies (e.g., demonstrating throughput improvements).

## Purpose

- **Measure GPU training performance** in images/second
- **Standardized workload** for comparing different GPU types, batch sizes, and configurations
- **Radix-compatible metrics output** for integration with experiment tracking

## Quick Start

### Build the Docker Image

```bash
docker build -t radix-resnet50-benchmark:latest benchmarks/resnet50/
```

### Run Locally (Requires GPU)

```bash
docker run --gpus all --rm \
  radix-resnet50-benchmark:latest \
  --lr 0.1 --batch-size 128 --epochs 1 --steps-per-epoch 200
```

### Run with Custom Parameters

```bash
docker run --gpus all --rm \
  radix-resnet50-benchmark:latest \
  --lr 0.05 \
  --batch-size 256 \
  --epochs 2 \
  --steps-per-epoch 100
```

### Run on CPU (Development/Testing Only)

```bash
docker run --rm \
  radix-resnet50-benchmark:latest \
  --allow-cpu \
  --batch-size 32 \
  --steps-per-epoch 10
```

**Note:** CPU runs are very slow and not suitable for benchmarking.

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--lr` | float | 0.1 | Learning rate for SGD optimizer |
| `--batch-size` | int | 128 | Number of images per batch |
| `--epochs` | int | 1 | Number of training epochs |
| `--steps-per-epoch` | int | 200 | Training steps per epoch |
| `--allow-cpu` | flag | false | Allow running on CPU if GPU unavailable |

## Output Format

The benchmark outputs progress logs during training and a final JSON line with metrics:

```json
{
  "radix_metrics": {
    "images_per_second": 1234.56,
    "total_images": 25600,
    "elapsed_seconds": 20.75,
    "lr": 0.1,
    "batch_size": 128,
    "epochs": 1,
    "steps_per_epoch": 200
  }
}
```

### Metrics Explanation

- **`images_per_second`**: Training throughput (higher is better)
- **`total_images`**: Total number of images processed (batch_size × steps_per_epoch × epochs)
- **`elapsed_seconds`**: Wall-clock time for the training loop
- **`lr`, `batch_size`, `epochs`, `steps_per_epoch`**: Configuration parameters

## Integration with Radix

The `radix_metrics` JSON object is designed to be parsed by Radix pipelines and stored in `ExperimentRunsTable`. When this benchmark is integrated with the Radix scheduler:

1. Jobs will be submitted to GPU nodes via Kubernetes
2. The container will run with specified hyperparameters
3. The final JSON line will be captured and parsed
4. Metrics will populate the experiment run's `metrics` field

## Expected Performance

Typical throughput on common GPUs (batch_size=128, steps_per_epoch=200):

| GPU | Images/Second (approx) | Runtime |
|-----|------------------------|---------|
| NVIDIA V100 | ~1200-1500 | ~17-21 sec |
| NVIDIA A100 | ~2000-2500 | ~10-13 sec |
| NVIDIA T4 | ~600-800 | ~32-43 sec |

**Note:** Actual performance varies based on GPU memory, CUDA version, and system configuration.

## Requirements

- **Docker** with GPU support (`nvidia-docker` or Docker 19.03+ with `--gpus` flag)
- **NVIDIA GPU** with CUDA 11.8+ support
- **GPU drivers** compatible with CUDA 11.8

## Troubleshooting

### "CUDA not available" Error

If you see:
```
[ERROR] CUDA not available and --allow-cpu not specified
```

**Solutions:**
1. Ensure you're using `--gpus all` flag with `docker run`
2. Verify GPU drivers are installed: `nvidia-smi`
3. Check Docker GPU support: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`
4. For testing only, add `--allow-cpu` flag (not for benchmarking)

### Out of Memory (OOM) Errors

If training crashes with CUDA OOM:
- Reduce `--batch-size` (try 64, 32, or 16)
- Check GPU memory: `nvidia-smi`

### Slow Performance

If throughput is unexpectedly low:
- Verify GPU is being used (check logs for GPU name)
- Ensure no other processes are using the GPU
- Check GPU utilization: `nvidia-smi dmon`

## Development

### Local Testing Without Docker

```bash
cd benchmarks/resnet50
pip install torch torchvision
python train_resnet50.py --batch-size 64 --steps-per-epoch 50
```

### Modifying the Benchmark

The training script (`train_resnet50.py`) is self-contained and can be modified for:
- Different models (e.g., ResNet-101, EfficientNet)
- Different optimizers (Adam, AdamW)
- Mixed precision training (torch.cuda.amp)
- Different synthetic data distributions

## Future Enhancements

- [ ] Mixed precision (FP16) support for faster training
- [ ] Multi-GPU support with DistributedDataParallel
- [ ] Additional models (ResNet-101, EfficientNet, ViT)
- [ ] Real ImageNet data loading option
- [ ] Integration with Radix scheduler and experiment tracking
