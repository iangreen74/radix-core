# GPU Optional Generation Demo

This example demonstrates text generation with optional GPU support, graceful fallback to CPU, and safety-first design principles.

## Overview

This demo shows how to:
- Conditionally use GPU when available (PyTorch)
- Gracefully fall back to CPU-only simulation
- Implement safety guards for resource usage
- Compare GPU vs CPU performance characteristics

## Safety Notice

🚨 **This demo runs with comprehensive safety guards**:
- GPU usage is **OPTIONAL** and **LOCAL ONLY**
- No cloud GPU instances or external APIs
- Cost caps remain at $0.00
- All operations run in DRY_RUN mode
- Automatic fallback to CPU if GPU unavailable

## Prerequisites

### Required
- Python 3.9+
- NumPy for numerical computations

### Optional (for GPU support)
- PyTorch (`pip install torch`)
- transformers (`pip install transformers`)

**Note**: If PyTorch is not installed, the demo automatically falls back to CPU simulation mode.

## Quick Start

```bash
# From the radix root directory
cd examples/gpu_optional_generation

# Run with automatic GPU detection
python run_demo.py

# Force CPU-only mode
python run_demo.py --force-cpu

# Run with specific batch size
python run_demo.py --batch-size 8

# Compare GPU vs CPU performance (if GPU available)
python run_demo.py --compare-modes
```

## What This Demo Does

### 1. GPU Detection and Fallback
- Automatically detects PyTorch and GPU availability
- Falls back gracefully to CPU simulation if GPU unavailable
- Reports which mode is being used

### 2. Text Generation Simulation
- Simulates text generation workloads
- Uses actual PyTorch models if available
- Falls back to deterministic simulation otherwise

### 3. Performance Comparison
- Measures generation speed and quality
- Compares GPU vs CPU performance when both available
- Analyzes memory usage patterns

### 4. Safety Validation
- Ensures no cloud resources are used
- Validates cost caps remain at $0.00
- Confirms local-only execution

## Example Output

### With GPU Available
```
🔒 Radix GPU Optional Generation Demo
====================================

Safety Check: ✅ DRY_RUN=true, COST_CAP=$0.00

Hardware Detection:
- PyTorch: ✅ Available (version 2.1.0)
- CUDA: ✅ Available (GPU: NVIDIA RTX 3080)
- GPU Memory: 10240 MB available
- Mode: GPU-accelerated generation

Initializing text generator... ✅ GPU model loaded

Running text generation experiment:
- Batch size: 4
- Sequence length: 128 tokens
- Total prompts: 100
- Device: cuda:0

Processing batches: 100%|████████████| 25/25 [00:08<00:00,  3.12batch/s]

Results:
========
Throughput: 12.5 sequences/second
Average generation time: 0.32 seconds/sequence
GPU memory usage: 2.8 GB peak
Power efficiency: 45.2 sequences/watt-hour (estimated)

Generated sample:
"The future of artificial intelligence will likely involve..."

✅ All safety checks passed
✅ No cloud resources used
✅ Cost remained at $0.00
```

### With CPU Fallback
```
🔒 Radix GPU Optional Generation Demo
====================================

Safety Check: ✅ DRY_RUN=true, COST_CAP=$0.00

Hardware Detection:
- PyTorch: ❌ Not available
- CUDA: ❌ Not available
- Mode: CPU simulation (safe fallback)

⚠️  GPU libraries not found, using CPU simulation mode
ℹ️  Install PyTorch for actual GPU acceleration: pip install torch

Initializing text generator... ✅ CPU simulator ready

Running text generation experiment:
- Batch size: 4
- Sequence length: 128 tokens
- Total prompts: 100
- Device: cpu (simulated)

Processing batches: 100%|████████████| 25/25 [00:12<00:00,  2.08batch/s]

Results:
========
Throughput: 8.3 sequences/second (simulated)
Average generation time: 0.48 seconds/sequence
CPU usage: 65% average
Memory usage: 180 MB

Generated sample (simulated):
"This is a simulated text generation output for demonstration..."

✅ All safety checks passed
✅ No cloud resources used
✅ Cost remained at $0.00
```

## Configuration Options

### Hardware Configuration
```python
# In run_demo.py
hardware_config = {
    'prefer_gpu': True,         # Use GPU if available
    'force_cpu': False,         # Force CPU-only mode
    'gpu_memory_limit': 8192,   # MB, prevent OOM
    'cpu_threads': 4,           # CPU threads for fallback
}
```

### Generation Configuration
```python
generation_config = {
    'batch_size': 4,            # Sequences per batch
    'max_length': 128,          # Maximum tokens per sequence
    'temperature': 0.7,         # Generation randomness
    'top_p': 0.9,              # Nucleus sampling parameter
}
```

### Safety Configuration
```python
# Safety settings (DO NOT MODIFY)
safety_config = {
    'dry_run': True,            # Always enabled
    'cost_cap_usd': 0.00,      # Always $0.00
    'no_deploy_mode': True,     # Always enabled
    'local_only': True,         # Always enabled
    'gpu_cloud_blocked': True,  # Block cloud GPU access
}
```

## Performance Characteristics

### Expected Performance (with GPU)
- **Throughput**: 10-50 sequences/second (depending on GPU)
- **Latency**: 0.1-0.5 seconds per sequence
- **Memory**: 1-8 GB GPU memory usage
- **Efficiency**: Significantly faster than CPU

### Expected Performance (CPU fallback)
- **Throughput**: 5-15 sequences/second (simulated)
- **Latency**: 0.3-1.0 seconds per sequence
- **Memory**: 100-500 MB system memory
- **CPU Usage**: 50-90% during generation

### GPU vs CPU Comparison
When both modes are available:
- GPU typically 3-10x faster than CPU
- GPU uses dedicated memory, reducing system RAM pressure
- CPU mode more predictable and doesn't require special hardware

## Code Structure

### Main Components

#### `run_demo.py`
Main demo script with hardware detection and experiment runner.

#### `text_generator.py`
Implements GPU and CPU text generation with automatic fallback.

#### `hardware_detector.py`
Detects available hardware and selects optimal configuration.

#### `performance_monitor.py`
Monitors GPU/CPU usage, memory, and generation performance.

### Key Classes

```python
class TextGenerator:
    """Base class for text generation."""

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate text for a batch of prompts."""

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the generation device."""

class GPUTextGenerator(TextGenerator):
    """GPU-accelerated text generation using PyTorch."""

    def __init__(self, model_name: str = "gpt2"):
        # Load actual PyTorch model

    def generate_batch(self, prompts: List[str]) -> List[str]:
        # Use GPU for actual generation

class CPUTextGenerator(TextGenerator):
    """CPU fallback with simulated text generation."""

    def generate_batch(self, prompts: List[str]) -> List[str]:
        # Simulate generation with deterministic output

class HardwareDetector:
    """Detects available hardware and capabilities."""

    def detect_pytorch(self) -> bool:
        """Check if PyTorch is available."""

    def detect_cuda(self) -> bool:
        """Check if CUDA GPU is available."""

    def get_optimal_config(self) -> HardwareConfig:
        """Get optimal configuration for available hardware."""
```

## Hardware Detection Logic

```python
def select_generation_mode():
    """Select the best available generation mode."""

    if force_cpu_mode:
        return CPUTextGenerator()

    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory > MIN_GPU_MEMORY:
                return GPUTextGenerator()
    except ImportError:
        pass

    # Fallback to CPU simulation
    print("⚠️  Falling back to CPU simulation mode")
    return CPUTextGenerator()
```

## Safety Features

### GPU Safety Guards
```python
@DryRunGuard.protect
def gpu_generation_with_guards(prompts: List[str]) -> List[str]:
    """GPU generation with comprehensive safety guards."""

    # Check memory limits
    if torch.cuda.memory_allocated() > MAX_GPU_MEMORY:
        raise ResourceLimitError("GPU memory limit exceeded")

    # Ensure local execution only
    if not is_local_device():
        raise SafetyViolationError("Non-local GPU detected")

    # Generate with monitoring
    return generate_with_monitoring(prompts)
```

### Cost Monitoring
```python
def monitor_generation_costs():
    """Monitor and cap generation costs."""

    # Estimate compute cost (always $0.00 in dry-run)
    estimated_cost = cost_simulator.estimate_gpu_cost(
        gpu_hours=generation_time / 3600,
        gpu_type="local"  # Always local
    )

    assert estimated_cost == 0.0, "Cost must be $0.00 in dry-run mode"
```

## Extending the Demo

### Add Custom Models
```python
class CustomTextGenerator(GPUTextGenerator):
    """Custom text generation implementation."""

    def __init__(self, custom_model_path: str):
        # Load your custom model
        self.model = load_custom_model(custom_model_path)

    def generate_batch(self, prompts: List[str]) -> List[str]:
        # Custom generation logic
        return self.model.generate(prompts)
```

### Add New Performance Metrics
```python
class ExtendedPerformanceMonitor(PerformanceMonitor):
    """Extended monitoring with custom metrics."""

    def monitor_generation_quality(self, generated_texts: List[str]):
        """Monitor text generation quality metrics."""
        # Your quality assessment logic
        pass

    def monitor_power_usage(self):
        """Monitor GPU power consumption."""
        # Power monitoring logic
        pass
```

## Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Check PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Out of Memory Errors
- Reduce batch size: `--batch-size 2`
- Reduce sequence length in configuration
- Monitor GPU memory usage

#### Slow Performance
- Verify GPU is being used (check output)
- Ensure CUDA drivers are properly installed
- Try different batch sizes for optimization

### Debug Mode
```bash
# Run with debug information
DEBUG=true python run_demo.py

# Run with detailed GPU monitoring
GPU_DEBUG=true python run_demo.py

# Force CPU mode for comparison
python run_demo.py --force-cpu
```

## Research Applications

This demo enables research into:

### Hardware Optimization
- GPU vs CPU performance characteristics
- Memory usage patterns
- Power efficiency analysis

### Generation Quality
- Batch size impact on generation quality
- Temperature and sampling parameter effects
- Model size vs performance trade-offs

### Safety Validation
- Resource limit enforcement
- Graceful degradation testing
- Cost monitoring accuracy

## Next Steps

After running this demo:

1. **Experiment with parameters**: Try different batch sizes and generation settings
2. **Compare hardware modes**: Run with and without GPU to see differences
3. **Explore the CPU demo**: Try `../cpu_batch_embeddings/` for comparison
4. **Implement custom models**: Add your own text generation models

## Safety Reminder

🔒 **This demo prioritizes safety**:
- All GPU usage is local-only (no cloud instances)
- Automatic fallback prevents failures
- Cost caps prevent accidental spending
- Comprehensive monitoring ensures safe operation

For questions or issues, please see the main [README](../../README.md) or create an issue on GitHub.
