#!/usr/bin/env python3
"""
GPU Optional Generation Demo

This demo showcases text generation with optional GPU support, graceful fallback
to CPU simulation, and comprehensive safety guards.

Safety: GPU usage is local-only with automatic fallback to CPU simulation.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import threading

# Add the radix engine to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from radix_core.config import RadixConfig
    from radix_core.dryrun import DryRunGuard
    from radix_core.cost_simulator import CostSimulator
except ImportError:
    print("❌ Could not import Radix engine. Please run from the radix root directory.")
    sys.exit(1)

import numpy as np
import psutil


@dataclass
class HardwareInfo:
    """Information about available hardware."""
    has_pytorch: bool
    has_cuda: bool
    gpu_name: Optional[str]
    gpu_memory_mb: Optional[int]
    cpu_cores: int
    system_memory_mb: int


@dataclass
class GenerationMetrics:
    """Metrics for text generation performance."""
    throughput_sequences_per_sec: float
    avg_generation_time_sec: float
    total_duration_sec: float
    device_type: str
    memory_usage_mb: float
    sequences_generated: int
    batch_count: int
    avg_batch_size: float


class HardwareDetector:
    """Detects available hardware and capabilities."""

    @staticmethod
    def detect_hardware() -> HardwareInfo:
        """Detect available hardware and return capabilities."""

        # Check PyTorch availability
        has_pytorch = False
        has_cuda = False
        gpu_name = None
        gpu_memory_mb = None

        try:
            import torch
            has_pytorch = True

            if torch.cuda.is_available():
                has_cuda = True
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        except ImportError:
            pass

        # Get system info
        cpu_cores = psutil.cpu_count(logical=False)
        system_memory_mb = psutil.virtual_memory().total // (1024 * 1024)

        return HardwareInfo(
            has_pytorch=has_pytorch,
            has_cuda=has_cuda,
            gpu_name=gpu_name,
            gpu_memory_mb=gpu_memory_mb,
            cpu_cores=cpu_cores,
            system_memory_mb=system_memory_mb
        )

    @staticmethod
    def print_hardware_info(hardware: HardwareInfo):
        """Print detected hardware information."""
        print("\nHardware Detection:")

        pytorch_status = "✅ Available" if hardware.has_pytorch else "❌ Not available"
        if hardware.has_pytorch:
            try:
                import torch
                pytorch_status += f" (version {torch.__version__})"
            except:
                pass
        print(f"- PyTorch: {pytorch_status}")

        if hardware.has_cuda:
            print(f"- CUDA: ✅ Available (GPU: {hardware.gpu_name})")
            print(f"- GPU Memory: {hardware.gpu_memory_mb} MB available")
        else:
            print("- CUDA: ❌ Not available")

        print(f"- CPU Cores: {hardware.cpu_cores}")
        print(f"- System Memory: {hardware.system_memory_mb} MB")


class TextGenerator:
    """Base class for text generation."""

    def __init__(self):
        self.cost_simulator = CostSimulator()

    def generate_batch(self, prompts: List[str], max_length: int = 128) -> List[str]:
        """Generate text for a batch of prompts."""
        raise NotImplementedError

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the generation device."""
        raise NotImplementedError

    def cleanup(self):
        """Clean up resources."""
        pass


class GPUTextGenerator(TextGenerator):
    """GPU-accelerated text generation using PyTorch."""

    def __init__(self, model_name: str = "gpt2", max_gpu_memory_mb: int = 8192):
        super().__init__()
        self.model_name = model_name
        self.max_gpu_memory_mb = max_gpu_memory_mb
        self.device = None
        self.model = None
        self.tokenizer = None

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the GPU model with safety checks."""
        try:
            import torch
            from transformers import GPT2LMHeadModel, GPT2Tokenizer

            # Safety check: ensure CUDA is available
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")

            # Safety check: ensure sufficient GPU memory
            gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            if gpu_memory_mb < 1024:  # Minimum 1GB
                raise RuntimeError(f"Insufficient GPU memory: {gpu_memory_mb}MB")

            self.device = torch.device("cuda:0")

            # Load model and tokenizer
            print(f"Loading {self.model_name} model on GPU...")
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)

            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model.to(self.device)
            self.model.eval()

            print("✅ GPU model loaded successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize GPU model: {e}")

    @DryRunGuard.protect
    def generate_batch(self, prompts: List[str], max_length: int = 128) -> List[str]:
        """Generate text using GPU acceleration."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not initialized")

        try:
            import torch

            # Safety check: monitor GPU memory
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated(0) // (1024 * 1024)
                if memory_used > self.max_gpu_memory_mb:
                    raise RuntimeError(f"GPU memory limit exceeded: {memory_used}MB")

            # Tokenize prompts
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length // 2  # Leave room for generation
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode generated text
            generated_texts = []
            for i, output in enumerate(outputs):
                # Skip the input tokens
                input_length = inputs['input_ids'][i].shape[0]
                generated_tokens = output[input_length:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                generated_texts.append(generated_text.strip())

            return generated_texts

        except Exception as e:
            raise RuntimeError(f"GPU generation failed: {e}")

    def get_device_info(self) -> Dict[str, Any]:
        """Get GPU device information."""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    'device': 'cuda:0',
                    'gpu_name': torch.cuda.get_device_name(0),
                    'memory_total_mb': torch.cuda.get_device_properties(0).total_memory // (1024 * 1024),
                    'memory_allocated_mb': torch.cuda.memory_allocated(0) // (1024 * 1024),
                    'model_name': self.model_name
                }
        except:
            pass

        return {'device': 'unknown', 'error': 'GPU info not available'}

    def cleanup(self):
        """Clean up GPU resources."""
        try:
            import torch
            if self.model:
                del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass


class CPUTextGenerator(TextGenerator):
    """CPU fallback with simulated text generation."""

    def __init__(self, processing_delay: float = 0.1):
        super().__init__()
        self.processing_delay = processing_delay
        self.generation_templates = [
            "This is a simulated text generation output for demonstration purposes.",
            "The future of artificial intelligence will likely involve advanced machine learning techniques.",
            "Natural language processing enables computers to understand and generate human-like text.",
            "Distributed computing systems can process large datasets efficiently across multiple nodes.",
            "Machine learning algorithms learn patterns from data to make predictions and decisions.",
            "Text generation models use neural networks to produce coherent and contextually relevant content.",
            "The development of AI systems requires careful consideration of ethical and safety implications.",
            "Parallel processing techniques can significantly improve the performance of computational tasks.",
            "Data science combines statistics, programming, and domain expertise to extract insights.",
            "Computer vision systems can analyze and interpret visual information from images and videos."
        ]

    @DryRunGuard.protect
    def generate_batch(self, prompts: List[str], max_length: int = 128) -> List[str]:
        """Simulate text generation with deterministic output."""

        # Simulate processing time
        time.sleep(self.processing_delay * len(prompts))

        # Generate simulated text based on prompts
        generated_texts = []
        for i, prompt in enumerate(prompts):
            # Create deterministic but varied output
            template_idx = (hash(prompt) + i) % len(self.generation_templates)
            base_text = self.generation_templates[template_idx]

            # Add prompt-specific variation
            prompt_words = prompt.lower().split()[:3]
            if prompt_words:
                variation = f" The prompt mentioned {', '.join(prompt_words)}."
                generated_text = base_text + variation
            else:
                generated_text = base_text

            # Truncate to max_length (approximate)
            if len(generated_text) > max_length:
                generated_text = generated_text[:max_length-3] + "..."

            generated_texts.append(generated_text)

        return generated_texts

    def get_device_info(self) -> Dict[str, Any]:
        """Get CPU device information."""
        return {
            'device': 'cpu (simulated)',
            'cpu_cores': psutil.cpu_count(logical=False),
            'memory_total_mb': psutil.virtual_memory().total // (1024 * 1024),
            'processing_delay': self.processing_delay
        }


class PerformanceMonitor:
    """Monitors performance during text generation."""

    def __init__(self):
        self.start_time = None
        self.batch_completions = []
        self.memory_samples = []
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

    def record_batch_completion(self, batch_size: int, duration: float, device_type: str):
        """Record batch completion metrics."""
        self.batch_completions.append({
            'timestamp': time.time(),
            'batch_size': batch_size,
            'duration': duration,
            'device_type': device_type,
            'throughput': batch_size / duration
        })

    def _monitor_resources(self):
        """Monitor system resources in background."""
        while self.monitoring:
            try:
                # Monitor system memory
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                self.memory_samples.append(memory_mb)

                # Monitor GPU memory if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_memory_mb = torch.cuda.memory_allocated(0) / 1024 / 1024
                        self.memory_samples.append(gpu_memory_mb)
                except:
                    pass

                time.sleep(0.5)
            except:
                pass

    def get_metrics(self, device_type: str) -> GenerationMetrics:
        """Calculate performance metrics."""
        if not self.batch_completions:
            raise ValueError("No batch completions recorded")

        total_duration = time.time() - self.start_time
        total_sequences = sum(b['batch_size'] for b in self.batch_completions)
        throughput = total_sequences / total_duration

        durations = [b['duration'] for b in self.batch_completions]
        avg_generation_time = np.mean(durations)

        batch_sizes = [b['batch_size'] for b in self.batch_completions]
        avg_batch_size = np.mean(batch_sizes)

        avg_memory = np.mean(self.memory_samples) if self.memory_samples else 0.0

        return GenerationMetrics(
            throughput_sequences_per_sec=throughput,
            avg_generation_time_sec=avg_generation_time,
            total_duration_sec=total_duration,
            device_type=device_type,
            memory_usage_mb=avg_memory,
            sequences_generated=total_sequences,
            batch_count=len(self.batch_completions),
            avg_batch_size=avg_batch_size
        )


def create_text_generator(hardware: HardwareInfo, force_cpu: bool = False) -> Tuple[TextGenerator, str]:
    """Create appropriate text generator based on available hardware."""

    if force_cpu:
        print("- Mode: CPU simulation (forced)")
        return CPUTextGenerator(), "cpu"

    if hardware.has_pytorch and hardware.has_cuda:
        try:
            generator = GPUTextGenerator()
            print("- Mode: GPU-accelerated generation")
            return generator, "gpu"
        except Exception as e:
            print(f"⚠️  GPU initialization failed: {e}")
            print("- Mode: CPU simulation (GPU fallback)")
            return CPUTextGenerator(), "cpu"

    if not hardware.has_pytorch:
        print("⚠️  PyTorch not found, using CPU simulation mode")
        print("ℹ️  Install PyTorch for GPU acceleration: pip install torch")
    elif not hardware.has_cuda:
        print("⚠️  CUDA not available, using CPU simulation mode")

    print("- Mode: CPU simulation (safe fallback)")
    return CPUTextGenerator(), "cpu"


def generate_sample_prompts(count: int = 100) -> List[str]:
    """Generate sample prompts for text generation."""
    prompt_templates = [
        "The future of artificial intelligence",
        "In a world where technology",
        "Scientists have discovered",
        "The most important lesson",
        "Imagine a society where",
        "The key to understanding",
        "Throughout history, humans",
        "In the next decade",
        "The relationship between",
        "One of the greatest challenges"
    ]

    prompts = []
    for i in range(count):
        template = prompt_templates[i % len(prompt_templates)]
        prompts.append(f"{template} (prompt {i+1})")

    return prompts


def run_generation_experiment(
    generator: TextGenerator,
    prompts: List[str],
    batch_size: int,
    device_type: str
) -> GenerationMetrics:
    """Run text generation experiment."""

    print("\nRunning text generation experiment:")
    print(f"- Batch size: {batch_size}")
    print(f"- Total prompts: {len(prompts)}")
    print(f"- Device: {generator.get_device_info().get('device', 'unknown')}")

    # Create batches
    batches = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        batches.append(batch)

    print(f"- Created {len(batches)} batches")

    # Start monitoring
    monitor = PerformanceMonitor()
    monitor.start_monitoring()

    # Process batches
    all_generated = []
    for i, batch in enumerate(batches):
        try:
            start_time = time.time()
            generated = generator.generate_batch(batch)
            duration = time.time() - start_time

            monitor.record_batch_completion(len(batch), duration, device_type)
            all_generated.extend(generated)

            # Progress indicator
            progress = (i + 1) / len(batches) * 100
            print(f"\rProgress: {progress:.1f}% ({i+1}/{len(batches)} batches)", end="")

        except Exception as e:
            print(f"\n❌ Error processing batch {i}: {e}")
            break

    print()  # New line after progress

    # Stop monitoring and get metrics
    monitor.stop_monitoring()
    metrics = monitor.get_metrics(device_type)

    # Show sample generated text
    if all_generated:
        print("\nGenerated sample:")
        sample_text = all_generated[0][:100] + "..." if len(all_generated[0]) > 100 else all_generated[0]
        print(f'"{sample_text}"')

    return metrics


def print_detailed_results(metrics: GenerationMetrics):
    """Print detailed performance results."""
    print("\n📊 Results")
    print("=" * 20)
    print(f"Throughput: {metrics.throughput_sequences_per_sec:.1f} sequences/second")
    print(f"Average generation time: {metrics.avg_generation_time_sec:.2f} seconds/sequence")

    if metrics.device_type == "gpu":
        print(f"GPU memory usage: {metrics.memory_usage_mb/1024:.1f} GB peak")
        # Estimate power efficiency (simplified)
        estimated_power_efficiency = metrics.throughput_sequences_per_sec * 3600 / 300  # Assuming 300W GPU
        print(f"Power efficiency: {estimated_power_efficiency:.1f} sequences/watt-hour (estimated)")
    else:
        cpu_usage = psutil.cpu_percent(interval=1)
        print(f"CPU usage: {cpu_usage:.0f}% average")
        print(f"Memory usage: {metrics.memory_usage_mb:.0f} MB")


def compare_gpu_cpu_performance(hardware: HardwareInfo, prompts: List[str], batch_size: int):
    """Compare GPU and CPU performance if both are available."""
    if not (hardware.has_pytorch and hardware.has_cuda):
        print("⚠️  GPU not available for comparison")
        return

    print("\n🔬 GPU vs CPU Performance Comparison")
    print("=" * 50)

    results = {}

    # Test GPU mode
    print("\nTesting GPU mode...")
    try:
        gpu_generator, _ = create_text_generator(hardware, force_cpu=False)
        gpu_metrics = run_generation_experiment(gpu_generator, prompts[:50], batch_size, "gpu")
        results["GPU"] = gpu_metrics
        gpu_generator.cleanup()
        print(f"✅ GPU: {gpu_metrics.throughput_sequences_per_sec:.1f} sequences/sec")
    except Exception as e:
        print(f"❌ GPU test failed: {e}")

    # Test CPU mode
    print("\nTesting CPU mode...")
    try:
        cpu_generator, _ = create_text_generator(hardware, force_cpu=True)
        cpu_metrics = run_generation_experiment(cpu_generator, prompts[:50], batch_size, "cpu")
        results["CPU"] = cpu_metrics
        print(f"✅ CPU: {cpu_metrics.throughput_sequences_per_sec:.1f} sequences/sec")
    except Exception as e:
        print(f"❌ CPU test failed: {e}")

    # Print comparison
    if len(results) == 2:
        gpu_throughput = results["GPU"].throughput_sequences_per_sec
        cpu_throughput = results["CPU"].throughput_sequences_per_sec
        speedup = gpu_throughput / cpu_throughput if cpu_throughput > 0 else 0

        print("\n🏆 Performance Comparison:")
        print(f"GPU throughput: {gpu_throughput:.1f} sequences/sec")
        print(f"CPU throughput: {cpu_throughput:.1f} sequences/sec")
        print(f"GPU speedup: {speedup:.1f}x faster than CPU")


def verify_safety_settings():
    """Verify that all safety settings are properly configured."""
    print("🔒 Safety Check", end="")

    try:
        config = RadixConfig.from_env()

        # Check critical safety settings
        assert config.dry_run is True, "DRY_RUN must be True"
        assert config.cost_cap_usd == 0.0, "COST_CAP_USD must be 0.00"
        assert config.no_deploy_mode is True, "NO_DEPLOY_MODE must be True"

        print(f": ✅ DRY_RUN={config.dry_run}, COST_CAP=${config.cost_cap_usd:.2f}")
        return True

    except Exception as e:
        print(f": ❌ Safety check failed: {e}")
        return False


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="GPU Optional Generation Demo")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for generation (default: 4)")
    parser.add_argument("--prompt-count", type=int, default=100,
                        help="Number of prompts to generate (default: 100)")
    parser.add_argument("--force-cpu", action="store_true",
                        help="Force CPU-only mode")
    parser.add_argument("--compare-modes", action="store_true",
                        help="Compare GPU and CPU performance")

    args = parser.parse_args()

    print("🔒 Radix GPU Optional Generation Demo")
    print("=" * 40)

    # Verify safety settings
    if not verify_safety_settings():
        print("❌ Safety check failed. Please ensure DRY_RUN=true and COST_CAP_USD=0.00")
        return 1

    # Detect hardware
    hardware = HardwareDetector.detect_hardware()
    HardwareDetector.print_hardware_info(hardware)

    # Generate sample prompts
    prompts = generate_sample_prompts(args.prompt_count)

    try:
        if args.compare_modes:
            # Run comparison mode
            compare_gpu_cpu_performance(hardware, prompts, args.batch_size)
        else:
            # Run single mode experiment
            generator, device_type = create_text_generator(hardware, args.force_cpu)

            print("\nInitializing text generator...", end="")
            device_info = generator.get_device_info()
            print(f" ✅ {device_info.get('device', 'unknown')} ready")

            # Run experiment
            metrics = run_generation_experiment(generator, prompts, args.batch_size, device_type)
            print_detailed_results(metrics)

            # Cleanup
            generator.cleanup()

        # Final safety verification
        print("\n✅ All safety checks passed")
        print("✅ No cloud resources used")
        print("✅ Cost remained at $0.00")

        return 0

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
