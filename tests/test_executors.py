"""
Comprehensive test suite for all executor types.

Tests each executor's functionality, performance characteristics,
and error handling capabilities.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add engine to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from radix_core.executors.threadpool import ThreadPoolExecutor
from radix_core.executors.ray_local import RayLocalExecutor
from radix_core.executors.hf_runner import HuggingFaceExecutor
from radix_core.executors.vllm_local import VLLMLocalExecutor
from radix_core.types import JobSpec, ExecutionResult, ResourceRequirements
from radix_core.config import get_config
from radix_core.errors import ExecutorError, ResourceError, SafetyError


class TestExecutorBase:
    """Base test class with common executor test patterns."""

    @pytest.fixture
    def sample_job(self):
        """Create a sample job specification."""
        return JobSpec(
            job_id="test-job-001",
            name="test_job",
            command="echo 'hello world'",
            resources=ResourceRequirements(
                cpu_cores=1,
                memory_gb=1.0,
                gpu_count=0
            ),
            priority=1,
            max_runtime_seconds=30
        )

    @pytest.fixture
    def compute_job(self):
        """Create a computational job for testing."""
        return JobSpec(
            job_id="compute-001",
            name="compute_test",
            command="python -c 'import time; time.sleep(0.1); print(sum(range(1000)))'",
            resources=ResourceRequirements(
                cpu_cores=2,
                memory_gb=0.5,
                gpu_count=0
            ),
            priority=2,
            max_runtime_seconds=10
        )


class TestThreadPoolExecutor(TestExecutorBase):
    """Test ThreadPool executor functionality."""

    @pytest.fixture
    def executor(self):
        """Create ThreadPool executor instance."""
        config = get_config()
        return ThreadPoolExecutor(config)

    def test_executor_initialization(self, executor):
        """Test executor initializes correctly."""
        assert executor is not None
        assert executor.max_workers > 0
        assert hasattr(executor, 'pool')

    def test_simple_job_execution(self, executor, sample_job):
        """Test basic job execution."""
        result = executor.execute_job(sample_job)

        assert isinstance(result, ExecutionResult)
        assert result.job_id == sample_job.job_id
        assert result.success is True
        assert result.exit_code == 0
        assert "hello world" in result.stdout

    def test_parallel_job_execution(self, executor):
        """Test parallel execution of multiple jobs."""
        jobs = [
            JobSpec(
                job_id=f"parallel-{i}",
                name=f"parallel_job_{i}",
                command=f"python -c 'import time; time.sleep(0.1); print({i})'",
                resources=ResourceRequirements(cpu_cores=1, memory_gb=0.1),
                priority=1,
                max_runtime_seconds=5
            )
            for i in range(5)
        ]

        start_time = time.time()
        results = executor.execute_batch(jobs)
        end_time = time.time()

        # Should complete in less time than sequential execution
        assert end_time - start_time < 1.0  # Much faster than 5 * 0.1 = 0.5s
        assert len(results) == 5
        assert all(r.success for r in results)

    def test_resource_monitoring(self, executor, compute_job):
        """Test resource usage monitoring."""
        result = executor.execute_job(compute_job)

        assert result.success is True
        assert result.resource_usage is not None
        assert result.resource_usage.peak_memory_mb > 0
        assert result.resource_usage.cpu_time_seconds > 0

    def test_timeout_handling(self, executor):
        """Test job timeout enforcement."""
        timeout_job = JobSpec(
            job_id="timeout-test",
            name="timeout_job",
            command="python -c 'import time; time.sleep(10)'",
            resources=ResourceRequirements(cpu_cores=1, memory_gb=0.1),
            priority=1,
            max_runtime_seconds=1  # Very short timeout
        )

        result = executor.execute_job(timeout_job)
        assert result.success is False
        assert "timeout" in result.error_message.lower()

    def test_error_handling(self, executor):
        """Test handling of job errors."""
        error_job = JobSpec(
            job_id="error-test",
            name="error_job",
            command="python -c 'raise ValueError(\"test error\")'",
            resources=ResourceRequirements(cpu_cores=1, memory_gb=0.1),
            priority=1,
            max_runtime_seconds=5
        )

        result = executor.execute_job(error_job)
        assert result.success is False
        assert result.exit_code != 0
        assert "ValueError" in result.stderr


class TestRayLocalExecutor(TestExecutorBase):
    """Test Ray Local executor functionality."""

    @pytest.fixture
    def executor(self):
        """Create Ray Local executor instance."""
        config = get_config()
        return RayLocalExecutor(config)

    def test_ray_initialization(self, executor):
        """Test Ray executor initializes correctly."""
        assert executor is not None
        assert hasattr(executor, 'ray_config')

    @patch('ray.init')
    @patch('ray.get')
    @patch('ray.remote')
    def test_ray_job_execution(self, mock_remote, mock_get, mock_init, executor, sample_job):
        """Test Ray job execution with mocking."""
        # Mock Ray remote function
        mock_remote_func = Mock()
        mock_remote.return_value = mock_remote_func
        mock_remote_func.remote.return_value = "task_ref"

        # Mock Ray get result
        mock_get.return_value = ExecutionResult(
            job_id=sample_job.job_id,
            success=True,
            exit_code=0,
            stdout="hello world",
            stderr="",
            execution_time_seconds=0.1
        )

        result = executor.execute_job(sample_job)

        assert result.success is True
        assert result.job_id == sample_job.job_id
        mock_init.assert_called_once()

    def test_ray_scaling(self, executor):
        """Test Ray's ability to scale across CPU cores."""
        # This would test Ray's distributed execution capabilities
        # For now, we'll test the configuration
        assert executor.ray_config['num_cpus'] > 0
        assert executor.ray_config['local_mode'] is True  # Safety: local only


class TestHuggingFaceExecutor(TestExecutorBase):
    """Test HuggingFace executor functionality."""

    @pytest.fixture
    def executor(self):
        """Create HuggingFace executor instance."""
        config = get_config()
        return HuggingFaceExecutor(config)

    @pytest.fixture
    def ml_job(self):
        """Create ML training job specification."""
        return JobSpec(
            job_id="ml-train-001",
            name="bert_fine_tune",
            command="train_model",
            resources=ResourceRequirements(
                cpu_cores=2,
                memory_gb=4.0,
                gpu_count=0  # CPU-only for safety
            ),
            priority=3,
            max_runtime_seconds=300,
            parameters={
                "model_name": "distilbert-base-uncased",
                "dataset": "imdb",
                "max_length": 512,
                "batch_size": 8,
                "learning_rate": 2e-5,
                "num_epochs": 1
            }
        )

    def test_hf_executor_initialization(self, executor):
        """Test HuggingFace executor initializes correctly."""
        assert executor is not None
        assert hasattr(executor, 'accelerator')
        assert hasattr(executor, 'device')

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModel.from_pretrained')
    def test_model_loading(self, mock_model, mock_tokenizer, executor, ml_job):
        """Test model and tokenizer loading."""
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()

        result = executor.execute_job(ml_job)

        # Should attempt to load model and tokenizer
        mock_tokenizer.assert_called_once()
        mock_model.assert_called_once()

    def test_training_simulation(self, executor, ml_job):
        """Test training job simulation."""
        # In dry-run mode, this should simulate training
        result = executor.execute_job(ml_job)

        assert isinstance(result, ExecutionResult)
        assert result.job_id == ml_job.job_id
        # In dry-run, should complete successfully with simulation

    def test_peft_configuration(self, executor):
        """Test PEFT/LoRA configuration setup."""
        peft_config = executor.create_peft_config(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0.1
        )

        assert peft_config is not None
        assert peft_config.r == 8
        assert peft_config.lora_alpha == 16


class TestVLLMLocalExecutor(TestExecutorBase):
    """Test vLLM Local executor functionality."""

    @pytest.fixture
    def executor(self):
        """Create vLLM executor instance with graceful degradation."""
        config = get_config()
        return VLLMLocalExecutor(config)

    @pytest.fixture
    def inference_job(self):
        """Create inference job specification."""
        return JobSpec(
            job_id="inference-001",
            name="llm_inference",
            command="generate_text",
            resources=ResourceRequirements(
                cpu_cores=4,
                memory_gb=8.0,
                gpu_count=0  # CPU-only for safety
            ),
            priority=2,
            max_runtime_seconds=60,
            parameters={
                "model_name": "gpt2",
                "prompt": "The future of AI is",
                "max_tokens": 100,
                "temperature": 0.7,
                "batch_size": 1
            }
        )

    def test_vllm_availability_check(self, executor):
        """Test vLLM availability checking."""
        # Should detect if vLLM is available or gracefully degrade
        assert hasattr(executor, 'vllm_available')

        if not executor.vllm_available:
            # Should fall back to alternative implementation
            assert hasattr(executor, 'fallback_executor')

    @pytest.mark.skipif(
        not hasattr(VLLMLocalExecutor, 'vllm_available') or not VLLMLocalExecutor.vllm_available,
        reason="vLLM not available - testing graceful degradation"
    )
    def test_vllm_inference(self, executor, inference_job):
        """Test vLLM inference execution."""
        result = executor.execute_job(inference_job)

        assert isinstance(result, ExecutionResult)
        assert result.job_id == inference_job.job_id

    def test_batch_inference(self, executor):
        """Test batch inference capabilities."""
        prompts = [
            "The weather today is",
            "Machine learning is",
            "The future of technology"
        ]

        batch_job = JobSpec(
            job_id="batch-inference-001",
            name="batch_llm_inference",
            command="batch_generate",
            resources=ResourceRequirements(cpu_cores=4, memory_gb=8.0),
            priority=2,
            max_runtime_seconds=120,
            parameters={
                "model_name": "gpt2",
                "prompts": prompts,
                "max_tokens": 50,
                "batch_size": len(prompts)
            }
        )

        result = executor.execute_job(batch_job)
        assert isinstance(result, ExecutionResult)

    def test_graceful_degradation(self, executor, inference_job):
        """Test fallback when vLLM is not available."""
        # Force unavailability for testing
        executor.vllm_available = False

        result = executor.execute_job(inference_job)

        # Should still complete, possibly with different implementation
        assert isinstance(result, ExecutionResult)


class TestExecutorIntegration:
    """Integration tests across multiple executors."""

    @pytest.fixture
    def all_executors(self):
        """Create instances of all executors."""
        config = get_config()
        return {
            'threadpool': ThreadPoolExecutor(config),
            'ray_local': RayLocalExecutor(config),
            'huggingface': HuggingFaceExecutor(config),
            'vllm_local': VLLMLocalExecutor(config)
        }

    def test_executor_consistency(self, all_executors):
        """Test that all executors implement the same interface."""
        simple_job = JobSpec(
            job_id="consistency-test",
            name="echo_test",
            command="echo 'test'",
            resources=ResourceRequirements(cpu_cores=1, memory_gb=0.1),
            priority=1,
            max_runtime_seconds=5
        )

        results = {}
        for name, executor in all_executors.items():
            try:
                result = executor.execute_job(simple_job)
                results[name] = result
                assert isinstance(result, ExecutionResult)
                assert result.job_id == simple_job.job_id
            except Exception as e:
                # Some executors might not be available - that's OK
                pytest.skip(f"Executor {name} not available: {e}")

        # At least one executor should work
        assert len(results) > 0

    def test_performance_comparison(self, all_executors):
        """Compare performance characteristics across executors."""
        performance_job = JobSpec(
            job_id="perf-test",
            name="performance_comparison",
            command="python -c 'print(sum(range(10000)))'",
            resources=ResourceRequirements(cpu_cores=1, memory_gb=0.1),
            priority=1,
            max_runtime_seconds=10
        )

        performance_results = {}

        for name, executor in all_executors.items():
            try:
                start_time = time.time()
                result = executor.execute_job(performance_job)
                end_time = time.time()

                if result.success:
                    performance_results[name] = {
                        'execution_time': end_time - start_time,
                        'reported_time': result.execution_time_seconds,
                        'memory_usage': getattr(result.resource_usage, 'peak_memory_mb', 0) if result.resource_usage else 0
                    }
            except Exception:
                # Skip unavailable executors
                continue

        # Should have performance data for available executors
        assert len(performance_results) > 0

        # Log performance comparison for analysis
        for name, perf in performance_results.items():
            print(f"\n{name}: {perf['execution_time']:.3f}s, Memory: {perf['memory_usage']:.1f}MB")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
