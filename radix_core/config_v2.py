"""
Production Configuration Management for Radix

This module provides centralized configuration management with safety-first
principles using Pydantic settings. All safety-critical settings are immutable and validated.
"""

from typing import Optional, List
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class SafetyConfig(BaseSettings):
    """Immutable safety configuration - cannot be modified at runtime."""

    # Core safety settings - NEVER modify these defaults
    dry_run: bool = Field(default=True, description="Enable dry-run mode (REQUIRED)")
    no_deploy_mode: bool = Field(default=True, description="Disable deployment operations (REQUIRED)")
    cost_cap_usd: float = Field(default=0.00, description="Maximum cost in USD (REQUIRED: 0.00)")
    max_job_cost_usd: float = Field(default=0.00, description="Maximum per-job cost in USD (REQUIRED: 0.00)")

    @field_validator('dry_run')
    @classmethod
    def validate_dry_run(cls, v):
        if not v:
            raise ValueError("DRY_RUN must be True for safety")
        return v

    @field_validator('cost_cap_usd')
    @classmethod
    def validate_cost_cap(cls, v):
        if v != 0.0:
            raise ValueError("COST_CAP_USD must be 0.00 for safety")
        return v

    @field_validator('max_job_cost_usd')
    @classmethod
    def validate_max_job_cost(cls, v):
        if v != 0.0:
            raise ValueError("MAX_JOB_COST_USD must be 0.00 for safety")
        return v

    @field_validator('no_deploy_mode')
    @classmethod
    def validate_no_deploy(cls, v):
        if not v:
            raise ValueError("NO_DEPLOY_MODE must be True for safety")
        return v

    class Config:
        env_prefix = ""
        case_sensitive = False
        frozen = True  # Immutable


class ExecutionConfig(BaseSettings):
    """Configuration for job execution with industry-grade defaults."""

    max_parallelism: int = Field(default=8, description="Maximum parallel jobs")
    default_executor: str = Field(default="threadpool", description="Default executor type")
    enable_cuda: bool = Field(default=False, description="Enable CUDA support")
    ray_local_mode: bool = Field(default=True, description="Ray local mode only")
    ray_num_cpus: int = Field(default=8, description="Ray CPU allocation")
    ray_num_gpus: int = Field(default=0, description="Ray GPU allocation")

    # Cost simulation rates
    cpu_cost_per_sec_usd: float = Field(default=0.00, description="CPU cost per second (dry-run: 0.00)")
    gpu_cost_per_sec_usd: float = Field(default=0.00, description="GPU cost per second (dry-run: 0.00)")

    @field_validator('max_parallelism')
    @classmethod
    def validate_max_parallelism(cls, v):
        if v < 1 or v > 64:
            raise ValueError("max_parallelism must be between 1 and 64")
        return v

    @field_validator('default_executor')
    @classmethod
    def validate_executor(cls, v):
        valid_executors = ["threadpool", "ray_local", "local_subprocess"]
        if v not in valid_executors:
            raise ValueError(f"default_executor must be one of {valid_executors}")
        return v

    @field_validator('ray_local_mode')
    @classmethod
    def validate_ray_local(cls, v):
        if not v:
            raise ValueError("Ray must be in local mode for safety")
        return v

    @field_validator('cpu_cost_per_sec_usd', 'gpu_cost_per_sec_usd')
    @classmethod
    def validate_cost_rates(cls, v):
        if v != 0.0:
            raise ValueError("All cost rates must be 0.00 in dry-run mode")
        return v

    class Config:
        env_prefix = ""
        case_sensitive = False


class BatchingConfig(BaseSettings):
    """Configuration for dynamic batching and microbatching."""

    batch_latency_ms: int = Field(default=100, description="Maximum batch latency in milliseconds")
    max_batch_size: int = Field(default=32, description="Maximum batch size")
    enable_dynamic_batching: bool = Field(default=True, description="Enable dynamic batching")
    microbatch_size: int = Field(default=8, description="Microbatch size for memory efficiency")

    @field_validator('batch_latency_ms')
    @classmethod
    def validate_latency(cls, v):
        if v < 1 or v > 10000:
            raise ValueError("batch_latency_ms must be between 1 and 10000")
        return v

    @field_validator('max_batch_size')
    @classmethod
    def validate_max_batch(cls, v):
        if v < 1 or v > 1024:
            raise ValueError("max_batch_size must be between 1 and 1024")
        return v

    @field_validator('microbatch_size')
    @classmethod
    def validate_microbatch(cls, v):
        # Note: Cross-field validation moved to model_validator in Pydantic v2
        return v

    class Config:
        env_prefix = ""
        case_sensitive = False


class MLConfig(BaseSettings):
    """Configuration for ML workloads and model management."""

    # Model defaults
    default_model_name: str = Field(default="gpt2", description="Default model for testing")
    max_sequence_length: int = Field(default=512, description="Maximum sequence length")
    precision: str = Field(default="fp16", description="Model precision")

    # PEFT/LoRA settings
    lora_r: int = Field(default=8, description="LoRA rank")
    lora_alpha: int = Field(default=16, description="LoRA alpha")
    lora_dropout: float = Field(default=0.1, description="LoRA dropout")

    # Training settings
    learning_rate: float = Field(default=1e-4, description="Learning rate")
    batch_size: int = Field(default=4, description="Training batch size")
    gradient_accumulation_steps: int = Field(default=4, description="Gradient accumulation")
    max_steps: int = Field(default=100, description="Maximum training steps")

    @field_validator('precision')
    @classmethod
    def validate_precision(cls, v):
        valid_precisions = ["fp16", "fp32", "bf16"]
        if v not in valid_precisions:
            raise ValueError(f"precision must be one of {valid_precisions}")
        return v

    @field_validator('lora_r')
    @classmethod
    def validate_lora_r(cls, v):
        if v < 1 or v > 256:
            raise ValueError("lora_r must be between 1 and 256")
        return v

    class Config:
        env_prefix = ""
        case_sensitive = False


class SwarmConfig(BaseSettings):
    """Configuration for swarm simulation and orchestration."""

    node_count: int = Field(default=3, description="Number of simulated nodes")
    heartbeat_interval_sec: int = Field(default=5, description="Heartbeat interval in seconds")
    failure_probability: float = Field(default=0.1, description="Node failure probability")
    recovery_time_sec: int = Field(default=30, description="Node recovery time in seconds")

    # Membership and rebalancing
    gossip_interval_sec: int = Field(default=2, description="Gossip protocol interval")
    rebalance_threshold: float = Field(default=0.2, description="Load imbalance threshold for rebalancing")
    checkpoint_interval_sec: int = Field(default=60, description="Checkpoint interval")

    @field_validator('node_count')
    @classmethod
    def validate_node_count(cls, v):
        if v < 1 or v > 100:
            raise ValueError("node_count must be between 1 and 100")
        return v

    @field_validator('failure_probability')
    @classmethod
    def validate_failure_prob(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError("failure_probability must be between 0.0 and 1.0")
        return v

    class Config:
        env_prefix = ""
        case_sensitive = False


class LoggingConfig(BaseSettings):
    """Configuration for structured logging and metrics."""

    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="rich", description="Log format (rich, json)")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_interval_sec: int = Field(default=10, description="Metrics collection interval")

    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()

    @field_validator('log_format')
    @classmethod
    def validate_log_format(cls, v):
        valid_formats = ["rich", "json", "console"]
        if v not in valid_formats:
            raise ValueError(f"log_format must be one of {valid_formats}")
        return v

    class Config:
        env_prefix = ""
        case_sensitive = False


class RadixConfig(BaseSettings):
    """Main configuration class combining all settings."""

    # Component configurations
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    batching: BatchingConfig = Field(default_factory=BatchingConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    swarm: SwarmConfig = Field(default_factory=SwarmConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Global settings
    debug: bool = Field(default=False, description="Enable debug mode")
    experiment_name: str = Field(default="radix_experiment", description="Experiment name")
    results_dir: str = Field(default="./results", description="Results directory")

    def validate_all(self) -> List[str]:
        """Validate the entire configuration and return any errors."""
        errors = []

        # Cross-component validation
        if self.execution.enable_cuda and self.execution.ray_num_gpus == 0:
            if self.execution.default_executor == "ray_local":
                errors.append("CUDA enabled but Ray has 0 GPUs allocated")

        if self.batching.microbatch_size > self.batching.max_batch_size:
            errors.append("Microbatch size cannot be larger than max batch size")

        if self.execution.max_parallelism > 32:
            errors.append("max_parallelism > 32 may cause resource exhaustion")

        # Ensure results directory is safe (local only)
        results_path = Path(self.results_dir)
        if results_path.is_absolute() and not str(results_path).startswith(("/tmp", "/var/tmp")):
            if not str(results_path).startswith(str(Path.home())):
                errors.append("results_dir must be relative or in user home directory")

        return errors

    class Config:
        env_prefix = ""
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global configuration instance
_global_config: Optional[RadixConfig] = None


def get_config() -> RadixConfig:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = RadixConfig()

        # Validate configuration on first load
        errors = _global_config.validate_all()
        if errors:
            raise ValueError(f"Configuration validation failed: {errors}")

    return _global_config


def set_config(config: RadixConfig):
    """Set the global configuration instance."""
    global _global_config

    # Validate configuration before setting
    errors = config.validate_all()
    if errors:
        raise ValueError(f"Configuration validation failed: {errors}")

    _global_config = config


def reset_config():
    """Reset the global configuration to default."""
    global _global_config
    _global_config = None
