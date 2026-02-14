"""Configuration Management for Radix

This module provides centralized configuration management with safety-first
principles using Pydantic settings. All safety-critical settings are immutable and validated.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class SafetyConfig(BaseSettings):
    """Immutable safety configuration - cannot be modified at runtime."""

    # Core safety settings - NEVER modify these defaults
    dry_run: bool = Field(default=True, description="Enable dry-run mode (REQUIRED)")
    no_deploy_mode: bool = Field(
        default=True, description="Disable deployment operations (REQUIRED)"
    )
    cost_cap_usd: float = Field(default=0.00, description="Maximum cost in USD (REQUIRED: 0.00)")
    max_job_cost_usd: float = Field(
        default=0.00, description="Maximum per-job cost in USD (REQUIRED: 0.00)"
    )

    @field_validator("dry_run")
    @classmethod
    def validate_dry_run(cls, v):
        from .mode import is_production

        if is_production():
            return v  # Allow any value in production
        if not v:
            raise ValueError("DRY_RUN must be True for safety")
        return v

    @field_validator("cost_cap_usd")
    @classmethod
    def validate_cost_cap(cls, v):
        from .mode import is_production

        if is_production():
            if v <= 0:
                raise ValueError("COST_CAP_USD must be > 0 in production mode")
            return v
        if v != 0.0:
            raise ValueError("COST_CAP_USD must be 0.00 for safety")
        return v

    @field_validator("max_job_cost_usd")
    @classmethod
    def validate_max_job_cost(cls, v):
        from .mode import is_production

        if is_production():
            if v <= 0:
                raise ValueError("MAX_JOB_COST_USD must be > 0 in production mode")
            return v
        if v != 0.0:
            raise ValueError("MAX_JOB_COST_USD must be 0.00 for safety")
        return v

    @field_validator("no_deploy_mode")
    @classmethod
    def validate_no_deploy(cls, v):
        from .mode import is_production

        if is_production():
            return v  # Allow any value in production
        if not v:
            raise ValueError("NO_DEPLOY_MODE must be True for safety")
        return v

    model_config = {"frozen": True}  # Immutable


@dataclass
class ExecutionConfig:
    """Configuration for job execution."""

    max_parallelism: int = 4
    default_executor: str = "local_subprocess"
    enable_gpu: bool = False
    ray_local_mode: bool = True
    ray_num_cpus: int = 4
    ray_num_gpus: int = 0

    def __post_init__(self):
        """Validate execution configuration."""
        if self.max_parallelism < 1:
            raise ValueError("max_parallelism must be at least 1")

        if self.default_executor not in ["local_subprocess", "threadpool", "ray_local"]:
            raise ValueError(f"Invalid executor: {self.default_executor}")

        # Safety: GPU must be local only
        if self.enable_gpu and not self._is_local_gpu_safe():
            raise ValueError("GPU usage must be local-only for safety")

        # Safety: Ray must be in local mode (unless production)
        if not self.ray_local_mode:
            from .mode import is_production

            if not is_production():
                raise ValueError("Ray must be in local mode for safety")

        if self.ray_num_gpus > 0 and not self.enable_gpu:
            raise ValueError("Cannot allocate GPUs when GPU is disabled")

    def _is_local_gpu_safe(self) -> bool:
        """Verify GPU usage is local and safe."""
        # In a real implementation, this would check for cloud GPU instances
        # For research purposes, we assume local GPU is safe
        return True


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""

    log_level: str = "INFO"
    log_format: str = "console"
    enable_metrics: bool = True
    metrics_interval: int = 60
    enable_profiling: bool = False

    def __post_init__(self):
        """Validate logging configuration."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_levels:
            raise ValueError(f"Invalid log level: {self.log_level}")

        valid_formats = ["console", "json"]
        if self.log_format not in valid_formats:
            raise ValueError(f"Invalid log format: {self.log_format}")

        if self.metrics_interval < 1:
            raise ValueError("metrics_interval must be at least 1 second")


@dataclass
class SwarmConfig:
    """Configuration for swarm simulation."""

    node_count: int = 3
    heartbeat_interval: int = 5
    failure_probability: float = 0.1
    recovery_time: int = 30

    def __post_init__(self):
        """Validate swarm configuration."""
        if self.node_count < 1:
            raise ValueError("node_count must be at least 1")

        if self.heartbeat_interval < 1:
            raise ValueError("heartbeat_interval must be at least 1 second")

        if not 0.0 <= self.failure_probability <= 1.0:
            raise ValueError("failure_probability must be between 0.0 and 1.0")

        if self.recovery_time < 0:
            raise ValueError("recovery_time must be non-negative")


@dataclass
class BatchingConfig:
    """Configuration for batching and microbatching."""

    default_batch_size: int = 32
    max_batch_wait: float = 5.0
    enable_dynamic_batching: bool = True
    microbatch_size: int = 8

    def __post_init__(self):
        """Validate batching configuration."""
        if self.default_batch_size < 1:
            raise ValueError("default_batch_size must be at least 1")

        if self.max_batch_wait < 0:
            raise ValueError("max_batch_wait must be non-negative")

        if self.microbatch_size < 1:
            raise ValueError("microbatch_size must be at least 1")

        if self.microbatch_size > self.default_batch_size:
            raise ValueError("microbatch_size cannot be larger than default_batch_size")


@dataclass
class ResearchConfig:
    """Configuration for research experiments."""

    experiment_name: str = "default"
    experiment_version: str = "v1"
    random_seed: int = 42
    log_experiments: bool = True
    results_dir: str = "./results"

    def __post_init__(self):
        """Validate research configuration."""
        if not self.experiment_name:
            raise ValueError("experiment_name cannot be empty")

        if not self.experiment_version:
            raise ValueError("experiment_version cannot be empty")

        # Ensure results directory is safe (local only)
        results_path = Path(self.results_dir)
        if results_path.is_absolute() and not str(results_path).startswith(("/tmp", "/var/tmp")):
            # Allow absolute paths only in safe temp directories
            if not str(results_path).startswith(str(Path.home())):
                raise ValueError("results_dir must be relative or in user home directory")


@dataclass
class RadixConfig:
    """Main configuration class for Radix."""

    # Immutable safety configuration
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    # Mutable operational configurations
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    swarm: SwarmConfig = field(default_factory=SwarmConfig)
    batching: BatchingConfig = field(default_factory=BatchingConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)

    # Additional settings
    debug: bool = False

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "RadixConfig":
        """Load configuration from environment variables."""

        # Load from .env file if specified
        if env_file:
            cls._load_env_file(env_file)

        # Mode-aware defaults
        from .mode import is_production

        prod = is_production()

        # Create safety config from environment
        safety = SafetyConfig(
            dry_run=cls._get_bool_env("DRY_RUN", False if prod else True),
            no_deploy_mode=cls._get_bool_env("NO_DEPLOY_MODE", False if prod else True),
            cost_cap_usd=cls._get_float_env("COST_CAP_USD", 100.0 if prod else 0.0),
            max_job_cost_usd=cls._get_float_env("MAX_JOB_COST_USD", 10.0 if prod else 0.0),
        )

        # Create execution config from environment
        execution = ExecutionConfig(
            max_parallelism=cls._get_int_env("MAX_PARALLELISM", 4),
            default_executor=os.getenv("DEFAULT_EXECUTOR", "local_subprocess"),
            enable_gpu=cls._get_bool_env("ENABLE_GPU", False),
            ray_local_mode=cls._get_bool_env("RAY_LOCAL_MODE", True),
            ray_num_cpus=cls._get_int_env("RAY_NUM_CPUS", 4),
            ray_num_gpus=cls._get_int_env("RAY_NUM_GPUS", 0),
        )

        # Create logging config from environment
        logging = LoggingConfig(
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_format=os.getenv("LOG_FORMAT", "console"),
            enable_metrics=cls._get_bool_env("ENABLE_METRICS", True),
            metrics_interval=cls._get_int_env("METRICS_INTERVAL", 60),
            enable_profiling=cls._get_bool_env("ENABLE_PROFILING", False),
        )

        # Create swarm config from environment
        swarm = SwarmConfig(
            node_count=cls._get_int_env("SWARM_NODE_COUNT", 3),
            heartbeat_interval=cls._get_int_env("HEARTBEAT_INTERVAL", 5),
            failure_probability=cls._get_float_env("FAILURE_PROBABILITY", 0.1),
            recovery_time=cls._get_int_env("RECOVERY_TIME", 30),
        )

        # Create batching config from environment
        batching = BatchingConfig(
            default_batch_size=cls._get_int_env("DEFAULT_BATCH_SIZE", 32),
            max_batch_wait=cls._get_float_env("MAX_BATCH_WAIT", 5.0),
            enable_dynamic_batching=cls._get_bool_env("ENABLE_DYNAMIC_BATCHING", True),
            microbatch_size=cls._get_int_env("MICROBATCH_SIZE", 8),
        )

        # Create research config from environment
        research = ResearchConfig(
            experiment_name=os.getenv("EXPERIMENT_NAME", "default"),
            experiment_version=os.getenv("EXPERIMENT_VERSION", "v1"),
            random_seed=cls._get_int_env("RANDOM_SEED", 42),
            log_experiments=cls._get_bool_env("LOG_EXPERIMENTS", True),
            results_dir=os.getenv("RESULTS_DIR", "./results"),
        )

        # Additional settings
        debug = cls._get_bool_env("DEBUG", False)

        return cls(
            safety=safety,
            execution=execution,
            logging=logging,
            swarm=swarm,
            batching=batching,
            research=research,
            debug=debug,
        )

    @staticmethod
    def _load_env_file(env_file: str):
        """Load environment variables from file."""
        env_path = Path(env_file)
        if not env_path.exists():
            return

        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()

    @staticmethod
    def _get_bool_env(key: str, default: bool) -> bool:
        """Get boolean environment variable."""
        value = os.getenv(key, str(default)).lower()
        return value in ("true", "1", "yes", "on")

    @staticmethod
    def _get_int_env(key: str, default: int) -> int:
        """Get integer environment variable."""
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            return default

    @staticmethod
    def _get_float_env(key: str, default: float) -> float:
        """Get float environment variable."""
        try:
            return float(os.getenv(key, str(default)))
        except ValueError:
            return default

    def validate(self) -> List[str]:
        """Validate the entire configuration and return any errors."""
        errors = []

        # Safety validation is handled in SafetyConfig.__post_init__
        try:
            # Re-create safety config to trigger validation
            SafetyConfig(
                dry_run=self.safety.dry_run,
                no_deploy_mode=self.safety.no_deploy_mode,
                cost_cap_usd=self.safety.cost_cap_usd,
                max_job_cost_usd=self.safety.max_job_cost_usd,
            )
        except ValueError as e:
            errors.append(f"Safety configuration error: {e}")

        # Cross-component validation
        if self.execution.enable_gpu and self.execution.ray_num_gpus == 0:
            if self.execution.default_executor == "ray_local":
                errors.append("GPU enabled but Ray has 0 GPUs allocated")

        if self.batching.microbatch_size > self.batching.default_batch_size:
            errors.append("Microbatch size cannot be larger than default batch size")

        if self.execution.max_parallelism > 64:
            errors.append("max_parallelism > 64 may cause resource exhaustion")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "safety": {
                "dry_run": self.safety.dry_run,
                "no_deploy_mode": self.safety.no_deploy_mode,
                "cost_cap_usd": self.safety.cost_cap_usd,
                "max_job_cost_usd": self.safety.max_job_cost_usd,
            },
            "execution": {
                "max_parallelism": self.execution.max_parallelism,
                "default_executor": self.execution.default_executor,
                "enable_gpu": self.execution.enable_gpu,
                "ray_local_mode": self.execution.ray_local_mode,
                "ray_num_cpus": self.execution.ray_num_cpus,
                "ray_num_gpus": self.execution.ray_num_gpus,
            },
            "logging": {
                "log_level": self.logging.log_level,
                "log_format": self.logging.log_format,
                "enable_metrics": self.logging.enable_metrics,
                "metrics_interval": self.logging.metrics_interval,
                "enable_profiling": self.logging.enable_profiling,
            },
            "swarm": {
                "node_count": self.swarm.node_count,
                "heartbeat_interval": self.swarm.heartbeat_interval,
                "failure_probability": self.swarm.failure_probability,
                "recovery_time": self.swarm.recovery_time,
            },
            "batching": {
                "default_batch_size": self.batching.default_batch_size,
                "max_batch_wait": self.batching.max_batch_wait,
                "enable_dynamic_batching": self.batching.enable_dynamic_batching,
                "microbatch_size": self.batching.microbatch_size,
            },
            "research": {
                "experiment_name": self.research.experiment_name,
                "experiment_version": self.research.experiment_version,
                "random_seed": self.research.random_seed,
                "log_experiments": self.research.log_experiments,
                "results_dir": self.research.results_dir,
            },
            "debug": self.debug,
        }

    def __str__(self) -> str:
        """String representation of configuration."""
        return (
            f"RadixConfig(dry_run={self.safety.dry_run}, cost_cap=${self.safety.cost_cap_usd:.2f})"
        )


# Global configuration instance
_global_config: Optional[RadixConfig] = None


def get_config() -> RadixConfig:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = RadixConfig.from_env()
    return _global_config


def set_config(config: RadixConfig):
    """Set the global configuration instance."""
    global _global_config

    # Validate configuration before setting
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration validation failed: {errors}")

    _global_config = config


def reset_config():
    """Reset the global configuration to default."""
    global _global_config
    _global_config = None
