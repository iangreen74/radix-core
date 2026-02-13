"""Configuration management for the information-theoretic GPU scheduler."""

import os
from typing import Dict, Any
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings, Field


class SchedulerConfig(BaseSettings):
    """Configuration for the scheduler-agent service."""

    # Information theory parameters
    lambda_uncertainty: float = Field(default=0.3, description="Uncertainty weight (λ)")
    beta_exploration: float = Field(default=0.7, description="Information gain weight (β)")
    gamma_interference: float = Field(default=0.0, description="Interference penalty weight (γ)")
    tau_squared: float = Field(default=0.15, description="Observation noise variance (τ²)")

    # Exploration control
    exploration_cap: float = Field(default=0.25, description="Maximum exploration ratio")

    # Feature flags
    enable_interference: bool = Field(default=False, description="Enable interference learning")
    enable_sinkhorn: bool = Field(default=False, description="Enable Sinkhorn assignment")
    enable_mutation: bool = Field(default=True, description="Enable webhook mutation")

    # Storage and retention
    retention_observations: int = Field(default=1000, description="Max observations per (job_type, gpu_type)")
    checkpoint_interval: int = Field(default=300, description="Checkpoint interval in seconds")

    # Service configuration
    host: str = Field(default="0.0.0.0", description="Service host")
    port: int = Field(default=8080, description="Service port")
    log_level: str = Field(default="INFO", description="Log level")

    # Storage backend
    redis_url: str = Field(default="", description="Redis URL (optional)")
    sqlite_path: str = Field(default="/tmp/scheduler.db", description="SQLite database path")

    class Config:
        env_prefix = "SCHEDULER_"
        case_sensitive = False


# Global config instance
_config: SchedulerConfig = None


def get_config() -> SchedulerConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = SchedulerConfig()
    return _config


def reload_config():
    """Reload configuration from environment."""
    global _config
    _config = SchedulerConfig()
    return _config
