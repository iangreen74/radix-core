"""Dashboard configuration."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Dashboard settings, loaded from env vars."""

    # Service URLs
    observer_url: str = "http://radix-observer:8080"
    scheduler_url: str = "http://radix-scheduler-agent:8080"

    # Dashboard settings
    dashboard_port: int = 8080
    dashboard_title: str = "Radix Core"
    refresh_interval_seconds: int = 5

    # Build metadata (injected at image build time)
    build_commit: str = ""
    build_branch: str = ""
    build_timestamp: str = ""

    # Cluster identity
    cluster_name: str = ""

    # Cluster info (injected by Helm)
    radix_namespace: str = "default"
    radix_cluster_uid: str = ""
    radix_license_mode: str = "offline"
    radix_enforcement_enabled: str = "false"
    radix_savings_method: str = "throughput_delta"

    # Feature flags
    radix_mvp_instant_preview: str = "true"
    radix_ts_retention_days: int = 7

    model_config = {"env_prefix": "", "case_sensitive": False}


def get_settings() -> Settings:
    return Settings()
