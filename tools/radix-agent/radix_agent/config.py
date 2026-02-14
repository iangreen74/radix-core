"""Agent configuration via Pydantic Settings."""

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentConfig(BaseSettings):
    """Configuration for radix-agent."""

    aws_region: str = "us-west-2"
    aws_profile: Optional[str] = None
    output_dir: str = "tools/radix-agent/output"
    tf_dir: str = "infra/aws"
    state_file: str = "tools/radix-agent/.agent-state.json"

    model_config = SettingsConfigDict(env_prefix="RADIX_AGENT_")


def get_config() -> AgentConfig:
    """Return agent configuration."""
    return AgentConfig()
