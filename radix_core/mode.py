"""Radix execution mode — development (safe simulation) vs production (real execution)."""

import os
from enum import Enum


class RadixMode(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"


def get_mode() -> RadixMode:
    """Get the current Radix execution mode from RADIX_MODE env var."""
    raw = os.environ.get("RADIX_MODE", "development").lower().strip()
    if raw == "production":
        return RadixMode.PRODUCTION
    return RadixMode.DEVELOPMENT


def is_production() -> bool:
    """Check if running in production mode."""
    return get_mode() == RadixMode.PRODUCTION
