"""Resumable state management for radix-agent."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """Persisted state between agent runs."""

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    phase: str = "idle"
    completed_services: List[str] = Field(default_factory=list)
    inventory_path: Optional[str] = None
    generated_files: List[str] = Field(default_factory=list)
    import_commands: List[str] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)

    def save(self, path: str) -> None:
        """Save state to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str) -> "AgentState":
        """Load state from JSON file, or create new if missing."""
        p = Path(path)
        if p.exists():
            data = json.loads(p.read_text())
            return cls(**data)
        return cls()

    def reset(self) -> None:
        """Reset state for a fresh run."""
        self.run_id = str(uuid.uuid4())[:8]
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.phase = "idle"
        self.completed_services = []
        self.inventory_path = None
        self.generated_files = []
        self.import_commands = []
        self.errors = []

    def add_error(self, service: str, error: str) -> None:
        """Record an error."""
        self.errors.append(
            {
                "service": service,
                "error": error,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
