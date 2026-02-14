"""Pydantic models for discovered AWS resources."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DiscoveredResource(BaseModel):
    """A single discovered AWS resource."""

    service: str
    resource_type: str
    resource_id: str
    arn: Optional[str] = None
    name: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)
    properties: Dict[str, Any] = Field(default_factory=dict)
    terraform_address: Optional[str] = None
    exists_in_tf: bool = False
    tf_file: Optional[str] = None


class ServiceInventory(BaseModel):
    """Inventory for a single AWS service."""

    service: str
    region: str
    resource_count: int
    resources: List[DiscoveredResource]


class Inventory(BaseModel):
    """Full AWS account inventory."""

    account_id: Optional[str] = None
    region: str
    discovered_at: str
    services: List[ServiceInventory]
    total_resources: int = 0


class ReconcileResult(BaseModel):
    """Result of reconciling inventory against existing Terraform."""

    managed: List[DiscoveredResource] = Field(default_factory=list)
    unmanaged: List[DiscoveredResource] = Field(default_factory=list)
    framework_only: List[str] = Field(default_factory=list)
    summary: Dict[str, int] = Field(default_factory=dict)
