"""Abstract base class for AWS resource discoverers."""

from abc import ABC, abstractmethod
from typing import List

import boto3

from ..models import DiscoveredResource


class BaseDiscoverer(ABC):
    """Base class that all service discoverers inherit from.

    Provides a common interface for discovering AWS resources, along with
    helper methods for creating boto3 clients and normalizing AWS tags.
    All subclasses must implement the ``discover`` method and set
    ``service_name`` to a short identifier (e.g. "route53", "ec2").
    """

    service_name: str = ""

    def __init__(self, session: boto3.Session, region: str) -> None:
        self.session = session
        self.region = region

    @abstractmethod
    def discover(self) -> List[DiscoveredResource]:
        """Discover all resources for this service.

        Returns a list of ``DiscoveredResource`` objects.  Implementations
        must use only read-only AWS API calls (Describe / List / Get).
        """
        ...

    def _client(self, service: str):
        """Return a boto3 client for *service* in the configured region."""
        return self.session.client(service, region_name=self.region)

    def _safe_tags(self, tag_list) -> dict:
        """Convert AWS tag list format to a simple ``{Key: Value}`` dict.

        AWS returns tags as ``[{"Key": "k", "Value": "v"}, ...]``.
        This helper gracefully handles ``None`` or unexpected shapes.
        """
        if not tag_list:
            return {}
        return {t.get("Key", ""): t.get("Value", "") for t in tag_list if "Key" in t}

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Turn an arbitrary string into a Terraform-safe identifier.

        Replaces dots, dashes, slashes and other non-alphanumeric characters
        with underscores, strips leading/trailing underscores, and collapses
        runs of underscores.
        """
        import re

        safe = re.sub(r"[^a-zA-Z0-9]", "_", name)
        safe = re.sub(r"_+", "_", safe)
        return safe.strip("_").lower()
