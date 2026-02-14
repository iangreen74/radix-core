"""Discover ECR repositories."""

import logging
from typing import List

from ..models import DiscoveredResource
from .base import BaseDiscoverer

logger = logging.getLogger(__name__)


class EcrDiscoverer(BaseDiscoverer):
    """Discover Amazon ECR repositories via ``describe_repositories``."""

    service_name = "ecr"

    def discover(self) -> List[DiscoveredResource]:
        resources: List[DiscoveredResource] = []
        try:
            client = self._client("ecr")
            repos = self._list_all_repositories(client)

            for repo in repos:
                repo_name = repo.get("repositoryName", "")
                repo_arn = repo.get("repositoryArn", "")
                repo_uri = repo.get("repositoryUri", "")
                registry_id = repo.get("registryId", "")
                label = self._sanitize_name(repo_name)

                properties = {
                    "repository_name": repo_name,
                    "repository_uri": repo_uri,
                    "registry_id": registry_id,
                    "image_tag_mutability": repo.get(
                        "imageTagMutability", "MUTABLE"
                    ),
                    "image_scanning_configuration": repo.get(
                        "imageScanningConfiguration", {}
                    ),
                    "encryption_configuration": repo.get(
                        "encryptionConfiguration", {}
                    ),
                    "created_at": str(repo.get("createdAt", "")),
                }

                resources.append(
                    DiscoveredResource(
                        service="ecr",
                        resource_type="aws_ecr_repository",
                        resource_id=repo_name,
                        arn=repo_arn,
                        name=repo_name,
                        properties=properties,
                        terraform_address=f"aws_ecr_repository.{label}",
                    )
                )

        except Exception as exc:
            logger.warning("ECR discovery failed: %s", exc)

        return resources

    # ------------------------------------------------------------------
    # Pagination
    # ------------------------------------------------------------------

    @staticmethod
    def _list_all_repositories(client) -> list:
        """Paginate through describe_repositories."""
        repos: list = []
        paginator = client.get_paginator("describe_repositories")
        for page in paginator.paginate():
            repos.extend(page.get("repositories", []))
        return repos
