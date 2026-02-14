"""Discover KMS keys used by radix / EKS."""

import logging
from typing import List

from ..models import DiscoveredResource
from .base import BaseDiscoverer

logger = logging.getLogger(__name__)

# Only include aliases that match these prefixes.
_ALIAS_PREFIXES = ("alias/radix", "alias/eks")


class KmsDiscoverer(BaseDiscoverer):
    """Discover KMS keys associated with radix or EKS.

    Uses ``list_aliases`` to find relevant keys by alias prefix, then
    calls ``describe_key`` for each to capture key metadata.  AWS-managed
    keys (``alias/aws/*``) are ignored.
    """

    service_name = "kms"

    def discover(self) -> List[DiscoveredResource]:
        resources: List[DiscoveredResource] = []
        try:
            client = self._client("kms")
            aliases = self._list_all_aliases(client)

            seen_key_ids: set = set()

            for alias in aliases:
                alias_name = alias.get("AliasName", "")
                key_id = alias.get("TargetKeyId", "")

                # Skip AWS-managed and non-radix/eks aliases
                if not alias_name.startswith(_ALIAS_PREFIXES):
                    continue
                if not key_id or key_id in seen_key_ids:
                    continue
                seen_key_ids.add(key_id)

                try:
                    key_detail = client.describe_key(KeyId=key_id)[
                        "KeyMetadata"
                    ]
                except Exception as exc:
                    logger.warning(
                        "Failed to describe KMS key %s: %s", key_id, exc
                    )
                    continue

                arn = key_detail.get("Arn", "")
                # Strip the "alias/" prefix for the label
                clean_alias = alias_name.replace("alias/", "", 1)
                label = self._sanitize_name(clean_alias)

                properties = {
                    "alias": alias_name,
                    "key_id": key_id,
                    "description": key_detail.get("Description", ""),
                    "key_state": key_detail.get("KeyState", ""),
                    "key_usage": key_detail.get("KeyUsage", ""),
                    "key_spec": key_detail.get("KeySpec", ""),
                    "key_manager": key_detail.get("KeyManager", ""),
                    "origin": key_detail.get("Origin", ""),
                    "multi_region": key_detail.get("MultiRegion", False),
                    "creation_date": str(key_detail.get("CreationDate", "")),
                    "enabled": key_detail.get("Enabled", True),
                }

                resources.append(
                    DiscoveredResource(
                        service="kms",
                        resource_type="aws_kms_key",
                        resource_id=key_id,
                        arn=arn,
                        name=clean_alias,
                        properties=properties,
                        terraform_address=f"aws_kms_key.{label}",
                    )
                )

                # Also record the alias itself
                alias_arn = alias.get("AliasArn", "")
                resources.append(
                    DiscoveredResource(
                        service="kms",
                        resource_type="aws_kms_alias",
                        resource_id=alias_name,
                        arn=alias_arn,
                        name=alias_name,
                        properties={
                            "alias_name": alias_name,
                            "target_key_id": key_id,
                            "target_key_arn": arn,
                        },
                        terraform_address=f"aws_kms_alias.{label}",
                    )
                )

        except Exception as exc:
            logger.warning("KMS discovery failed: %s", exc)

        return resources

    # ------------------------------------------------------------------
    # Pagination
    # ------------------------------------------------------------------

    @staticmethod
    def _list_all_aliases(client) -> list:
        """Paginate through list_aliases."""
        aliases: list = []
        paginator = client.get_paginator("list_aliases")
        for page in paginator.paginate():
            aliases.extend(page.get("Aliases", []))
        return aliases
