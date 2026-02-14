"""Discover CloudWatch Logs log groups."""

import logging
from typing import List

from ..models import DiscoveredResource
from .base import BaseDiscoverer

logger = logging.getLogger(__name__)


class CloudWatchDiscoverer(BaseDiscoverer):
    """Discover CloudWatch Logs log groups via ``describe_log_groups``.

    Results are paginated to handle accounts with many log groups.
    """

    service_name = "cloudwatch"

    def discover(self) -> List[DiscoveredResource]:
        resources: List[DiscoveredResource] = []
        try:
            client = self._client("logs")
            log_groups = self._list_all_log_groups(client)

            for lg in log_groups:
                lg_name = lg.get("logGroupName", "")
                lg_arn = lg.get("arn", "")
                label = self._sanitize_name(lg_name)

                properties = {
                    "log_group_name": lg_name,
                    "retention_in_days": lg.get("retentionInDays", 0),
                    "stored_bytes": lg.get("storedBytes", 0),
                    "kms_key_id": lg.get("kmsKeyId", ""),
                    "creation_time": lg.get("creationTime", 0),
                    "metric_filter_count": lg.get("metricFilterCount", 0),
                    "data_protection_status": lg.get(
                        "dataProtectionStatus", ""
                    ),
                }

                resources.append(
                    DiscoveredResource(
                        service="cloudwatch",
                        resource_type="aws_cloudwatch_log_group",
                        resource_id=lg_name,
                        arn=lg_arn,
                        name=lg_name,
                        properties=properties,
                        terraform_address=f"aws_cloudwatch_log_group.{label}",
                    )
                )

        except Exception as exc:
            logger.warning("CloudWatch Logs discovery failed: %s", exc)

        return resources

    # ------------------------------------------------------------------
    # Pagination
    # ------------------------------------------------------------------

    @staticmethod
    def _list_all_log_groups(client) -> list:
        """Paginate through describe_log_groups."""
        groups: list = []
        paginator = client.get_paginator("describe_log_groups")
        for page in paginator.paginate():
            groups.extend(page.get("logGroups", []))
        return groups
