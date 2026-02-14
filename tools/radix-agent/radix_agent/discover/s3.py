"""Discover S3 buckets and their tags."""

import logging
from typing import List

from ..models import DiscoveredResource
from .base import BaseDiscoverer

logger = logging.getLogger(__name__)


class S3Discoverer(BaseDiscoverer):
    """Discover S3 buckets.

    S3 is a global service — ``list_buckets`` returns all buckets regardless
    of region.  For each bucket, ``get_bucket_tagging`` is called to capture
    tags (gracefully handling the ``NoSuchTagConfiguration`` error when a
    bucket has no tags).
    """

    service_name = "s3"

    def discover(self) -> List[DiscoveredResource]:
        resources: List[DiscoveredResource] = []
        try:
            client = self._client("s3")
            resp = client.list_buckets()

            for bucket in resp.get("Buckets", []):
                bucket_name = bucket.get("Name", "")
                label = self._sanitize_name(bucket_name)

                # Fetch tags (may raise NoSuchTagConfiguration)
                tags = self._get_bucket_tags(client, bucket_name)

                # Try to get bucket region
                bucket_region = self._get_bucket_region(client, bucket_name)

                properties = {
                    "bucket_name": bucket_name,
                    "creation_date": str(bucket.get("CreationDate", "")),
                    "region": bucket_region,
                }

                resources.append(
                    DiscoveredResource(
                        service="s3",
                        resource_type="aws_s3_bucket",
                        resource_id=bucket_name,
                        arn=f"arn:aws:s3:::{bucket_name}",
                        name=bucket_name,
                        tags=tags,
                        properties=properties,
                        terraform_address=f"aws_s3_bucket.{label}",
                    )
                )

        except Exception as exc:
            logger.warning("S3 discovery failed: %s", exc)

        return resources

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_bucket_tags(client, bucket_name: str) -> dict:
        """Return tags for a bucket, or an empty dict if none exist."""
        try:
            resp = client.get_bucket_tagging(Bucket=bucket_name)
            tag_set = resp.get("TagSet", [])
            return {t["Key"]: t["Value"] for t in tag_set if "Key" in t}
        except client.exceptions.ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code == "NoSuchTagConfiguration":
                return {}
            # AccessDenied or other — log and continue
            logger.warning(
                "Failed to get tags for bucket %s: %s", bucket_name, exc
            )
            return {}
        except Exception as exc:
            logger.warning(
                "Failed to get tags for bucket %s: %s", bucket_name, exc
            )
            return {}

    @staticmethod
    def _get_bucket_region(client, bucket_name: str) -> str:
        """Return the region a bucket resides in."""
        try:
            resp = client.get_bucket_location(Bucket=bucket_name)
            # LocationConstraint is None for us-east-1
            location = resp.get("LocationConstraint")
            return location if location else "us-east-1"
        except Exception:
            return ""
