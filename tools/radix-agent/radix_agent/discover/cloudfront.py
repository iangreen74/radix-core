"""Discover CloudFront distributions."""

import logging
from typing import List

from ..models import DiscoveredResource
from .base import BaseDiscoverer

logger = logging.getLogger(__name__)


class CloudFrontDiscoverer(BaseDiscoverer):
    """Discover CloudFront distributions.

    CloudFront is a global service — all API calls are made against
    ``us-east-1`` regardless of the configured region.
    """

    service_name = "cloudfront"

    def discover(self) -> List[DiscoveredResource]:
        resources: List[DiscoveredResource] = []
        try:
            # CloudFront is global; always use us-east-1
            client = self.session.client("cloudfront", region_name="us-east-1")
            distributions = self._list_all_distributions(client)

            for dist in distributions:
                dist_id = dist["Id"]
                domain = dist.get("DomainName", "")
                aliases = (
                    dist.get("Aliases", {}).get("Items", [])
                    if dist.get("Aliases", {}).get("Quantity", 0) > 0
                    else []
                )

                # Use the first alias (or distribution id) for the TF name
                label = self._sanitize_name(aliases[0] if aliases else dist_id)
                tf_addr = f"aws_cloudfront_distribution.{label}"

                # Collect origins
                origin_items = dist.get("Origins", {}).get("Items", [])
                origins = []
                for o in origin_items:
                    origins.append(
                        {
                            "domain_name": o.get("DomainName", ""),
                            "origin_id": o.get("Id", ""),
                            "origin_path": o.get("OriginPath", ""),
                        }
                    )

                # Viewer certificate
                cert = dist.get("ViewerCertificate", {})
                viewer_certificate = {
                    "acm_certificate_arn": cert.get("ACMCertificateArn", ""),
                    "ssl_support_method": cert.get("SSLSupportMethod", ""),
                    "minimum_protocol_version": cert.get(
                        "MinimumProtocolVersion", ""
                    ),
                    "cloudfront_default_certificate": cert.get(
                        "CloudFrontDefaultCertificate", False
                    ),
                }

                properties = {
                    "domain_name": domain,
                    "aliases": aliases,
                    "status": dist.get("Status", ""),
                    "enabled": dist.get("Enabled", True),
                    "http_version": dist.get("HttpVersion", ""),
                    "price_class": dist.get("PriceClass", ""),
                    "origins": origins,
                    "viewer_certificate": viewer_certificate,
                    "comment": dist.get("Comment", ""),
                    "default_cache_behavior": {
                        "target_origin_id": dist.get(
                            "DefaultCacheBehavior", {}
                        ).get("TargetOriginId", ""),
                        "viewer_protocol_policy": dist.get(
                            "DefaultCacheBehavior", {}
                        ).get("ViewerProtocolPolicy", ""),
                    },
                    "web_acl_id": dist.get("WebACLId", ""),
                }

                resources.append(
                    DiscoveredResource(
                        service="cloudfront",
                        resource_type="aws_cloudfront_distribution",
                        resource_id=dist_id,
                        arn=dist.get("ARN", ""),
                        name=aliases[0] if aliases else domain,
                        properties=properties,
                        terraform_address=tf_addr,
                    )
                )

        except Exception as exc:
            logger.warning("CloudFront discovery failed: %s", exc)

        return resources

    # ------------------------------------------------------------------
    # Pagination
    # ------------------------------------------------------------------

    @staticmethod
    def _list_all_distributions(client) -> list:
        """Paginate through list_distributions."""
        distributions: list = []
        kwargs: dict = {}
        while True:
            resp = client.list_distributions(**kwargs)
            dist_list = resp.get("DistributionList", {})
            items = dist_list.get("Items", [])
            distributions.extend(items)
            marker = dist_list.get("NextMarker")
            if not dist_list.get("IsTruncated", False) or not marker:
                break
            kwargs["Marker"] = marker
        return distributions
