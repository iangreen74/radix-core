"""Discover Route 53 hosted zones and DNS records."""

import logging
from typing import List

from ..models import DiscoveredResource
from .base import BaseDiscoverer

logger = logging.getLogger(__name__)


class Route53Discoverer(BaseDiscoverer):
    """Discover Route 53 hosted zones and their resource record sets.

    Route 53 is a global service; the region parameter is accepted but
    the client always targets the global endpoint.

    NS and SOA records at the zone apex are skipped because they are
    automatically created by AWS and cannot be imported into Terraform.
    """

    service_name = "route53"

    def discover(self) -> List[DiscoveredResource]:
        resources: List[DiscoveredResource] = []
        try:
            client = self._client("route53")
            zones = self._list_all_zones(client)

            for zone in zones:
                zone_id = zone["Id"].split("/")[-1]
                zone_name = zone["Name"].rstrip(".")
                tf_zone_name = self._sanitize_name(zone_name)

                resources.append(
                    DiscoveredResource(
                        service="route53",
                        resource_type="aws_route53_zone",
                        resource_id=zone_id,
                        arn=f"arn:aws:route53:::hostedzone/{zone_id}",
                        name=zone_name,
                        tags={},
                        properties={
                            "name": zone_name,
                            "private_zone": zone.get("Config", {}).get(
                                "PrivateZone", False
                            ),
                            "record_count": zone.get("ResourceRecordSetCount", 0),
                            "comment": zone.get("Config", {}).get("Comment", ""),
                        },
                        terraform_address=f"aws_route53_zone.{tf_zone_name}",
                    )
                )

                records = self._list_all_records(client, zone_id)
                for record in records:
                    rr_name = record["Name"].rstrip(".")
                    rr_type = record["Type"]

                    # Skip apex NS/SOA — AWS manages these automatically
                    if rr_name == zone_name and rr_type in ("NS", "SOA"):
                        continue

                    # Build a readable Terraform address
                    record_label = self._sanitize_name(rr_name)
                    tf_addr = f"aws_route53_record.{record_label}_{rr_type}"

                    record_id = f"{zone_id}_{rr_name}_{rr_type}"

                    properties = {
                        "zone_id": zone_id,
                        "name": rr_name,
                        "type": rr_type,
                    }

                    if "AliasTarget" in record:
                        alias = record["AliasTarget"]
                        properties["alias"] = {
                            "name": alias.get("DNSName", ""),
                            "zone_id": alias.get("HostedZoneId", ""),
                            "evaluate_target_health": alias.get(
                                "EvaluateTargetHealth", False
                            ),
                        }
                    else:
                        properties["ttl"] = record.get("TTL", 300)
                        properties["records"] = [
                            rr.get("Value", "")
                            for rr in record.get("ResourceRecords", [])
                        ]

                    if "SetIdentifier" in record:
                        properties["set_identifier"] = record["SetIdentifier"]
                    if "Weight" in record:
                        properties["weighted_routing_policy"] = {
                            "weight": record["Weight"]
                        }

                    resources.append(
                        DiscoveredResource(
                            service="route53",
                            resource_type="aws_route53_record",
                            resource_id=record_id,
                            name=rr_name,
                            properties=properties,
                            terraform_address=tf_addr,
                        )
                    )

        except Exception as exc:
            logger.warning("Route53 discovery failed: %s", exc)

        return resources

    # ------------------------------------------------------------------
    # Pagination helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _list_all_zones(client) -> list:
        """Paginate through list_hosted_zones."""
        zones: list = []
        paginator = client.get_paginator("list_hosted_zones")
        for page in paginator.paginate():
            zones.extend(page.get("HostedZones", []))
        return zones

    @staticmethod
    def _list_all_records(client, zone_id: str) -> list:
        """Paginate through list_resource_record_sets for a single zone."""
        records: list = []
        kwargs = {"HostedZoneId": zone_id, "MaxItems": "300"}
        while True:
            resp = client.list_resource_record_sets(**kwargs)
            records.extend(resp.get("ResourceRecordSets", []))
            if not resp.get("IsTruncated", False):
                break
            kwargs["StartRecordName"] = resp["NextRecordName"]
            kwargs["StartRecordType"] = resp["NextRecordType"]
            if "NextRecordIdentifier" in resp:
                kwargs["StartRecordIdentifier"] = resp["NextRecordIdentifier"]
        return records
