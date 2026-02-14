"""Discover VPC resources: VPCs, subnets, IGWs, NAT GWs, route tables, EIPs, security groups."""

import logging
from typing import Dict, List

from ..models import DiscoveredResource
from .base import BaseDiscoverer

logger = logging.getLogger(__name__)


class VpcDiscoverer(BaseDiscoverer):
    """Discover VPC networking resources.

    Covers:
    - VPCs
    - Subnets
    - Internet Gateways
    - NAT Gateways
    - Route Tables
    - Elastic IPs (EIPs)
    - Security Groups

    Each resource is tagged with its parent VPC ID in ``properties``
    so that downstream Terraform generation can group resources correctly.
    """

    service_name = "vpc"

    def discover(self) -> List[DiscoveredResource]:
        resources: List[DiscoveredResource] = []
        try:
            client = self._client("ec2")

            # Build a quick VPC-name lookup for labelling
            vpcs = self._describe_all(client, "describe_vpcs", "Vpcs")
            vpc_names: Dict[str, str] = {}
            for vpc in vpcs:
                vpc_id = vpc["VpcId"]
                tags = self._safe_tags(vpc.get("Tags"))
                vpc_names[vpc_id] = tags.get("Name", vpc_id)

            resources.extend(self._discover_vpcs(vpcs, vpc_names))
            resources.extend(self._discover_subnets(client, vpc_names))
            resources.extend(self._discover_igws(client, vpc_names))
            resources.extend(self._discover_nat_gateways(client, vpc_names))
            resources.extend(self._discover_route_tables(client, vpc_names))
            resources.extend(self._discover_eips(client, vpc_names))
            resources.extend(self._discover_security_groups(client, vpc_names))

        except Exception as exc:
            logger.warning("VPC discovery failed: %s", exc)

        return resources

    # ------------------------------------------------------------------
    # VPCs
    # ------------------------------------------------------------------

    def _discover_vpcs(
        self, vpcs: list, vpc_names: Dict[str, str]
    ) -> List[DiscoveredResource]:
        resources: List[DiscoveredResource] = []
        for vpc in vpcs:
            vpc_id = vpc["VpcId"]
            tags = self._safe_tags(vpc.get("Tags"))
            name = vpc_names.get(vpc_id, vpc_id)
            label = self._sanitize_name(name)

            resources.append(
                DiscoveredResource(
                    service="vpc",
                    resource_type="aws_vpc",
                    resource_id=vpc_id,
                    name=name,
                    tags=tags,
                    properties={
                        "cidr_block": vpc.get("CidrBlock", ""),
                        "is_default": vpc.get("IsDefault", False),
                        "state": vpc.get("State", ""),
                        "dhcp_options_id": vpc.get("DhcpOptionsId", ""),
                        "instance_tenancy": vpc.get("InstanceTenancy", ""),
                    },
                    terraform_address=f"aws_vpc.{label}",
                )
            )
        return resources

    # ------------------------------------------------------------------
    # Subnets
    # ------------------------------------------------------------------

    def _discover_subnets(
        self, client, vpc_names: Dict[str, str]
    ) -> List[DiscoveredResource]:
        resources: List[DiscoveredResource] = []
        try:
            subnets = self._describe_all(client, "describe_subnets", "Subnets")
            for subnet in subnets:
                subnet_id = subnet["SubnetId"]
                vpc_id = subnet.get("VpcId", "")
                tags = self._safe_tags(subnet.get("Tags"))
                name = tags.get("Name", subnet_id)
                label = self._sanitize_name(name)

                resources.append(
                    DiscoveredResource(
                        service="vpc",
                        resource_type="aws_subnet",
                        resource_id=subnet_id,
                        name=name,
                        tags=tags,
                        properties={
                            "vpc_id": vpc_id,
                            "vpc_name": vpc_names.get(vpc_id, ""),
                            "cidr_block": subnet.get("CidrBlock", ""),
                            "availability_zone": subnet.get(
                                "AvailabilityZone", ""
                            ),
                            "map_public_ip_on_launch": subnet.get(
                                "MapPublicIpOnLaunch", False
                            ),
                            "state": subnet.get("State", ""),
                            "available_ip_count": subnet.get(
                                "AvailableIpAddressCount", 0
                            ),
                        },
                        terraform_address=f"aws_subnet.{label}",
                    )
                )
        except Exception as exc:
            logger.warning("Subnet discovery failed: %s", exc)
        return resources

    # ------------------------------------------------------------------
    # Internet Gateways
    # ------------------------------------------------------------------

    def _discover_igws(
        self, client, vpc_names: Dict[str, str]
    ) -> List[DiscoveredResource]:
        resources: List[DiscoveredResource] = []
        try:
            igws = self._describe_all(
                client, "describe_internet_gateways", "InternetGateways"
            )
            for igw in igws:
                igw_id = igw["InternetGatewayId"]
                tags = self._safe_tags(igw.get("Tags"))
                name = tags.get("Name", igw_id)
                label = self._sanitize_name(name)

                attachments = igw.get("Attachments", [])
                vpc_id = attachments[0].get("VpcId", "") if attachments else ""

                resources.append(
                    DiscoveredResource(
                        service="vpc",
                        resource_type="aws_internet_gateway",
                        resource_id=igw_id,
                        name=name,
                        tags=tags,
                        properties={
                            "vpc_id": vpc_id,
                            "vpc_name": vpc_names.get(vpc_id, ""),
                            "attachments": [
                                {
                                    "vpc_id": a.get("VpcId", ""),
                                    "state": a.get("State", ""),
                                }
                                for a in attachments
                            ],
                        },
                        terraform_address=f"aws_internet_gateway.{label}",
                    )
                )
        except Exception as exc:
            logger.warning("Internet Gateway discovery failed: %s", exc)
        return resources

    # ------------------------------------------------------------------
    # NAT Gateways
    # ------------------------------------------------------------------

    def _discover_nat_gateways(
        self, client, vpc_names: Dict[str, str]
    ) -> List[DiscoveredResource]:
        resources: List[DiscoveredResource] = []
        try:
            nat_gws = self._describe_all(
                client, "describe_nat_gateways", "NatGateways"
            )
            for nat in nat_gws:
                nat_id = nat["NatGatewayId"]
                vpc_id = nat.get("VpcId", "")
                tags = self._safe_tags(nat.get("Tags"))
                name = tags.get("Name", nat_id)
                label = self._sanitize_name(name)

                # Only include active/available NAT gateways
                state = nat.get("State", "")
                if state in ("deleted", "deleting", "failed"):
                    continue

                addresses = nat.get("NatGatewayAddresses", [])
                eip_alloc = addresses[0].get("AllocationId", "") if addresses else ""
                public_ip = addresses[0].get("PublicIp", "") if addresses else ""

                resources.append(
                    DiscoveredResource(
                        service="vpc",
                        resource_type="aws_nat_gateway",
                        resource_id=nat_id,
                        name=name,
                        tags=tags,
                        properties={
                            "vpc_id": vpc_id,
                            "vpc_name": vpc_names.get(vpc_id, ""),
                            "subnet_id": nat.get("SubnetId", ""),
                            "allocation_id": eip_alloc,
                            "public_ip": public_ip,
                            "state": state,
                            "connectivity_type": nat.get(
                                "ConnectivityType", "public"
                            ),
                        },
                        terraform_address=f"aws_nat_gateway.{label}",
                    )
                )
        except Exception as exc:
            logger.warning("NAT Gateway discovery failed: %s", exc)
        return resources

    # ------------------------------------------------------------------
    # Route Tables
    # ------------------------------------------------------------------

    def _discover_route_tables(
        self, client, vpc_names: Dict[str, str]
    ) -> List[DiscoveredResource]:
        resources: List[DiscoveredResource] = []
        try:
            rts = self._describe_all(
                client, "describe_route_tables", "RouteTables"
            )
            for rt in rts:
                rt_id = rt["RouteTableId"]
                vpc_id = rt.get("VpcId", "")
                tags = self._safe_tags(rt.get("Tags"))
                name = tags.get("Name", rt_id)
                label = self._sanitize_name(name)

                associations = rt.get("Associations", [])
                is_main = any(a.get("Main", False) for a in associations)
                subnet_assocs = [
                    a.get("SubnetId", "")
                    for a in associations
                    if a.get("SubnetId")
                ]

                routes = []
                for route in rt.get("Routes", []):
                    routes.append(
                        {
                            "destination_cidr_block": route.get(
                                "DestinationCidrBlock", ""
                            ),
                            "destination_ipv6_cidr_block": route.get(
                                "DestinationIpv6CidrBlock", ""
                            ),
                            "gateway_id": route.get("GatewayId", ""),
                            "nat_gateway_id": route.get("NatGatewayId", ""),
                            "state": route.get("State", ""),
                            "origin": route.get("Origin", ""),
                        }
                    )

                resources.append(
                    DiscoveredResource(
                        service="vpc",
                        resource_type="aws_route_table",
                        resource_id=rt_id,
                        name=name,
                        tags=tags,
                        properties={
                            "vpc_id": vpc_id,
                            "vpc_name": vpc_names.get(vpc_id, ""),
                            "is_main": is_main,
                            "subnet_associations": subnet_assocs,
                            "routes": routes,
                        },
                        terraform_address=f"aws_route_table.{label}",
                    )
                )
        except Exception as exc:
            logger.warning("Route Table discovery failed: %s", exc)
        return resources

    # ------------------------------------------------------------------
    # Elastic IPs
    # ------------------------------------------------------------------

    def _discover_eips(
        self, client, vpc_names: Dict[str, str]
    ) -> List[DiscoveredResource]:
        resources: List[DiscoveredResource] = []
        try:
            resp = client.describe_addresses()
            for addr in resp.get("Addresses", []):
                alloc_id = addr.get("AllocationId", "")
                tags = self._safe_tags(addr.get("Tags"))
                name = tags.get("Name", alloc_id)
                label = self._sanitize_name(name)

                vpc_id = ""
                if addr.get("NetworkInterfaceId"):
                    # Attempt to resolve VPC from association
                    vpc_id = addr.get("NetworkBorderGroup", "")

                resources.append(
                    DiscoveredResource(
                        service="vpc",
                        resource_type="aws_eip",
                        resource_id=alloc_id,
                        name=name,
                        tags=tags,
                        properties={
                            "public_ip": addr.get("PublicIp", ""),
                            "allocation_id": alloc_id,
                            "association_id": addr.get("AssociationId", ""),
                            "instance_id": addr.get("InstanceId", ""),
                            "network_interface_id": addr.get(
                                "NetworkInterfaceId", ""
                            ),
                            "domain": addr.get("Domain", ""),
                            "vpc_id": vpc_id,
                        },
                        terraform_address=f"aws_eip.{label}",
                    )
                )
        except Exception as exc:
            logger.warning("EIP discovery failed: %s", exc)
        return resources

    # ------------------------------------------------------------------
    # Security Groups
    # ------------------------------------------------------------------

    def _discover_security_groups(
        self, client, vpc_names: Dict[str, str]
    ) -> List[DiscoveredResource]:
        resources: List[DiscoveredResource] = []
        try:
            sgs = self._describe_all(
                client, "describe_security_groups", "SecurityGroups"
            )
            for sg in sgs:
                sg_id = sg["GroupId"]
                vpc_id = sg.get("VpcId", "")
                tags = self._safe_tags(sg.get("Tags"))
                group_name = sg.get("GroupName", sg_id)
                name = tags.get("Name", group_name)
                label = self._sanitize_name(name)

                # Summarize ingress/egress rules
                ingress_rules = []
                for rule in sg.get("IpPermissions", []):
                    ingress_rules.append(self._summarize_rule(rule))
                egress_rules = []
                for rule in sg.get("IpPermissionsEgress", []):
                    egress_rules.append(self._summarize_rule(rule))

                resources.append(
                    DiscoveredResource(
                        service="vpc",
                        resource_type="aws_security_group",
                        resource_id=sg_id,
                        name=name,
                        tags=tags,
                        properties={
                            "vpc_id": vpc_id,
                            "vpc_name": vpc_names.get(vpc_id, ""),
                            "group_name": group_name,
                            "description": sg.get("Description", ""),
                            "ingress_rule_count": len(ingress_rules),
                            "egress_rule_count": len(egress_rules),
                            "ingress_rules": ingress_rules,
                            "egress_rules": egress_rules,
                        },
                        terraform_address=f"aws_security_group.{label}",
                    )
                )
        except Exception as exc:
            logger.warning("Security Group discovery failed: %s", exc)
        return resources

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _describe_all(client, method_name: str, result_key: str) -> list:
        """Generic paginator wrapper for EC2 describe calls."""
        items: list = []
        paginator = client.get_paginator(method_name)
        for page in paginator.paginate():
            items.extend(page.get(result_key, []))
        return items

    @staticmethod
    def _summarize_rule(rule: dict) -> dict:
        """Summarize an SG rule for storage."""
        return {
            "protocol": rule.get("IpProtocol", ""),
            "from_port": rule.get("FromPort", -1),
            "to_port": rule.get("ToPort", -1),
            "cidr_blocks": [r.get("CidrIp", "") for r in rule.get("IpRanges", [])],
            "ipv6_cidr_blocks": [
                r.get("CidrIpv6", "") for r in rule.get("Ipv6Ranges", [])
            ],
            "security_groups": [
                p.get("GroupId", "")
                for p in rule.get("UserIdGroupPairs", [])
            ],
            "prefix_list_ids": [
                p.get("PrefixListId", "") for p in rule.get("PrefixListIds", [])
            ],
        }
