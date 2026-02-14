"""Discover Elastic Load Balancing v2 resources (ALB / NLB / GWLB)."""

import logging
from typing import List

from ..models import DiscoveredResource
from .base import BaseDiscoverer

logger = logging.getLogger(__name__)


class ElbDiscoverer(BaseDiscoverer):
    """Discover ELBv2 resources.

    Covers:
    - Load Balancers (ALB, NLB, GWLB)
    - Target Groups
    - Listeners (attached to each load balancer)
    """

    service_name = "elb"

    def discover(self) -> List[DiscoveredResource]:
        resources: List[DiscoveredResource] = []
        try:
            client = self._client("elbv2")

            lbs = self._list_all_load_balancers(client)
            for lb in lbs:
                lb_arn = lb["LoadBalancerArn"]
                lb_name = lb.get("LoadBalancerName", "")
                label = self._sanitize_name(lb_name)

                az_info = [
                    {
                        "zone_name": az.get("ZoneName", ""),
                        "subnet_id": az.get("SubnetId", ""),
                    }
                    for az in lb.get("AvailabilityZones", [])
                ]

                resources.append(
                    DiscoveredResource(
                        service="elb",
                        resource_type="aws_lb",
                        resource_id=lb_arn,
                        arn=lb_arn,
                        name=lb_name,
                        properties={
                            "name": lb_name,
                            "dns_name": lb.get("DNSName", ""),
                            "type": lb.get("Type", ""),
                            "scheme": lb.get("Scheme", ""),
                            "state": lb.get("State", {}).get("Code", ""),
                            "vpc_id": lb.get("VpcId", ""),
                            "security_groups": lb.get("SecurityGroups", []),
                            "availability_zones": az_info,
                            "ip_address_type": lb.get("IpAddressType", ""),
                            "created_time": str(lb.get("CreatedTime", "")),
                        },
                        terraform_address=f"aws_lb.{label}",
                    )
                )

                # Listeners for this LB
                resources.extend(
                    self._discover_listeners(client, lb_arn, label)
                )

            # Target Groups (independent of LBs)
            resources.extend(self._discover_target_groups(client))

        except Exception as exc:
            logger.warning("ELB discovery failed: %s", exc)

        return resources

    # ------------------------------------------------------------------
    # Listeners
    # ------------------------------------------------------------------

    def _discover_listeners(
        self, client, lb_arn: str, lb_label: str
    ) -> List[DiscoveredResource]:
        resources: List[DiscoveredResource] = []
        try:
            listeners = self._list_all_listeners(client, lb_arn)
            for listener in listeners:
                listener_arn = listener["ListenerArn"]
                port = listener.get("Port", 0)
                protocol = listener.get("Protocol", "")
                label = f"{lb_label}_{protocol}_{port}"

                default_actions = []
                for action in listener.get("DefaultActions", []):
                    default_actions.append(
                        {
                            "type": action.get("Type", ""),
                            "target_group_arn": action.get("TargetGroupArn", ""),
                            "order": action.get("Order", 0),
                        }
                    )

                resources.append(
                    DiscoveredResource(
                        service="elb",
                        resource_type="aws_lb_listener",
                        resource_id=listener_arn,
                        arn=listener_arn,
                        name=f"{lb_label}-{protocol}-{port}",
                        properties={
                            "load_balancer_arn": lb_arn,
                            "port": port,
                            "protocol": protocol,
                            "ssl_policy": listener.get("SslPolicy", ""),
                            "certificate_arn": (
                                listener.get("Certificates", [{}])[0].get(
                                    "CertificateArn", ""
                                )
                                if listener.get("Certificates")
                                else ""
                            ),
                            "default_actions": default_actions,
                        },
                        terraform_address=f"aws_lb_listener.{label}",
                    )
                )
        except Exception as exc:
            logger.warning(
                "Listener discovery failed for LB %s: %s", lb_arn, exc
            )
        return resources

    # ------------------------------------------------------------------
    # Target Groups
    # ------------------------------------------------------------------

    def _discover_target_groups(self, client) -> List[DiscoveredResource]:
        resources: List[DiscoveredResource] = []
        try:
            tgs = self._list_all_target_groups(client)
            for tg in tgs:
                tg_arn = tg["TargetGroupArn"]
                tg_name = tg.get("TargetGroupName", "")
                label = self._sanitize_name(tg_name)

                health_check = {
                    "protocol": tg.get("HealthCheckProtocol", ""),
                    "port": tg.get("HealthCheckPort", ""),
                    "path": tg.get("HealthCheckPath", ""),
                    "interval": tg.get("HealthCheckIntervalSeconds", 0),
                    "timeout": tg.get("HealthCheckTimeoutSeconds", 0),
                    "healthy_threshold": tg.get("HealthyThresholdCount", 0),
                    "unhealthy_threshold": tg.get("UnhealthyThresholdCount", 0),
                    "matcher": tg.get("Matcher", {}).get("HttpCode", ""),
                }

                resources.append(
                    DiscoveredResource(
                        service="elb",
                        resource_type="aws_lb_target_group",
                        resource_id=tg_arn,
                        arn=tg_arn,
                        name=tg_name,
                        properties={
                            "name": tg_name,
                            "protocol": tg.get("Protocol", ""),
                            "port": tg.get("Port", 0),
                            "vpc_id": tg.get("VpcId", ""),
                            "target_type": tg.get("TargetType", ""),
                            "load_balancer_arns": tg.get(
                                "LoadBalancerArns", []
                            ),
                            "health_check": health_check,
                            "ip_address_type": tg.get("IpAddressType", ""),
                        },
                        terraform_address=f"aws_lb_target_group.{label}",
                    )
                )
        except Exception as exc:
            logger.warning("Target Group discovery failed: %s", exc)
        return resources

    # ------------------------------------------------------------------
    # Pagination helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _list_all_load_balancers(client) -> list:
        """Paginate through describe_load_balancers."""
        items: list = []
        paginator = client.get_paginator("describe_load_balancers")
        for page in paginator.paginate():
            items.extend(page.get("LoadBalancers", []))
        return items

    @staticmethod
    def _list_all_listeners(client, lb_arn: str) -> list:
        """Paginate through describe_listeners for a load balancer."""
        items: list = []
        paginator = client.get_paginator("describe_listeners")
        for page in paginator.paginate(LoadBalancerArn=lb_arn):
            items.extend(page.get("Listeners", []))
        return items

    @staticmethod
    def _list_all_target_groups(client) -> list:
        """Paginate through describe_target_groups."""
        items: list = []
        paginator = client.get_paginator("describe_target_groups")
        for page in paginator.paginate():
            items.extend(page.get("TargetGroups", []))
        return items
