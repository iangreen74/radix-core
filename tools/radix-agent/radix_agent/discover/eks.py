"""Discover EKS clusters, node groups, and add-ons."""

import logging
from typing import List

from ..models import DiscoveredResource
from .base import BaseDiscoverer

logger = logging.getLogger(__name__)


class EksDiscoverer(BaseDiscoverer):
    """Discover Amazon EKS resources.

    For each cluster the discoverer retrieves:
    - The cluster itself (``describe_cluster``)
    - All managed node groups (``list_nodegroups`` / ``describe_nodegroup``)
    - All installed add-ons (``list_addons``)
    """

    service_name = "eks"

    def discover(self) -> List[DiscoveredResource]:
        resources: List[DiscoveredResource] = []
        try:
            client = self._client("eks")
            cluster_names = self._list_all_clusters(client)

            for cluster_name in cluster_names:
                try:
                    cluster = client.describe_cluster(name=cluster_name)["cluster"]
                except Exception as exc:
                    logger.warning(
                        "Failed to describe EKS cluster %s: %s", cluster_name, exc
                    )
                    continue

                label = self._sanitize_name(cluster_name)

                # --- Cluster resource ---
                resources.append(
                    DiscoveredResource(
                        service="eks",
                        resource_type="aws_eks_cluster",
                        resource_id=cluster_name,
                        arn=cluster.get("arn", ""),
                        name=cluster_name,
                        tags=cluster.get("tags", {}),
                        properties={
                            "name": cluster_name,
                            "version": cluster.get("version", ""),
                            "status": cluster.get("status", ""),
                            "platform_version": cluster.get("platformVersion", ""),
                            "endpoint": cluster.get("endpoint", ""),
                            "role_arn": cluster.get("roleArn", ""),
                            "vpc_config": {
                                "subnet_ids": cluster.get("resourcesVpcConfig", {}).get(
                                    "subnetIds", []
                                ),
                                "security_group_ids": cluster.get(
                                    "resourcesVpcConfig", {}
                                ).get("securityGroupIds", []),
                                "endpoint_public_access": cluster.get(
                                    "resourcesVpcConfig", {}
                                ).get("endpointPublicAccess", True),
                                "endpoint_private_access": cluster.get(
                                    "resourcesVpcConfig", {}
                                ).get("endpointPrivateAccess", False),
                            },
                            "kubernetes_network_config": {
                                "service_ipv4_cidr": cluster.get(
                                    "kubernetesNetworkConfig", {}
                                ).get("serviceIpv4Cidr", ""),
                            },
                            "logging": cluster.get("logging", {}),
                            "encryption_config": cluster.get("encryptionConfig", []),
                        },
                        terraform_address=f"aws_eks_cluster.{label}",
                    )
                )

                # --- Node groups ---
                resources.extend(
                    self._discover_nodegroups(client, cluster_name, label)
                )

                # --- Add-ons ---
                resources.extend(
                    self._discover_addons(client, cluster_name, label)
                )

        except Exception as exc:
            logger.warning("EKS discovery failed: %s", exc)

        return resources

    # ------------------------------------------------------------------
    # Node groups
    # ------------------------------------------------------------------

    def _discover_nodegroups(
        self, client, cluster_name: str, cluster_label: str
    ) -> List[DiscoveredResource]:
        resources: List[DiscoveredResource] = []
        try:
            ng_names = self._list_all_nodegroups(client, cluster_name)
            for ng_name in ng_names:
                try:
                    ng = client.describe_nodegroup(
                        clusterName=cluster_name, nodegroupName=ng_name
                    )["nodegroup"]
                except Exception as exc:
                    logger.warning(
                        "Failed to describe nodegroup %s/%s: %s",
                        cluster_name,
                        ng_name,
                        exc,
                    )
                    continue

                ng_label = self._sanitize_name(ng_name)
                scaling = ng.get("scalingConfig", {})

                resources.append(
                    DiscoveredResource(
                        service="eks",
                        resource_type="aws_eks_node_group",
                        resource_id=f"{cluster_name}:{ng_name}",
                        arn=ng.get("nodegroupArn", ""),
                        name=ng_name,
                        tags=ng.get("tags", {}),
                        properties={
                            "cluster_name": cluster_name,
                            "node_group_name": ng_name,
                            "node_role_arn": ng.get("nodeRole", ""),
                            "subnet_ids": ng.get("subnets", []),
                            "instance_types": ng.get("instanceTypes", []),
                            "ami_type": ng.get("amiType", ""),
                            "capacity_type": ng.get("capacityType", ""),
                            "disk_size": ng.get("diskSize", 0),
                            "scaling_config": {
                                "desired_size": scaling.get("desiredSize", 0),
                                "max_size": scaling.get("maxSize", 0),
                                "min_size": scaling.get("minSize", 0),
                            },
                            "status": ng.get("status", ""),
                            "labels": ng.get("labels", {}),
                            "taints": ng.get("taints", []),
                            "launch_template": ng.get("launchTemplate", {}),
                        },
                        terraform_address=(
                            f"aws_eks_node_group.{cluster_label}_{ng_label}"
                        ),
                    )
                )
        except Exception as exc:
            logger.warning(
                "Failed to list nodegroups for %s: %s", cluster_name, exc
            )
        return resources

    # ------------------------------------------------------------------
    # Add-ons
    # ------------------------------------------------------------------

    def _discover_addons(
        self, client, cluster_name: str, cluster_label: str
    ) -> List[DiscoveredResource]:
        resources: List[DiscoveredResource] = []
        try:
            addon_names = self._list_all_addons(client, cluster_name)
            for addon_name in addon_names:
                addon_label = self._sanitize_name(addon_name)
                resources.append(
                    DiscoveredResource(
                        service="eks",
                        resource_type="aws_eks_addon",
                        resource_id=f"{cluster_name}:{addon_name}",
                        name=addon_name,
                        properties={
                            "cluster_name": cluster_name,
                            "addon_name": addon_name,
                        },
                        terraform_address=(
                            f"aws_eks_addon.{cluster_label}_{addon_label}"
                        ),
                    )
                )
        except Exception as exc:
            logger.warning(
                "Failed to list addons for %s: %s", cluster_name, exc
            )
        return resources

    # ------------------------------------------------------------------
    # Pagination helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _list_all_clusters(client) -> list:
        """Paginate through list_clusters."""
        names: list = []
        paginator = client.get_paginator("list_clusters")
        for page in paginator.paginate():
            names.extend(page.get("clusters", []))
        return names

    @staticmethod
    def _list_all_nodegroups(client, cluster_name: str) -> list:
        """Paginate through list_nodegroups for a cluster."""
        names: list = []
        paginator = client.get_paginator("list_nodegroups")
        for page in paginator.paginate(clusterName=cluster_name):
            names.extend(page.get("nodegroups", []))
        return names

    @staticmethod
    def _list_all_addons(client, cluster_name: str) -> list:
        """Paginate through list_addons for a cluster."""
        names: list = []
        paginator = client.get_paginator("list_addons")
        for page in paginator.paginate(clusterName=cluster_name):
            names.extend(page.get("addons", []))
        return names
