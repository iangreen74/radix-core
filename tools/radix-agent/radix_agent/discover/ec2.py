"""Discover EC2 instances (running) and key pairs."""

import logging
from typing import List

from ..models import DiscoveredResource
from .base import BaseDiscoverer

logger = logging.getLogger(__name__)


class Ec2Discoverer(BaseDiscoverer):
    """Discover EC2 resources.

    Covers:
    - Running instances (``describe_instances`` filtered by running state)
    - Key pairs (``describe_key_pairs``)

    Security groups are intentionally *not* covered here; the
    ``VpcDiscoverer`` handles those so that each SG can be tagged with
    its parent VPC.
    """

    service_name = "ec2"

    def discover(self) -> List[DiscoveredResource]:
        resources: List[DiscoveredResource] = []
        resources.extend(self._discover_instances())
        resources.extend(self._discover_key_pairs())
        return resources

    # ------------------------------------------------------------------
    # Instances
    # ------------------------------------------------------------------

    def _discover_instances(self) -> List[DiscoveredResource]:
        """Discover running EC2 instances."""
        resources: List[DiscoveredResource] = []
        try:
            client = self._client("ec2")
            paginator = client.get_paginator("describe_instances")
            filters = [{"Name": "instance-state-name", "Values": ["running"]}]

            for page in paginator.paginate(Filters=filters):
                for reservation in page.get("Reservations", []):
                    for inst in reservation.get("Instances", []):
                        instance_id = inst["InstanceId"]
                        tags = self._safe_tags(inst.get("Tags"))
                        name = tags.get("Name", instance_id)
                        label = self._sanitize_name(name)

                        # Collect security group ids
                        sg_ids = [
                            sg["GroupId"]
                            for sg in inst.get("SecurityGroups", [])
                        ]

                        properties = {
                            "instance_type": inst.get("InstanceType", ""),
                            "ami": inst.get("ImageId", ""),
                            "state": inst.get("State", {}).get("Name", ""),
                            "private_ip": inst.get("PrivateIpAddress", ""),
                            "public_ip": inst.get("PublicIpAddress", ""),
                            "subnet_id": inst.get("SubnetId", ""),
                            "vpc_id": inst.get("VpcId", ""),
                            "key_name": inst.get("KeyName", ""),
                            "security_group_ids": sg_ids,
                            "iam_instance_profile": (
                                inst.get("IamInstanceProfile", {}).get("Arn", "")
                            ),
                            "launch_time": str(inst.get("LaunchTime", "")),
                            "availability_zone": inst.get("Placement", {}).get(
                                "AvailabilityZone", ""
                            ),
                            "ebs_optimized": inst.get("EbsOptimized", False),
                            "architecture": inst.get("Architecture", ""),
                            "root_device_type": inst.get("RootDeviceType", ""),
                        }

                        resources.append(
                            DiscoveredResource(
                                service="ec2",
                                resource_type="aws_instance",
                                resource_id=instance_id,
                                name=name,
                                tags=tags,
                                properties=properties,
                                terraform_address=f"aws_instance.{label}",
                            )
                        )

        except Exception as exc:
            logger.warning("EC2 instance discovery failed: %s", exc)

        return resources

    # ------------------------------------------------------------------
    # Key pairs
    # ------------------------------------------------------------------

    def _discover_key_pairs(self) -> List[DiscoveredResource]:
        """Discover EC2 key pairs."""
        resources: List[DiscoveredResource] = []
        try:
            client = self._client("ec2")
            resp = client.describe_key_pairs()

            for kp in resp.get("KeyPairs", []):
                kp_name = kp.get("KeyName", "")
                kp_id = kp.get("KeyPairId", kp_name)
                label = self._sanitize_name(kp_name)
                tags = self._safe_tags(kp.get("Tags"))

                resources.append(
                    DiscoveredResource(
                        service="ec2",
                        resource_type="aws_key_pair",
                        resource_id=kp_id,
                        name=kp_name,
                        tags=tags,
                        properties={
                            "key_name": kp_name,
                            "key_type": kp.get("KeyType", ""),
                            "fingerprint": kp.get("KeyFingerprint", ""),
                            "create_time": str(kp.get("CreateTime", "")),
                        },
                        terraform_address=f"aws_key_pair.{label}",
                    )
                )

        except Exception as exc:
            logger.warning("EC2 key pair discovery failed: %s", exc)

        return resources
