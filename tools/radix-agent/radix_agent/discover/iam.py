"""Discover IAM roles, instance profiles, and OIDC providers."""

import logging
from typing import List

from ..models import DiscoveredResource
from .base import BaseDiscoverer

logger = logging.getLogger(__name__)

# Only discover roles whose name (or path) matches these prefixes.
_ROLE_PREFIXES = ("radix", "eks", "Radix", "EKS")


class IamDiscoverer(BaseDiscoverer):
    """Discover IAM resources.

    IAM is a global service — the ``region`` parameter is accepted for
    interface consistency but ignored when creating the IAM client.

    Only roles prefixed with ``radix`` or ``eks`` (case-insensitive) are
    included to avoid pulling in hundreds of AWS-managed / third-party roles.
    """

    service_name = "iam"

    def discover(self) -> List[DiscoveredResource]:
        resources: List[DiscoveredResource] = []
        resources.extend(self._discover_roles())
        resources.extend(self._discover_instance_profiles())
        resources.extend(self._discover_oidc_providers())
        return resources

    # ------------------------------------------------------------------
    # Roles
    # ------------------------------------------------------------------

    def _discover_roles(self) -> List[DiscoveredResource]:
        resources: List[DiscoveredResource] = []
        try:
            client = self.session.client("iam")
            roles = self._list_all_roles(client)

            for role in roles:
                role_name = role["RoleName"]
                # Filter to radix/eks-related roles
                if not role_name.startswith(_ROLE_PREFIXES):
                    continue

                label = self._sanitize_name(role_name)
                arn = role.get("Arn", "")

                resources.append(
                    DiscoveredResource(
                        service="iam",
                        resource_type="aws_iam_role",
                        resource_id=role_name,
                        arn=arn,
                        name=role_name,
                        tags=self._safe_tags(role.get("Tags")),
                        properties={
                            "role_name": role_name,
                            "path": role.get("Path", "/"),
                            "description": role.get("Description", ""),
                            "max_session_duration": role.get(
                                "MaxSessionDuration", 3600
                            ),
                            "create_date": str(role.get("CreateDate", "")),
                            "assume_role_policy_document": role.get(
                                "AssumeRolePolicyDocument", {}
                            ),
                        },
                        terraform_address=f"aws_iam_role.{label}",
                    )
                )

        except Exception as exc:
            logger.warning("IAM role discovery failed: %s", exc)

        return resources

    # ------------------------------------------------------------------
    # Instance Profiles
    # ------------------------------------------------------------------

    def _discover_instance_profiles(self) -> List[DiscoveredResource]:
        resources: List[DiscoveredResource] = []
        try:
            client = self.session.client("iam")
            profiles = self._list_all_instance_profiles(client)

            for profile in profiles:
                profile_name = profile["InstanceProfileName"]
                # Filter to radix/eks-related profiles
                if not profile_name.startswith(_ROLE_PREFIXES):
                    continue

                label = self._sanitize_name(profile_name)
                arn = profile.get("Arn", "")
                role_names = [
                    r["RoleName"] for r in profile.get("Roles", [])
                ]

                resources.append(
                    DiscoveredResource(
                        service="iam",
                        resource_type="aws_iam_instance_profile",
                        resource_id=profile_name,
                        arn=arn,
                        name=profile_name,
                        tags=self._safe_tags(profile.get("Tags")),
                        properties={
                            "instance_profile_name": profile_name,
                            "path": profile.get("Path", "/"),
                            "roles": role_names,
                            "create_date": str(profile.get("CreateDate", "")),
                        },
                        terraform_address=f"aws_iam_instance_profile.{label}",
                    )
                )

        except Exception as exc:
            logger.warning("IAM instance profile discovery failed: %s", exc)

        return resources

    # ------------------------------------------------------------------
    # OIDC Providers
    # ------------------------------------------------------------------

    def _discover_oidc_providers(self) -> List[DiscoveredResource]:
        resources: List[DiscoveredResource] = []
        try:
            client = self.session.client("iam")
            resp = client.list_open_id_connect_providers()

            for provider in resp.get("OpenIDConnectProviderList", []):
                arn = provider["Arn"]
                # Extract the URL from the ARN (last segment after /)
                provider_url = arn.split("/", 1)[-1] if "/" in arn else arn
                label = self._sanitize_name(provider_url)

                try:
                    detail = client.get_open_id_connect_provider(
                        OpenIDConnectProviderArn=arn
                    )
                    properties = {
                        "url": detail.get("Url", provider_url),
                        "client_id_list": detail.get("ClientIDList", []),
                        "thumbprint_list": detail.get("ThumbprintList", []),
                        "create_date": str(detail.get("CreateDate", "")),
                    }
                    tags = self._safe_tags(detail.get("Tags"))
                except Exception as exc:
                    logger.warning(
                        "Failed to describe OIDC provider %s: %s", arn, exc
                    )
                    properties = {"url": provider_url}
                    tags = {}

                resources.append(
                    DiscoveredResource(
                        service="iam",
                        resource_type="aws_iam_openid_connect_provider",
                        resource_id=arn,
                        arn=arn,
                        name=provider_url,
                        tags=tags,
                        properties=properties,
                        terraform_address=(
                            f"aws_iam_openid_connect_provider.{label}"
                        ),
                    )
                )

        except Exception as exc:
            logger.warning("IAM OIDC provider discovery failed: %s", exc)

        return resources

    # ------------------------------------------------------------------
    # Pagination helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _list_all_roles(client) -> list:
        """Paginate through list_roles."""
        roles: list = []
        paginator = client.get_paginator("list_roles")
        for page in paginator.paginate():
            roles.extend(page.get("Roles", []))
        return roles

    @staticmethod
    def _list_all_instance_profiles(client) -> list:
        """Paginate through list_instance_profiles."""
        profiles: list = []
        paginator = client.get_paginator("list_instance_profiles")
        for page in paginator.paginate():
            profiles.extend(page.get("InstanceProfiles", []))
        return profiles
