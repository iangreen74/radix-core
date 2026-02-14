"""Discover Cognito User Pools."""

import logging
from typing import List

from ..models import DiscoveredResource
from .base import BaseDiscoverer

logger = logging.getLogger(__name__)


class CognitoDiscoverer(BaseDiscoverer):
    """Discover Amazon Cognito User Pools.

    Lists user pools (up to 60 per call), then calls
    ``describe_user_pool`` for each to capture full configuration.
    """

    service_name = "cognito"

    def discover(self) -> List[DiscoveredResource]:
        resources: List[DiscoveredResource] = []
        try:
            client = self._client("cognito-idp")
            pools = self._list_all_user_pools(client)

            for pool_summary in pools:
                pool_id = pool_summary.get("Id", "")
                pool_name = pool_summary.get("Name", pool_id)

                try:
                    detail = client.describe_user_pool(UserPoolId=pool_id)[
                        "UserPool"
                    ]
                except Exception as exc:
                    logger.warning(
                        "Failed to describe user pool %s: %s", pool_id, exc
                    )
                    continue

                label = self._sanitize_name(pool_name)
                arn = detail.get("Arn", "")

                # Password policy
                password_policy = detail.get("Policies", {}).get(
                    "PasswordPolicy", {}
                )

                # MFA configuration
                mfa_config = detail.get("MfaConfiguration", "OFF")

                # Schema attributes
                schema_attrs = []
                for attr in detail.get("SchemaAttributes", []):
                    schema_attrs.append(
                        {
                            "name": attr.get("Name", ""),
                            "attribute_data_type": attr.get(
                                "AttributeDataType", ""
                            ),
                            "mutable": attr.get("Mutable", True),
                            "required": attr.get("Required", False),
                        }
                    )

                properties = {
                    "user_pool_id": pool_id,
                    "name": pool_name,
                    "status": detail.get("Status", ""),
                    "creation_date": str(detail.get("CreationDate", "")),
                    "last_modified_date": str(
                        detail.get("LastModifiedDate", "")
                    ),
                    "estimated_number_of_users": detail.get(
                        "EstimatedNumberOfUsers", 0
                    ),
                    "mfa_configuration": mfa_config,
                    "password_policy": {
                        "minimum_length": password_policy.get(
                            "MinimumLength", 8
                        ),
                        "require_lowercase": password_policy.get(
                            "RequireLowercase", False
                        ),
                        "require_uppercase": password_policy.get(
                            "RequireUppercase", False
                        ),
                        "require_numbers": password_policy.get(
                            "RequireNumbers", False
                        ),
                        "require_symbols": password_policy.get(
                            "RequireSymbols", False
                        ),
                        "temporary_password_validity_days": password_policy.get(
                            "TemporaryPasswordValidityDays", 7
                        ),
                    },
                    "auto_verified_attributes": detail.get(
                        "AutoVerifiedAttributes", []
                    ),
                    "username_attributes": detail.get(
                        "UsernameAttributes", []
                    ),
                    "schema_attributes": schema_attrs,
                    "deletion_protection": detail.get(
                        "DeletionProtection", "INACTIVE"
                    ),
                    "domain": detail.get("Domain", ""),
                    "custom_domain": detail.get("CustomDomain", ""),
                    "email_configuration": {
                        "email_sending_account": detail.get(
                            "EmailConfiguration", {}
                        ).get("EmailSendingAccount", ""),
                        "source_arn": detail.get(
                            "EmailConfiguration", {}
                        ).get("SourceArn", ""),
                    },
                }

                resources.append(
                    DiscoveredResource(
                        service="cognito",
                        resource_type="aws_cognito_user_pool",
                        resource_id=pool_id,
                        arn=arn,
                        name=pool_name,
                        properties=properties,
                        terraform_address=f"aws_cognito_user_pool.{label}",
                    )
                )

        except Exception as exc:
            logger.warning("Cognito discovery failed: %s", exc)

        return resources

    # ------------------------------------------------------------------
    # Pagination
    # ------------------------------------------------------------------

    @staticmethod
    def _list_all_user_pools(client) -> list:
        """Paginate through list_user_pools (max 60 per call)."""
        pools: list = []
        kwargs = {"MaxResults": 60}
        while True:
            resp = client.list_user_pools(**kwargs)
            pools.extend(resp.get("UserPools", []))
            token = resp.get("NextToken")
            if not token:
                break
            kwargs["NextToken"] = token
        return pools
