"""Discover SSM Parameter Store parameters under the /radix/ prefix."""

import logging
from typing import List

from ..models import DiscoveredResource
from .base import BaseDiscoverer

logger = logging.getLogger(__name__)


class SsmDiscoverer(BaseDiscoverer):
    """Discover SSM parameters under ``/radix/`` (recursive, paginated).

    Uses ``get_parameters_by_path`` with ``Recursive=True`` to capture the
    full parameter tree.  Only metadata is collected — actual ``Value``
    fields are intentionally not stored to avoid leaking secrets.
    """

    service_name = "ssm"

    def discover(self) -> List[DiscoveredResource]:
        resources: List[DiscoveredResource] = []
        try:
            client = self._client("ssm")
            parameters = self._get_all_parameters(client, "/radix/")

            for param in parameters:
                param_name = param.get("Name", "")
                param_arn = param.get("ARN", "")
                label = self._sanitize_name(param_name)

                properties = {
                    "name": param_name,
                    "type": param.get("Type", ""),
                    "version": param.get("Version", 0),
                    "tier": param.get("Tier", "Standard"),
                    "data_type": param.get("DataType", "text"),
                    "last_modified_date": str(
                        param.get("LastModifiedDate", "")
                    ),
                    "last_modified_user": param.get(
                        "LastModifiedUser", ""
                    ),
                }

                resources.append(
                    DiscoveredResource(
                        service="ssm",
                        resource_type="aws_ssm_parameter",
                        resource_id=param_name,
                        arn=param_arn,
                        name=param_name,
                        properties=properties,
                        terraform_address=f"aws_ssm_parameter.{label}",
                    )
                )

        except Exception as exc:
            logger.warning("SSM discovery failed: %s", exc)

        return resources

    # ------------------------------------------------------------------
    # Pagination
    # ------------------------------------------------------------------

    @staticmethod
    def _get_all_parameters(client, path: str) -> list:
        """Paginate through get_parameters_by_path."""
        params: list = []
        kwargs = {
            "Path": path,
            "Recursive": True,
            "WithDecryption": False,
            "MaxResults": 10,
        }
        while True:
            resp = client.get_parameters_by_path(**kwargs)
            params.extend(resp.get("Parameters", []))
            token = resp.get("NextToken")
            if not token:
                break
            kwargs["NextToken"] = token
        return params
