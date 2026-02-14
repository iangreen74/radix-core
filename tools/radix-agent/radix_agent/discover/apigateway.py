"""Discover API Gateway v1 (REST) and v2 (HTTP/WebSocket) APIs."""

import logging
from typing import List

from ..models import DiscoveredResource
from .base import BaseDiscoverer

logger = logging.getLogger(__name__)


class ApiGatewayDiscoverer(BaseDiscoverer):
    """Discover API Gateway resources.

    Covers both API Gateway v1 (REST APIs via ``get_rest_apis``) and
    API Gateway v2 (HTTP / WebSocket APIs via ``get_apis``).
    """

    service_name = "apigateway"

    def discover(self) -> List[DiscoveredResource]:
        resources: List[DiscoveredResource] = []
        resources.extend(self._discover_v1())
        resources.extend(self._discover_v2())
        return resources

    # ------------------------------------------------------------------
    # REST API (v1)
    # ------------------------------------------------------------------

    def _discover_v1(self) -> List[DiscoveredResource]:
        """Discover API Gateway v1 REST APIs."""
        resources: List[DiscoveredResource] = []
        try:
            client = self._client("apigateway")
            apis = self._list_all_rest_apis(client)

            for api in apis:
                api_id = api["id"]
                api_name = api.get("name", api_id)
                label = self._sanitize_name(api_name)

                endpoint_config = api.get("endpointConfiguration", {})
                endpoint_types = endpoint_config.get("types", [])

                properties = {
                    "name": api_name,
                    "description": api.get("description", ""),
                    "endpoint_configuration": endpoint_types,
                    "created_date": str(api.get("createdDate", "")),
                    "api_key_source": api.get("apiKeySource", ""),
                    "protocol": "REST",
                }

                resources.append(
                    DiscoveredResource(
                        service="apigateway",
                        resource_type="aws_api_gateway_rest_api",
                        resource_id=api_id,
                        name=api_name,
                        tags=self._safe_tags(api.get("tags")),
                        properties=properties,
                        terraform_address=f"aws_api_gateway_rest_api.{label}",
                    )
                )

        except Exception as exc:
            logger.warning("API Gateway v1 discovery failed: %s", exc)

        return resources

    # ------------------------------------------------------------------
    # HTTP / WebSocket API (v2)
    # ------------------------------------------------------------------

    def _discover_v2(self) -> List[DiscoveredResource]:
        """Discover API Gateway v2 HTTP and WebSocket APIs."""
        resources: List[DiscoveredResource] = []
        try:
            client = self._client("apigatewayv2")
            apis = self._list_all_v2_apis(client)

            for api in apis:
                api_id = api["ApiId"]
                api_name = api.get("Name", api_id)
                label = self._sanitize_name(api_name)
                protocol = api.get("ProtocolType", "HTTP")

                endpoint = api.get("ApiEndpoint", "")

                properties = {
                    "name": api_name,
                    "description": api.get("Description", ""),
                    "protocol_type": protocol,
                    "api_endpoint": endpoint,
                    "route_selection_expression": api.get(
                        "RouteSelectionExpression", ""
                    ),
                    "disable_execute_api_endpoint": api.get(
                        "DisableExecuteApiEndpoint", False
                    ),
                }

                resources.append(
                    DiscoveredResource(
                        service="apigateway",
                        resource_type="aws_apigatewayv2_api",
                        resource_id=api_id,
                        name=api_name,
                        tags=api.get("Tags", {}),
                        properties=properties,
                        terraform_address=f"aws_apigatewayv2_api.{label}",
                    )
                )

        except Exception as exc:
            logger.warning("API Gateway v2 discovery failed: %s", exc)

        return resources

    # ------------------------------------------------------------------
    # Pagination helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _list_all_rest_apis(client) -> list:
        """Paginate through get_rest_apis."""
        apis: list = []
        kwargs: dict = {"limit": 500}
        while True:
            resp = client.get_rest_apis(**kwargs)
            apis.extend(resp.get("items", []))
            position = resp.get("position")
            if not position:
                break
            kwargs["position"] = position
        return apis

    @staticmethod
    def _list_all_v2_apis(client) -> list:
        """Paginate through get_apis (v2)."""
        apis: list = []
        kwargs: dict = {"MaxResults": "500"}
        while True:
            resp = client.get_apis(**kwargs)
            apis.extend(resp.get("Items", []))
            token = resp.get("NextToken")
            if not token:
                break
            kwargs["NextToken"] = token
        return apis
