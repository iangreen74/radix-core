"""AWS resource discovery orchestrator.

Iterates through all registered discoverers, collects resources, writes
per-service JSON output files, and returns a complete ``Inventory``.

Usage from the CLI::

    from radix_agent.discover import run_discovery
    inventory = run_discovery(session, region, ["all"], state, "output/")
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Type

import boto3

from ..models import DiscoveredResource, Inventory, ServiceInventory
from ..state import AgentState
from .acm import AcmDiscoverer
from .apigateway import ApiGatewayDiscoverer
from .base import BaseDiscoverer
from .cloudfront import CloudFrontDiscoverer
from .cloudwatch import CloudWatchDiscoverer
from .cognito import CognitoDiscoverer
from .ec2 import Ec2Discoverer
from .ecr import EcrDiscoverer
from .eks import EksDiscoverer
from .elb import ElbDiscoverer
from .iam import IamDiscoverer
from .kms import KmsDiscoverer
from .route53 import Route53Discoverer
from .s3 import S3Discoverer
from .ssm import SsmDiscoverer
from .vpc import VpcDiscoverer

logger = logging.getLogger(__name__)

# Ordered list of all discoverers.  The order determines the discovery
# sequence — services with fewer API calls run first so that partial
# results are available quickly when running interactively.
ALL_DISCOVERERS: List[Type[BaseDiscoverer]] = [
    Route53Discoverer,
    CloudFrontDiscoverer,
    AcmDiscoverer,
    ApiGatewayDiscoverer,
    IamDiscoverer,
    KmsDiscoverer,
    S3Discoverer,
    SsmDiscoverer,
    EcrDiscoverer,
    CognitoDiscoverer,
    VpcDiscoverer,
    Ec2Discoverer,
    ElbDiscoverer,
    EksDiscoverer,
    CloudWatchDiscoverer,
]

# Mapping from service_name to discoverer class for quick lookup.
_DISCOVERER_MAP: Dict[str, Type[BaseDiscoverer]] = {
    cls.service_name: cls for cls in ALL_DISCOVERERS
}


def run_discovery(
    session: boto3.Session,
    region: str,
    services: List[str],
    state: AgentState,
    output_dir: str,
    account_id: Optional[str] = None,
) -> Inventory:
    """Run discovery across all (or selected) AWS services.

    Parameters
    ----------
    session:
        An authenticated boto3 ``Session``.
    region:
        Primary AWS region to scan (global services override this).
    services:
        List of service names to scan, or ``["all"]`` for everything.
    state:
        ``AgentState`` used for resumable runs — already-completed
        services are skipped.
    output_dir:
        Directory where per-service JSON files are written.
    account_id:
        AWS account ID (included in the inventory for reference).

    Returns
    -------
    Inventory
        Complete inventory of discovered resources.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Determine which discoverers to run
    if "all" in services:
        discoverers_to_run = ALL_DISCOVERERS
    else:
        discoverers_to_run = []
        for svc_name in services:
            svc_name = svc_name.strip().lower()
            if svc_name in _DISCOVERER_MAP:
                discoverers_to_run.append(_DISCOVERER_MAP[svc_name])
            else:
                logger.warning("Unknown service '%s' — skipping", svc_name)

    service_inventories: List[ServiceInventory] = []
    total_resources = 0

    for discoverer_cls in discoverers_to_run:
        svc_name = discoverer_cls.service_name

        # Skip services already completed in a previous (resumed) run
        if svc_name in state.completed_services:
            logger.info("Skipping %s (already completed)", svc_name)
            # Load previously saved results if present
            svc_file = out_path / f"{svc_name}.json"
            if svc_file.exists():
                try:
                    svc_inv = ServiceInventory.model_validate_json(
                        svc_file.read_text()
                    )
                    service_inventories.append(svc_inv)
                    total_resources += svc_inv.resource_count
                except Exception:
                    pass
            continue

        logger.info("Discovering %s ...", svc_name)
        try:
            discoverer = discoverer_cls(session, region)
            resources: List[DiscoveredResource] = discoverer.discover()
        except Exception as exc:
            msg = f"{svc_name} discovery raised an unhandled exception: {exc}"
            logger.error(msg)
            state.add_error(svc_name, str(exc))
            resources = []

        svc_inventory = ServiceInventory(
            service=svc_name,
            region=region,
            resource_count=len(resources),
            resources=resources,
        )
        service_inventories.append(svc_inventory)
        total_resources += len(resources)

        # Persist per-service JSON for incremental visibility
        svc_file = out_path / f"{svc_name}.json"
        svc_file.write_text(svc_inventory.model_dump_json(indent=2))

        # Mark as completed in state (for resume support)
        state.completed_services.append(svc_name)

        logger.info(
            "  %s: %d resource(s) discovered", svc_name, len(resources)
        )

    inventory = Inventory(
        account_id=account_id,
        region=region,
        discovered_at=datetime.now(timezone.utc).isoformat(),
        services=service_inventories,
        total_resources=total_resources,
    )

    return inventory
