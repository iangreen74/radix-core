"""Reconcile discovered AWS resources against existing Terraform configuration.

Parses .tf files in the infrastructure directory to determine which resource
types are already managed by Terraform and which were discovered in the live
account but have no corresponding HCL definition.
"""

import os
import re
from typing import Dict, List, Set, Tuple

from ..models import DiscoveredResource, ReconcileResult

# Matches `resource "aws_xxx" "name" {` with optional whitespace.
_RESOURCE_BLOCK_RE = re.compile(
    r'^\s*resource\s+"([^"]+)"\s+"([^"]+)"\s*\{',
    re.MULTILINE,
)

# Index suffixes produced by count / for_each in Terraform state or plans.
_INDEX_SUFFIX_RE = re.compile(r"\[.*\]$")


class TerraformReconciler:
    """Compare a live-account resource inventory with existing .tf files.

    Parameters
    ----------
    tf_dir:
        Absolute path to the directory containing the Terraform configuration
        (e.g. ``infra/aws``).
    """

    def __init__(self, tf_dir: str) -> None:
        self.tf_dir = tf_dir

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse_existing_resources(self) -> Set[Tuple[str, str]]:
        """Return every ``(resource_type, logical_name)`` declared in .tf files.

        Handles ``count`` and ``for_each`` variants by stripping any trailing
        index (``[0]``, ``[count.index]``, ``["key"]``, etc.) from the
        logical name if present — though in practice the name in the
        ``resource`` block itself never contains these; the stripping is a
        safety-net for downstream callers that may pass state-style addresses.

        Returns
        -------
        set of (str, str)
            Unique ``(resource_type, logical_name)`` pairs found across all
            ``.tf`` files in ``self.tf_dir``.
        """
        resources: Set[Tuple[str, str]] = set()

        if not os.path.isdir(self.tf_dir):
            return resources

        for filename in sorted(os.listdir(self.tf_dir)):
            if not filename.endswith(".tf"):
                continue
            filepath = os.path.join(self.tf_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as fh:
                    content = fh.read()
            except OSError:
                continue

            for match in _RESOURCE_BLOCK_RE.finditer(content):
                rtype = match.group(1)
                rname = _INDEX_SUFFIX_RE.sub("", match.group(2))
                resources.add((rtype, rname))

        return resources

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def reconcile(self, inventory: List[DiscoveredResource]) -> ReconcileResult:
        """Reconcile *inventory* against the Terraform configuration on disk.

        Classification rules:

        * **managed** — the resource's ``resource_type`` (e.g.
          ``aws_eks_cluster``) matches a type present in at least one ``.tf``
          file.
        * **unmanaged** — the resource's ``resource_type`` does *not* appear
          in any ``.tf`` file.
        * **framework_only** — Terraform addresses (``type.name``) that exist
          in ``.tf`` files but have no corresponding entry in *inventory*.

        Parameters
        ----------
        inventory:
            Resources discovered in the live AWS account.

        Returns
        -------
        ReconcileResult
        """
        existing = self.parse_existing_resources()

        # Build a set of resource *types* present in the .tf files.
        existing_types: Set[str] = {rtype for rtype, _ in existing}

        managed: List[DiscoveredResource] = []
        unmanaged: List[DiscoveredResource] = []

        # Types seen in the inventory — used to detect framework-only later.
        inventory_types: Set[str] = set()

        for resource in inventory:
            inventory_types.add(resource.resource_type)

            if resource.resource_type in existing_types:
                resource.exists_in_tf = True
                managed.append(resource)
            else:
                resource.exists_in_tf = False
                unmanaged.append(resource)

        # Framework-only: terraform addresses whose *type* was never seen in
        # the discovered inventory.
        framework_only: List[str] = sorted(
            f"{rtype}.{rname}"
            for rtype, rname in existing
            if rtype not in inventory_types
        )

        summary: Dict[str, int] = {
            "total_discovered": len(inventory),
            "managed": len(managed),
            "unmanaged": len(unmanaged),
            "framework_only": len(framework_only),
        }

        return ReconcileResult(
            managed=managed,
            unmanaged=unmanaged,
            framework_only=framework_only,
            summary=summary,
        )
