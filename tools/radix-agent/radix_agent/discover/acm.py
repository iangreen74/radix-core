"""Discover ACM (AWS Certificate Manager) certificates."""

import logging
from typing import List

from ..models import DiscoveredResource
from .base import BaseDiscoverer

logger = logging.getLogger(__name__)


class AcmDiscoverer(BaseDiscoverer):
    """Discover ACM certificates.

    Checks both the configured region and ``us-east-1`` (where certificates
    used by CloudFront must reside).  De-duplicates if the configured region
    is already ``us-east-1``.
    """

    service_name = "acm"

    def discover(self) -> List[DiscoveredResource]:
        resources: List[DiscoveredResource] = []
        seen_arns: set = set()

        # Discover in the configured region
        resources.extend(self._discover_region(self.region, seen_arns))

        # Also check us-east-1 for CloudFront certs (skip if already done)
        if self.region != "us-east-1":
            resources.extend(self._discover_region("us-east-1", seen_arns))

        return resources

    def _discover_region(
        self, region: str, seen_arns: set
    ) -> List[DiscoveredResource]:
        """Discover ACM certificates in a specific region."""
        resources: List[DiscoveredResource] = []
        try:
            client = self.session.client("acm", region_name=region)
            cert_summaries = self._list_all_certificates(client)

            for summary in cert_summaries:
                cert_arn = summary["CertificateArn"]
                if cert_arn in seen_arns:
                    continue
                seen_arns.add(cert_arn)

                try:
                    detail = client.describe_certificate(
                        CertificateArn=cert_arn
                    )["Certificate"]
                except Exception as exc:
                    logger.warning(
                        "Failed to describe certificate %s: %s", cert_arn, exc
                    )
                    continue

                domain = detail.get("DomainName", "")
                label = self._sanitize_name(domain)
                tf_addr = f"aws_acm_certificate.{label}"

                # Subject alternative names
                sans = detail.get("SubjectAlternativeNames", [])

                # Domain validation options
                dvo = []
                for opt in detail.get("DomainValidationOptions", []):
                    dvo_entry = {
                        "domain_name": opt.get("DomainName", ""),
                        "validation_status": opt.get("ValidationStatus", ""),
                    }
                    rr = opt.get("ResourceRecord", {})
                    if rr:
                        dvo_entry["resource_record"] = {
                            "name": rr.get("Name", ""),
                            "type": rr.get("Type", ""),
                            "value": rr.get("Value", ""),
                        }
                    dvo.append(dvo_entry)

                properties = {
                    "domain_name": domain,
                    "subject_alternative_names": sans,
                    "status": detail.get("Status", ""),
                    "type": detail.get("Type", ""),
                    "issuer": detail.get("Issuer", ""),
                    "key_algorithm": detail.get("KeyAlgorithm", ""),
                    "validation_method": detail.get(
                        "DomainValidationOptions", [{}]
                    )[0].get("ValidationMethod", "")
                    if detail.get("DomainValidationOptions")
                    else "",
                    "in_use_by": detail.get("InUseBy", []),
                    "region": region,
                    "not_before": str(detail.get("NotBefore", "")),
                    "not_after": str(detail.get("NotAfter", "")),
                    "renewal_eligibility": detail.get(
                        "RenewalEligibility", ""
                    ),
                    "domain_validation_options": dvo,
                }

                resources.append(
                    DiscoveredResource(
                        service="acm",
                        resource_type="aws_acm_certificate",
                        resource_id=cert_arn,
                        arn=cert_arn,
                        name=domain,
                        properties=properties,
                        terraform_address=tf_addr,
                    )
                )

        except Exception as exc:
            logger.warning("ACM discovery failed in %s: %s", region, exc)

        return resources

    # ------------------------------------------------------------------
    # Pagination
    # ------------------------------------------------------------------

    @staticmethod
    def _list_all_certificates(client) -> list:
        """Paginate through list_certificates."""
        certs: list = []
        paginator = client.get_paginator("list_certificates")
        for page in paginator.paginate(
            Includes={
                "keyTypes": [
                    "RSA_1024",
                    "RSA_2048",
                    "RSA_3072",
                    "RSA_4096",
                    "EC_prime256v1",
                    "EC_secp384r1",
                    "EC_secp521r1",
                ]
            }
        ):
            certs.extend(page.get("CertificateSummaryList", []))
        return certs
