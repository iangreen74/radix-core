"""Tests for individual AWS resource discoverers and the orchestrator."""

import json
from pathlib import Path

import pytest

from radix_agent.discover import run_discovery, ALL_DISCOVERERS
from radix_agent.discover.route53 import Route53Discoverer
from radix_agent.discover.cloudfront import CloudFrontDiscoverer
from radix_agent.discover.s3 import S3Discoverer
from radix_agent.discover.vpc import VpcDiscoverer
from radix_agent.discover.iam import IamDiscoverer
from radix_agent.discover.ssm import SsmDiscoverer
from radix_agent.state import AgentState


# ===================================================================
# Route 53
# ===================================================================


class TestRoute53Discover:
    """Route53Discoverer should find zones and records, skipping apex NS/SOA."""

    def test_route53_discover(self, aws_session, aws_region, route53_resources):
        discoverer = Route53Discoverer(aws_session, aws_region)
        resources = discoverer.discover()

        # Should have 1 zone + 3 user-created records (A, CNAME, ALIAS)
        # The apex NS and SOA records are auto-created by moto but should be skipped.
        zone_resources = [r for r in resources if r.resource_type == "aws_route53_zone"]
        record_resources = [r for r in resources if r.resource_type == "aws_route53_record"]

        assert len(zone_resources) == 1
        assert zone_resources[0].name == "vaultscaler.com"
        assert zone_resources[0].properties["private_zone"] is False

        # Should have our 3 records and NOT any apex NS/SOA
        record_names = {r.name for r in record_resources}
        assert "api.vaultscaler.com" in record_names
        assert "www.vaultscaler.com" in record_names
        assert "dashboard.vaultscaler.com" in record_names

        # Verify the alias record has alias properties
        alias_records = [
            r for r in record_resources
            if r.name == "dashboard.vaultscaler.com"
        ]
        assert len(alias_records) == 1
        assert "alias" in alias_records[0].properties

        # Verify no apex NS or SOA records leaked through
        record_types_at_apex = [
            r.properties["type"]
            for r in record_resources
            if r.name == "vaultscaler.com"
        ]
        assert "NS" not in record_types_at_apex
        assert "SOA" not in record_types_at_apex


# ===================================================================
# CloudFront
# ===================================================================


class TestCloudFrontDiscover:
    """CloudFrontDiscoverer should find distributions with correct properties."""

    def test_cloudfront_discover(
        self, aws_session, aws_region, cloudfront_resources
    ):
        discoverer = CloudFrontDiscoverer(aws_session, aws_region)
        resources = discoverer.discover()

        assert len(resources) == 1

        dist = resources[0]
        assert dist.resource_type == "aws_cloudfront_distribution"
        assert "dashboard.vaultscaler.com" in dist.properties.get("aliases", [])
        assert dist.name == "dashboard.vaultscaler.com"
        assert dist.properties["enabled"] is True

        # Origins should be populated
        origins = dist.properties.get("origins", [])
        assert len(origins) == 1
        assert "s3.amazonaws.com" in origins[0]["domain_name"]

        # terraform_address should use sanitized alias
        assert dist.terraform_address == "aws_cloudfront_distribution.dashboard_vaultscaler_com"


# ===================================================================
# S3
# ===================================================================


class TestS3Discover:
    """S3Discoverer should find buckets."""

    def test_s3_discover(self, aws_session, aws_region, s3_resources):
        discoverer = S3Discoverer(aws_session, aws_region)
        resources = discoverer.discover()

        bucket_names = {r.name for r in resources}
        assert "radix-artifacts-bucket" in bucket_names
        assert "radix-logs-bucket" in bucket_names
        assert len(resources) >= 2

        for r in resources:
            assert r.resource_type == "aws_s3_bucket"
            assert r.arn.startswith("arn:aws:s3:::")
            assert r.properties["bucket_name"] == r.name


# ===================================================================
# VPC
# ===================================================================


class TestVpcDiscover:
    """VpcDiscoverer should find VPC, subnets, and other networking resources."""

    def test_vpc_discover(self, aws_session, aws_region, vpc_resources):
        discoverer = VpcDiscoverer(aws_session, aws_region)
        resources = discoverer.discover()

        vpc_res = [r for r in resources if r.resource_type == "aws_vpc"]
        subnet_res = [r for r in resources if r.resource_type == "aws_subnet"]
        sg_res = [r for r in resources if r.resource_type == "aws_security_group"]

        # At least 1 VPC (moto may also create a default VPC)
        assert len(vpc_res) >= 1
        # Find our named VPC
        radix_vpcs = [v for v in vpc_res if v.name == "radix-vpc"]
        assert len(radix_vpcs) == 1
        assert radix_vpcs[0].properties["cidr_block"] == "10.0.0.0/16"

        # At least 2 subnets we created
        our_subnets = [s for s in subnet_res if s.name in ("radix-public-a", "radix-public-b")]
        assert len(our_subnets) == 2

        # At least our security group (moto creates default SGs too)
        our_sgs = [s for s in sg_res if s.name == "radix-sg"]
        assert len(our_sgs) == 1


# ===================================================================
# IAM
# ===================================================================


class TestIamDiscover:
    """IamDiscoverer should find roles matching radix/eks prefixes."""

    def test_iam_discover(self, aws_session, aws_region, iam_resources):
        discoverer = IamDiscoverer(aws_session, aws_region)
        resources = discoverer.discover()

        role_resources = [r for r in resources if r.resource_type == "aws_iam_role"]
        role_names = {r.name for r in role_resources}

        assert "radix-platform-role" in role_names
        assert "eks-node-role" in role_names


# ===================================================================
# SSM
# ===================================================================


class TestSsmDiscover:
    """SsmDiscoverer should find parameters under /radix/."""

    def test_ssm_discover(self, aws_session, aws_region, ssm_resources):
        discoverer = SsmDiscoverer(aws_session, aws_region)
        resources = discoverer.discover()

        param_names = {r.name for r in resources}
        assert "/radix/cluster/name" in param_names
        assert "/radix/cluster/version" in param_names
        assert len(resources) == 2

        for r in resources:
            assert r.resource_type == "aws_ssm_parameter"
            assert r.properties["type"] == "String"


# ===================================================================
# Discovery orchestrator
# ===================================================================


class TestRunDiscovery:
    """Tests for the top-level ``run_discovery`` orchestrator."""

    def test_run_discovery_all(
        self,
        aws_session,
        aws_region,
        route53_resources,
        s3_resources,
        vpc_resources,
        iam_resources,
        ssm_resources,
        tmp_path,
    ):
        """Running with services=['all'] should exercise every discoverer."""
        state = AgentState()
        output_dir = str(tmp_path / "discovery_output")

        inventory = run_discovery(
            session=aws_session,
            region=aws_region,
            services=["all"],
            state=state,
            output_dir=output_dir,
            account_id="123456789012",
        )

        # Verify we got an inventory back with resources
        assert inventory.account_id == "123456789012"
        assert inventory.region == aws_region
        assert inventory.total_resources > 0
        assert len(inventory.services) > 0

        # All discoverers should have been marked as completed
        for disc_cls in ALL_DISCOVERERS:
            assert disc_cls.service_name in state.completed_services

        # Per-service JSON files should be written
        output_path = Path(output_dir)
        json_files = list(output_path.glob("*.json"))
        assert len(json_files) > 0

        # Verify at least one file is valid JSON
        first_file = json_files[0]
        data = json.loads(first_file.read_text())
        assert "service" in data
        assert "resources" in data

    def test_run_discovery_resume(
        self,
        aws_session,
        aws_region,
        s3_resources,
        tmp_path,
    ):
        """Services already in ``state.completed_services`` should be skipped."""
        state = AgentState()
        output_dir = str(tmp_path / "resume_output")

        # First run: discover just S3
        run_discovery(
            session=aws_session,
            region=aws_region,
            services=["s3"],
            state=state,
            output_dir=output_dir,
        )

        assert "s3" in state.completed_services
        s3_json = Path(output_dir) / "s3.json"
        assert s3_json.exists()
        first_run_content = s3_json.read_text()

        # Modify the persisted JSON so we can detect if it gets re-written
        s3_json.write_text('{"service":"s3","region":"us-west-2","resource_count":0,"resources":[]}')

        # Second run: ask for S3 again -- should be skipped
        run_discovery(
            session=aws_session,
            region=aws_region,
            services=["s3"],
            state=state,
            output_dir=output_dir,
        )

        # The file should NOT have been overwritten with fresh discovery results
        # because S3 was already in completed_services.
        current_content = s3_json.read_text()
        assert current_content != first_run_content
