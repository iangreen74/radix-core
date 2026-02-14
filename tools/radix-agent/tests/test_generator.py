"""Tests for the Terraform HCL generator."""

import os

import pytest

from radix_agent.models import DiscoveredResource
from radix_agent.terraform.generator import TerraformGenerator


# ===================================================================
# Helpers
# ===================================================================


def _make_resource(
    resource_type: str,
    resource_id: str,
    name: str = "",
    properties: dict = None,
) -> DiscoveredResource:
    """Build a minimal DiscoveredResource for testing."""
    return DiscoveredResource(
        service=resource_type.replace("aws_", "").split("_")[0],
        resource_type=resource_type,
        resource_id=resource_id,
        name=name or resource_id,
        properties=properties or {},
    )


# ===================================================================
# Tests
# ===================================================================


class TestGenerateRoute53Tf:
    """Generator should produce valid .tf content for Route 53 resources."""

    def test_generate_route53_tf(self, tmp_path):
        tf_dir = str(tmp_path / "infra")
        generator = TerraformGenerator(tf_dir)

        unmanaged = [
            _make_resource(
                "aws_route53_zone",
                "Z1234",
                "vaultscaler.com",
                {"name": "vaultscaler.com"},
            ),
            _make_resource(
                "aws_route53_record",
                "Z1234_api.vaultscaler.com_A",
                "api.vaultscaler.com",
                {
                    "zone_id": "Z1234",
                    "name": "api.vaultscaler.com",
                    "type": "A",
                    "ttl": 300,
                    "records": ["1.2.3.4"],
                },
            ),
        ]

        result = generator.generate(unmanaged)

        # Should produce two files: one for zones, one for records
        assert len(result) == 2

        # Verify the zone file
        zone_file = os.path.join(tf_dir, "radix_managed_route53_zone.tf")
        assert os.path.exists(zone_file)
        with open(zone_file) as f:
            content = f.read()
        assert 'resource "aws_route53_zone"' in content
        assert "vaultscaler.com" in content
        assert "radix-agent" in content  # header marker

        # Verify the record file
        record_file = os.path.join(tf_dir, "radix_managed_route53_record.tf")
        assert os.path.exists(record_file)
        with open(record_file) as f:
            content = f.read()
        assert 'resource "aws_route53_record"' in content
        assert "api.vaultscaler.com" in content
        assert "Z1234" in content


class TestGenerateS3Tf:
    """Generator should produce valid .tf content for S3 resources."""

    def test_generate_s3_tf(self, tmp_path):
        tf_dir = str(tmp_path / "infra")
        generator = TerraformGenerator(tf_dir)

        unmanaged = [
            _make_resource(
                "aws_s3_bucket",
                "radix-artifacts",
                "radix-artifacts",
                {"bucket": "radix-artifacts"},
            ),
            _make_resource(
                "aws_s3_bucket",
                "radix-logs",
                "radix-logs",
                {"bucket": "radix-logs"},
            ),
        ]

        result = generator.generate(unmanaged)

        assert len(result) == 1  # both go into one file

        s3_file = os.path.join(tf_dir, "radix_managed_s3_bucket.tf")
        assert os.path.exists(s3_file)
        assert result[s3_file] == 2

        with open(s3_file) as f:
            content = f.read()

        assert 'resource "aws_s3_bucket"' in content
        assert "radix_artifacts" in content  # sanitized name
        assert "radix_logs" in content
        assert content.count('resource "aws_s3_bucket"') == 2


class TestGenerateNoUnmanaged:
    """Generator should return empty dict when there are no unmanaged resources."""

    def test_generate_no_unmanaged(self, tmp_path):
        tf_dir = str(tmp_path / "infra")
        generator = TerraformGenerator(tf_dir)

        result = generator.generate([])

        assert result == {}
        # Directory should NOT have been created (no files to write)
        assert not os.path.exists(tf_dir)


class TestGenerateUnsupportedType:
    """Resources with no template should be silently skipped."""

    def test_generate_unsupported_type(self, tmp_path):
        tf_dir = str(tmp_path / "infra")
        generator = TerraformGenerator(tf_dir)

        unmanaged = [
            _make_resource(
                "aws_some_unknown_resource",
                "id-1",
                "mystery",
            ),
        ]

        result = generator.generate(unmanaged)

        # No template for this type, so nothing should be generated
        assert result == {}


class TestGenerateDuplicateNames:
    """Resources with the same sanitized name should get unique suffixes."""

    def test_generate_duplicate_names(self, tmp_path):
        tf_dir = str(tmp_path / "infra")
        generator = TerraformGenerator(tf_dir)

        unmanaged = [
            _make_resource(
                "aws_s3_bucket",
                "my-bucket",
                "my-bucket",
                {"bucket": "my-bucket"},
            ),
            _make_resource(
                "aws_s3_bucket",
                "my-bucket-2",
                "my-bucket",  # same name -> collision
                {"bucket": "my-bucket-2"},
            ),
        ]

        result = generator.generate(unmanaged)

        s3_file = os.path.join(tf_dir, "radix_managed_s3_bucket.tf")
        with open(s3_file) as f:
            content = f.read()

        # Should have two resource blocks with different tf names
        assert content.count('resource "aws_s3_bucket"') == 2
        assert "my_bucket" in content
        # The second one should get a suffix to avoid collision
        assert "my_bucket_1" in content
