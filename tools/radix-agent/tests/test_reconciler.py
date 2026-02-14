"""Tests for the Terraform reconciler."""

import os

import pytest

from radix_agent.models import DiscoveredResource
from radix_agent.terraform.reconciler import TerraformReconciler


# ===================================================================
# Helpers
# ===================================================================


def _make_resource(resource_type: str, resource_id: str, name: str = "") -> DiscoveredResource:
    """Helper to build a minimal DiscoveredResource for testing."""
    return DiscoveredResource(
        service=resource_type.split("_")[1] if "_" in resource_type else "unknown",
        resource_type=resource_type,
        resource_id=resource_id,
        name=name or resource_id,
    )


@pytest.fixture
def tf_dir(tmp_path):
    """Write sample .tf files into a temp directory and return the path."""
    tf = tmp_path / "infra"
    tf.mkdir()

    # Write a file with two resource blocks
    (tf / "main.tf").write_text(
        '''\
resource "aws_eks_cluster" "radix" {
  name     = "radix"
  role_arn = aws_iam_role.eks_cluster.arn
}

resource "aws_eks_node_group" "radix_workers" {
  cluster_name    = aws_eks_cluster.radix.name
  node_group_name = "radix-workers"
}
'''
    )

    # Write another file with a different resource type
    (tf / "s3.tf").write_text(
        '''\
resource "aws_s3_bucket" "artifacts" {
  bucket = "radix-artifacts"
}
'''
    )

    return str(tf)


# ===================================================================
# Tests
# ===================================================================


class TestParseExistingResources:
    """TerraformReconciler.parse_existing_resources should parse .tf files."""

    def test_parse_existing_resources(self, tf_dir):
        reconciler = TerraformReconciler(tf_dir)
        existing = reconciler.parse_existing_resources()

        # Should find three resource blocks
        assert ("aws_eks_cluster", "radix") in existing
        assert ("aws_eks_node_group", "radix_workers") in existing
        assert ("aws_s3_bucket", "artifacts") in existing
        assert len(existing) == 3

    def test_parse_empty_dir(self, tmp_path):
        """An empty directory should return an empty set."""
        empty_dir = str(tmp_path / "empty")
        os.makedirs(empty_dir)
        reconciler = TerraformReconciler(empty_dir)
        assert reconciler.parse_existing_resources() == set()

    def test_parse_nonexistent_dir(self, tmp_path):
        """A non-existent directory should return an empty set."""
        reconciler = TerraformReconciler(str(tmp_path / "does_not_exist"))
        assert reconciler.parse_existing_resources() == set()


class TestReconcileManaged:
    """Resources whose type exists in .tf files should be classified as managed."""

    def test_reconcile_managed(self, tf_dir):
        reconciler = TerraformReconciler(tf_dir)

        inventory = [
            _make_resource("aws_eks_cluster", "radix"),
            _make_resource("aws_s3_bucket", "radix-artifacts"),
        ]

        result = reconciler.reconcile(inventory)

        assert len(result.managed) == 2
        assert len(result.unmanaged) == 0
        assert result.summary["managed"] == 2
        assert result.summary["unmanaged"] == 0

        # Both resources should be flagged as existing in tf
        for r in result.managed:
            assert r.exists_in_tf is True


class TestReconcileUnmanaged:
    """Resources whose type is not in any .tf file should be classified as unmanaged."""

    def test_reconcile_unmanaged(self, tf_dir):
        reconciler = TerraformReconciler(tf_dir)

        inventory = [
            _make_resource("aws_route53_zone", "Z1234", "vaultscaler.com"),
            _make_resource("aws_cloudfront_distribution", "E1234"),
        ]

        result = reconciler.reconcile(inventory)

        assert len(result.managed) == 0
        assert len(result.unmanaged) == 2
        assert result.summary["unmanaged"] == 2

        for r in result.unmanaged:
            assert r.exists_in_tf is False


class TestReconcileFrameworkOnly:
    """Resources in .tf that have no matching inventory type should be framework_only."""

    def test_reconcile_framework_only(self, tf_dir):
        reconciler = TerraformReconciler(tf_dir)

        # Pass inventory that only contains S3 resources.
        # EKS cluster and node group types are NOT in the inventory,
        # so they should appear in framework_only.
        inventory = [
            _make_resource("aws_s3_bucket", "radix-artifacts"),
        ]

        result = reconciler.reconcile(inventory)

        assert len(result.managed) == 1  # s3 bucket type matches
        assert len(result.unmanaged) == 0

        # EKS resources exist in .tf but not in inventory -> framework_only
        assert "aws_eks_cluster.radix" in result.framework_only
        assert "aws_eks_node_group.radix_workers" in result.framework_only
        assert len(result.framework_only) == 2
        assert result.summary["framework_only"] == 2


class TestReconcileMixed:
    """A mixed inventory should correctly split into managed, unmanaged, and framework_only."""

    def test_reconcile_mixed(self, tf_dir):
        reconciler = TerraformReconciler(tf_dir)

        inventory = [
            # Matches aws_eks_cluster in .tf -> managed
            _make_resource("aws_eks_cluster", "radix"),
            # aws_iam_role not in .tf -> unmanaged
            _make_resource("aws_iam_role", "radix-role"),
        ]

        result = reconciler.reconcile(inventory)

        assert len(result.managed) == 1
        assert result.managed[0].resource_type == "aws_eks_cluster"

        assert len(result.unmanaged) == 1
        assert result.unmanaged[0].resource_type == "aws_iam_role"

        # aws_s3_bucket and aws_eks_node_group are in .tf but their types
        # are not in the inventory -> framework_only
        assert "aws_s3_bucket.artifacts" in result.framework_only
        assert "aws_eks_node_group.radix_workers" in result.framework_only
