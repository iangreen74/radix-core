"""Shared fixtures for radix-agent tests.

Uses moto v5 ``mock_aws`` to spin up fake AWS services so that every
discoverer can be tested against realistic API responses without touching
real infrastructure.

The ``mock_aws_env`` fixture is the central piece: it starts the moto mock
context **before** any resource-provisioning fixtures run, and tears it down
after the test completes.  All other fixtures depend on it (directly or
transitively through ``aws_session``).
"""

import json
import os

import boto3
import pytest
from moto import mock_aws


# -----------------------------------------------------------------------
# Core mock context -- every test that touches AWS must depend on this
# -----------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_aws_env():
    """Activate moto's mock_aws context for every test in this directory."""
    # Set dummy credentials so boto3 never reaches real AWS
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-west-2"

    with mock_aws():
        yield


# -----------------------------------------------------------------------
# Low-level session fixture
# -----------------------------------------------------------------------

@pytest.fixture
def aws_region():
    """Default AWS region used throughout the test suite."""
    return "us-west-2"


@pytest.fixture
def aws_session(aws_region):
    """Return a boto3 Session wired to the test region with dummy creds."""
    return boto3.Session(
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        aws_session_token="testing",
        region_name=aws_region,
    )


# -----------------------------------------------------------------------
# Route 53
# -----------------------------------------------------------------------

@pytest.fixture
def route53_resources(aws_session):
    """Create a Route 53 hosted zone with several record types."""
    client = aws_session.client("route53", region_name="us-east-1")

    zone = client.create_hosted_zone(
        Name="vaultscaler.com",
        CallerReference="ref-1",
        HostedZoneConfig={"Comment": "Primary zone", "PrivateZone": False},
    )
    zone_id = zone["HostedZone"]["Id"].split("/")[-1]

    # A record
    client.change_resource_record_sets(
        HostedZoneId=zone_id,
        ChangeBatch={
            "Changes": [
                {
                    "Action": "CREATE",
                    "ResourceRecordSet": {
                        "Name": "api.vaultscaler.com",
                        "Type": "A",
                        "TTL": 300,
                        "ResourceRecords": [{"Value": "1.2.3.4"}],
                    },
                },
            ]
        },
    )

    # CNAME record
    client.change_resource_record_sets(
        HostedZoneId=zone_id,
        ChangeBatch={
            "Changes": [
                {
                    "Action": "CREATE",
                    "ResourceRecordSet": {
                        "Name": "www.vaultscaler.com",
                        "Type": "CNAME",
                        "TTL": 300,
                        "ResourceRecords": [{"Value": "vaultscaler.com"}],
                    },
                },
            ]
        },
    )

    # ALIAS (A record with AliasTarget)
    client.change_resource_record_sets(
        HostedZoneId=zone_id,
        ChangeBatch={
            "Changes": [
                {
                    "Action": "CREATE",
                    "ResourceRecordSet": {
                        "Name": "dashboard.vaultscaler.com",
                        "Type": "A",
                        "AliasTarget": {
                            "HostedZoneId": "Z2FDTNDATAQYW2",
                            "DNSName": "d111111abcdef8.cloudfront.net",
                            "EvaluateTargetHealth": False,
                        },
                    },
                },
            ]
        },
    )

    return {"zone_id": zone_id, "zone_name": "vaultscaler.com"}


# -----------------------------------------------------------------------
# CloudFront
# -----------------------------------------------------------------------

@pytest.fixture
def cloudfront_resources(aws_session):
    """Create a CloudFront distribution with an alias."""
    client = aws_session.client("cloudfront", region_name="us-east-1")

    dist_config = {
        "CallerReference": "ref-cf-1",
        "Aliases": {"Quantity": 1, "Items": ["dashboard.vaultscaler.com"]},
        "DefaultRootObject": "index.html",
        "Comment": "Dashboard CDN",
        "Enabled": True,
        "Origins": {
            "Quantity": 1,
            "Items": [
                {
                    "Id": "S3-dashboard",
                    "DomainName": "dashboard-bucket.s3.amazonaws.com",
                    "S3OriginConfig": {"OriginAccessIdentity": ""},
                }
            ],
        },
        "DefaultCacheBehavior": {
            "TargetOriginId": "S3-dashboard",
            "ViewerProtocolPolicy": "redirect-to-https",
            "TrustedSigners": {"Enabled": False, "Quantity": 0},
            "ForwardedValues": {
                "QueryString": False,
                "Cookies": {"Forward": "none"},
            },
            "MinTTL": 0,
        },
        "ViewerCertificate": {
            "CloudFrontDefaultCertificate": True,
        },
    }

    resp = client.create_distribution(DistributionConfig=dist_config)
    dist_id = resp["Distribution"]["Id"]
    return {"distribution_id": dist_id}


# -----------------------------------------------------------------------
# S3
# -----------------------------------------------------------------------

@pytest.fixture
def s3_resources(aws_session, aws_region):
    """Create two S3 buckets."""
    client = aws_session.client("s3", region_name=aws_region)

    # Bucket 1
    client.create_bucket(
        Bucket="radix-artifacts-bucket",
        CreateBucketConfiguration={"LocationConstraint": aws_region},
    )
    # Bucket 2
    client.create_bucket(
        Bucket="radix-logs-bucket",
        CreateBucketConfiguration={"LocationConstraint": aws_region},
    )

    return {"buckets": ["radix-artifacts-bucket", "radix-logs-bucket"]}


# -----------------------------------------------------------------------
# EKS
# -----------------------------------------------------------------------

@pytest.fixture
def eks_resources(aws_session, aws_region, vpc_resources):
    """Create an EKS cluster with one node group.

    Depends on ``vpc_resources`` for subnet and security group IDs.
    """
    # We need an IAM role ARN for the cluster and node group
    iam_client = aws_session.client("iam")
    assume_role_doc = json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "eks.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    })
    cluster_role = iam_client.create_role(
        RoleName="eks-cluster-role-for-test",
        AssumeRolePolicyDocument=assume_role_doc,
        Path="/",
    )
    cluster_role_arn = cluster_role["Role"]["Arn"]

    node_role = iam_client.create_role(
        RoleName="eks-node-role-for-test",
        AssumeRolePolicyDocument=assume_role_doc,
        Path="/",
    )
    node_role_arn = node_role["Role"]["Arn"]

    eks_client = aws_session.client("eks", region_name=aws_region)

    subnet_ids = vpc_resources["subnet_ids"]
    sg_id = vpc_resources["security_group_id"]

    eks_client.create_cluster(
        name="radix",
        version="1.28",
        roleArn=cluster_role_arn,
        resourcesVpcConfig={
            "subnetIds": subnet_ids,
            "securityGroupIds": [sg_id],
        },
    )

    eks_client.create_nodegroup(
        clusterName="radix",
        nodegroupName="radix-workers",
        nodeRole=node_role_arn,
        subnets=subnet_ids,
        scalingConfig={"minSize": 1, "maxSize": 3, "desiredSize": 2},
        instanceTypes=["t3.medium"],
    )

    return {
        "cluster_name": "radix",
        "nodegroup_name": "radix-workers",
        "cluster_role_arn": cluster_role_arn,
        "node_role_arn": node_role_arn,
    }


# -----------------------------------------------------------------------
# ACM
# -----------------------------------------------------------------------

@pytest.fixture
def acm_resources(aws_session, aws_region):
    """Request an ACM wildcard certificate."""
    client = aws_session.client("acm", region_name=aws_region)

    resp = client.request_certificate(
        DomainName="*.vaultscaler.com",
        ValidationMethod="DNS",
    )
    cert_arn = resp["CertificateArn"]
    return {"certificate_arn": cert_arn}


# -----------------------------------------------------------------------
# EC2 / VPC
# -----------------------------------------------------------------------

@pytest.fixture
def vpc_resources(aws_session, aws_region):
    """Create a VPC with 2 subnets and 1 security group."""
    ec2 = aws_session.client("ec2", region_name=aws_region)

    vpc = ec2.create_vpc(CidrBlock="10.0.0.0/16")
    vpc_id = vpc["Vpc"]["VpcId"]
    ec2.create_tags(
        Resources=[vpc_id],
        Tags=[{"Key": "Name", "Value": "radix-vpc"}],
    )

    subnet1 = ec2.create_subnet(
        VpcId=vpc_id,
        CidrBlock="10.0.1.0/24",
        AvailabilityZone=f"{aws_region}a",
    )
    subnet1_id = subnet1["Subnet"]["SubnetId"]
    ec2.create_tags(
        Resources=[subnet1_id],
        Tags=[{"Key": "Name", "Value": "radix-public-a"}],
    )

    subnet2 = ec2.create_subnet(
        VpcId=vpc_id,
        CidrBlock="10.0.2.0/24",
        AvailabilityZone=f"{aws_region}b",
    )
    subnet2_id = subnet2["Subnet"]["SubnetId"]
    ec2.create_tags(
        Resources=[subnet2_id],
        Tags=[{"Key": "Name", "Value": "radix-public-b"}],
    )

    sg = ec2.create_security_group(
        GroupName="radix-sg",
        Description="Radix security group",
        VpcId=vpc_id,
    )
    sg_id = sg["GroupId"]
    ec2.create_tags(
        Resources=[sg_id],
        Tags=[{"Key": "Name", "Value": "radix-sg"}],
    )

    return {
        "vpc_id": vpc_id,
        "subnet_ids": [subnet1_id, subnet2_id],
        "security_group_id": sg_id,
    }


# -----------------------------------------------------------------------
# IAM
# -----------------------------------------------------------------------

@pytest.fixture
def iam_resources(aws_session):
    """Create two IAM roles: one radix-prefixed, one eks-prefixed."""
    client = aws_session.client("iam")

    assume_role_doc = json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    })

    client.create_role(
        RoleName="radix-platform-role",
        AssumeRolePolicyDocument=assume_role_doc,
        Path="/",
        Description="Radix platform role",
    )

    client.create_role(
        RoleName="eks-node-role",
        AssumeRolePolicyDocument=assume_role_doc,
        Path="/",
        Description="EKS node role",
    )

    return {"roles": ["radix-platform-role", "eks-node-role"]}


# -----------------------------------------------------------------------
# SSM
# -----------------------------------------------------------------------

@pytest.fixture
def ssm_resources(aws_session, aws_region):
    """Create two SSM parameters under /radix/."""
    client = aws_session.client("ssm", region_name=aws_region)

    client.put_parameter(
        Name="/radix/cluster/name",
        Value="radix",
        Type="String",
    )
    client.put_parameter(
        Name="/radix/cluster/version",
        Value="1.28",
        Type="String",
    )

    return {"parameters": ["/radix/cluster/name", "/radix/cluster/version"]}
