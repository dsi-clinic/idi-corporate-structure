"""Pulumi configuration and shared constants."""

import pulumi_aws as aws

import pulumi

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
config = pulumi.Config("idi")
project_name = pulumi.get_project()
app_name = config.get("app_name") or "corporate-structure"
stack_name = pulumi.get_stack()
name_prefix = f"{project_name}-{stack_name}-{app_name}"
bucket_name = config.require("bucket_name")

# AWS
aws_config = pulumi.Config("aws")
aws_region = aws_config.require("region")
caller = aws.get_caller_identity()


def tags(extra: dict | None = None) -> dict:
    """Common resource tags."""
    t = {
        "project": project_name,
        "environment": stack_name,
        "managed_by": "Pulumi",
        "app_name": app_name,
    }
    if extra:
        t.update(extra)
    return t
