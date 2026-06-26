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
# Shared values are published by the shared stack to SSM (/idi/<stack>/shared/*)
# and read here. We use SSM rather than a StackReference because every Pulumi
# project in this org shares the name `idi`, which makes a cross-stack
# StackReference ambiguous on the self-managed S3 backend.
bucket_name = aws.ssm.get_parameter(
    name=f"/idi/{stack_name}/shared/processor_bucket_name"
).value
shared_dlq_name = aws.ssm.get_parameter(
    name=f"/idi/{stack_name}/shared/dlq_name"
).value
log_retention_days = int(config.get("log_retention_days") or "30")
dlq_retention_days = int(config.get("dlq_retention_days") or "14")
ecr_image_count = int(config.get("ecr_image_count") or "5")
# SEC EDGAR User-Agent header ("Name email@example.com") — a required public
# contact string, not a credential, so it lives in committed config (not SSM).
sec_user_agent = config.require("sec_user_agent")

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
