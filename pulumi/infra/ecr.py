"""ECR repository for the corporate structure orchestrator image."""

import pulumi_aws as aws

import pulumi

from . import config

# -----------------------------------------------------------------------------
# ECR Registry & Image URIs
# -----------------------------------------------------------------------------
ecr_registry = pulumi.Output.from_input(config.caller.account_id).apply(
    lambda aid: f"{aid}.dkr.ecr.{config.aws_region}.amazonaws.com"
)

ecr_repo = aws.ecr.Repository(
    "idi-ecr-orchestrator",
    name=f"{config.name_prefix}-orchestrator",
    force_delete=True,
)

orchestrator_image = ecr_registry.apply(
    lambda r: f"{r}/{config.name_prefix}-orchestrator:latest"
)
