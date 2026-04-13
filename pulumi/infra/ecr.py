"""ECR repository for the corporate structure orchestrator image."""

import json

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

orchestrator_image = ecr_registry.apply(lambda r: f"{r}/{config.name_prefix}-orchestrator:latest")

# Lifecycle policy — expire images beyond the last 5 to avoid unbounded storage growth
ecr_lifecycle_policy = aws.ecr.LifecyclePolicy(
    "idi-ecr-lifecycle",
    repository=ecr_repo.name,
    policy=json.dumps({
        "rules": [
            {
                "rulePriority": 1,
                "description": "Keep last 5 images",
                "selection": {
                    "tagStatus": "any",
                    "countType": "imageCountMoreThan",
                    "countNumber": 5,
                },
                "action": {"type": "expire"},
            }
        ]
    }),
)
