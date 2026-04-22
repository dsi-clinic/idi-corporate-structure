"""ECS cluster and Fargate task definition for the corporate structure processor."""

import json

import pulumi_aws as aws

import pulumi

from . import config, ecr, iam, secrets

# -----------------------------------------------------------------------------
# ECS Cluster (Fargate only)
# -----------------------------------------------------------------------------
cluster = aws.ecs.Cluster(
    "idi-ecs-cluster",
    name=f"{config.name_prefix}-cluster",
    settings=[
        aws.ecs.ClusterSettingArgs(
            name="containerInsights",
            value="enabled",
        )
    ],
    tags=config.tags(),
)

# -----------------------------------------------------------------------------
# CloudWatch Log Group for awslogs driver
# -----------------------------------------------------------------------------
log_group = aws.cloudwatch.LogGroup(
    "idi-ecs-log-group",
    name=f"/ecs/{config.name_prefix}",
    retention_in_days=config.log_retention_days,
    tags=config.tags(),
)

# -----------------------------------------------------------------------------
# Task Definition
# -----------------------------------------------------------------------------
cpu = config.config.get("cpu") or "1024"
memory = config.config.get("memory") or "4096"
rate_limit = config.config.get("rate_limit") or "0.2"
num_workers = config.config.get("num_workers") or "10"
stale_threshold_days = config.config.get("stale_threshold_days") or "30"
input_sample_size = config.config.get("input_sample_size") or "0"

# Build S3 paths from externally managed bucket (name from config)
input_file = config.config.require("input_file")
output_file = f"s3://{config.bucket_name}/{config.app_name}/output/subsidiaries.parquet"
failure_file = f"s3://{config.bucket_name}/{config.app_name}/failures/failures.json"

# Container definition as JSON (required by aws.ecs.TaskDefinition)
container_definitions = pulumi.Output.all(
    image=ecr.orchestrator_image,
    log_group_name=log_group.name,
    region=config.aws_region,
    secret_arn=secrets.openai_secret.arn,
).apply(
    lambda args: json.dumps(
        [
            {
                "name": "corporate-structure-orchestrator",
                "image": args["image"],
                "essential": True,
                "command": [
                    "--input-file",
                    input_file,
                    "--output-file",
                    output_file,
                    "--failure-file",
                    failure_file,
                    "--rate-limit",
                    rate_limit,
                    "--num-workers",
                    num_workers,
                    "--stale-threshold-days",
                    stale_threshold_days,
                ],
                "environment": [
                    {"name": "AWS_REGION", "value": args["region"]},
                    {"name": "CLOUDWATCH_LOGS_ENABLED", "value": "false"},
                    {"name": "INPUT_SAMPLE_SIZE", "value": input_sample_size},
                    {"name": "PYTHONUNBUFFERED", "value": "1"},
                ],
                "secrets": [
                    {
                        "name": "OPENAI_API_KEY",
                        "valueFrom": args["secret_arn"],
                    }
                ],
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": args["log_group_name"],
                        "awslogs-region": args["region"],
                        "awslogs-stream-prefix": "orchestrator",
                    },
                },
                "stopTimeout": 30,
            }
        ]
    )
)

task_definition = aws.ecs.TaskDefinition(
    "idi-ecs-task-definition",
    family=f"{config.name_prefix}",
    requires_compatibilities=["FARGATE"],
    network_mode="awsvpc",
    cpu=cpu,
    memory=memory,
    execution_role_arn=iam.task_execution_role.arn,
    task_role_arn=iam.task_role.arn,
    container_definitions=container_definitions,
    tags=config.tags(),
)
