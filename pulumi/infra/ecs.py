"""ECS cluster and Fargate task definition for the corporate structure processor.

Per-pipeline arguments (input file, type, batch size, etc.) are injected by the
EventBridge schedules via ECS containerOverrides — see scheduling.py. The task
definition's baseline command is `--help` so a misconfigured override fails
loudly instead of silently running a default pipeline.
"""

import json

import pulumi_aws as aws

import pulumi

from . import config, ecr, iam, logs, secrets

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
# Task Definition
# -----------------------------------------------------------------------------
CONTAINER_NAME = "corporate-structure-orchestrator"

cpu = config.config.get("cpu") or "1024"
memory = config.config.get("memory") or "4096"
rate_limit = config.config.get("rate_limit") or "0.2"
num_workers = config.config.get("num_workers") or "10"
input_sample_size = config.config.get("input_sample_size") or "0"
openai_model = config.config.get("openai_model") or "gpt-4.1-nano"

# Build S3 paths from externally managed bucket (name from config)
input_file = config.config.require("input_file")
output_file = f"s3://{config.bucket_name}/{config.app_name}/output/subsidiaries.parquet"
failure_file = f"s3://{config.bucket_name}/{config.app_name}/failures/failures.json"

# Container definition as JSON (required by aws.ecs.TaskDefinition)
container_definitions = pulumi.Output.all(
    image=ecr.orchestrator_image,
    log_group_name=logs.log_group.name,
    region=config.aws_region,
    openai_secret_arn=secrets.openai_api_key_param.arn,
).apply(
    lambda args: json.dumps(
        [
            {
                "name": CONTAINER_NAME,
                "image": args["image"],
                "essential": True,
                "command": ["--help"],
                "environment": [
                    {"name": "AWS_REGION", "value": args["region"]},
                    {"name": "CLOUDWATCH_LOGS_ENABLED", "value": "false"},
                    {"name": "INPUT_SAMPLE_SIZE", "value": input_sample_size},
                    {"name": "PYTHONUNBUFFERED", "value": "1"},
                    # Public SEC contact string (committed config), not a secret.
                    {"name": "SEC_USER_AGENT", "value": config.sec_user_agent},
                ],
                "secrets": [
                    {
                        "name": "OPENAI_API_KEY",
                        "valueFrom": args["openai_secret_arn"],
                    },
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
