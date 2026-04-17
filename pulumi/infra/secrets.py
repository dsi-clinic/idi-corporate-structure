"""AWS Secrets Manager resources for the OpenAI API key."""

import pulumi_aws as aws

import pulumi

from . import config

# -----------------------------------------------------------------------------
# Config (required — Pulumi fails at deploy time if missing)
# -----------------------------------------------------------------------------
openai_api_key = config.config.require_secret("openai_api_key")

# -----------------------------------------------------------------------------
# Secret
# -----------------------------------------------------------------------------
openai_secret = aws.secretsmanager.Secret(
    "idi-secret-openai-api-key",
    name=f"{config.name_prefix}-openai-api-key",
    description="OpenAI API key for GPT-based subsidiary extraction",
    tags=config.tags(),
)

openai_secret_version = aws.secretsmanager.SecretVersion(
    "idi-secret-version-openai",
    secret_id=openai_secret.id,
    secret_string=openai_api_key,
    opts=pulumi.ResourceOptions(
        depends_on=[openai_secret],
        ignore_changes=["secret_string"],
    ),
)
