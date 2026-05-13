"""AWS Secrets Manager resources for pipeline secrets."""

import pulumi_aws as aws

import pulumi

from . import config

# -----------------------------------------------------------------------------
# Config (required — Pulumi fails at deploy time if missing)
# -----------------------------------------------------------------------------
openai_api_key = config.config.require_secret("openai_api_key")
sec_user_agent = config.config.require_secret("sec_user_agent")

# -----------------------------------------------------------------------------
# Secrets
# -----------------------------------------------------------------------------
openai_secret = aws.secretsmanager.Secret(
    "idi-secret-openai-api-key",
    name=f"{config.name_prefix}-openai-api-key",
    recovery_window_in_days=0,
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

sec_user_agent_secret = aws.secretsmanager.Secret(
    "idi-secret-sec-user-agent",
    name=f"{config.name_prefix}-sec-user-agent",
    description="SEC EDGAR User-Agent header value (Name email@example.com)",
    recovery_window_in_days=0,
    tags=config.tags(),
)

sec_user_agent_secret_version = aws.secretsmanager.SecretVersion(
    "idi-secret-version-sec-user-agent",
    secret_id=sec_user_agent_secret.id,
    secret_string=sec_user_agent,
    opts=pulumi.ResourceOptions(
        depends_on=[sec_user_agent_secret],
        ignore_changes=["secret_string"],
    ),
)
