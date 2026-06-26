"""Genuine secrets as SSM Parameter Store SecureString parameters.

The processor's one real credential — the OpenAI API key — is stored as an SSM
`SecureString` rather than in Secrets Manager or Pulumi config. This keeps the
value out of git and out of Pulumi state (these repos are public): Pulumi only
declares the parameter with a placeholder and `ignore_changes=["value"]`, so the
resource + IAM are codified while the real value is set out-of-band:

    aws ssm put-parameter --overwrite --type SecureString \
      --name /idi/<stack>/<app>/secrets/openai_api_key --value '<real key>'

The ECS task definition injects it by ARN via `secrets:`, so the value never
touches CI logs or state. Rotation is a `put-parameter --overwrite`, picked up
at the next task launch — no deploy.

(`sec_user_agent` is NOT here — it's a public SEC contact string, kept in
committed config and injected as a plain env var.)
"""

import pulumi_aws as aws

import pulumi

from . import config

_secrets_prefix = f"/idi/{config.stack_name}/{config.app_name}/secrets"

openai_api_key_param = aws.ssm.Parameter(
    "idi-ssm-secret-openai-api-key",
    name=f"{_secrets_prefix}/openai_api_key",
    type="SecureString",
    # Placeholder only — the real value is set out-of-band and never managed by
    # Pulumi (hence ignore_changes), so it stays out of git and state.
    value="PLACEHOLDER-set-via-aws-ssm-put-parameter",
    description="OpenAI API key (real value set out-of-band).",
    tags=config.tags(),
    opts=pulumi.ResourceOptions(ignore_changes=["value"]),
)
