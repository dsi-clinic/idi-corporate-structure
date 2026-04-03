"""Provides loggers for use across the application."""

# Standard library imports
import datetime
import logging
import os

# Third party imports
import boto3
import requests
import tqdm
import watchtower

_configured_loggers: set[str] = set()

_EXECUTION_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

EC2_METADATA_BASE = "http://169.254.169.254"
EC2_METADATA_TOKEN_URL = f"{EC2_METADATA_BASE}/latest/api/token"
EC2_METADATA_INSTANCE_ID_URL = f"{EC2_METADATA_BASE}/latest/meta-data/instance-id"

LOG_GROUP_NAME = "idi-ftm2j"
LOG_STREAM_PREFIX = "/company-info"
LOG_RETENTION_DAYS = (
    30  # Possible values are: 1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, ...
)


def _get_instance_id() -> str:
    """Returns the EC2 instance ID when available, otherwise a fallback identifier."""
    # Prefer explicit env var (e.g. when running in Docker where metadata may be unreachable)
    if instance_id := os.environ.get("INSTANCE_ID"):
        return instance_id
    try:
        # IMDSv2: obtain session token first (required when IMDSv2 is enforced)
        token_resp = requests.put(
            EC2_METADATA_TOKEN_URL,
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
            timeout=1,
        )
        token_resp.raise_for_status()
        token = token_resp.text.strip()

        # Fetch instance-id with token
        instance_resp = requests.get(
            EC2_METADATA_INSTANCE_ID_URL,
            headers={"X-aws-ec2-metadata-token": token},
            timeout=1,
        )
        instance_resp.raise_for_status()
        return instance_resp.text.strip()

    except Exception:
        hostname = os.environ.get("HOSTNAME", "unknown")
        return hostname.split(".")[0]


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that writes via tqdm.write() to avoid disrupting progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Creates a logger with the given name and level.

    Attaches a stream handler that prints logs in a
    standard format to the console.

    Args:
        name: The logger name.

        level: The initial level. Defaults to 20 ("INFO").

    Returns:
        The logger.
    """
    # Check if logger has already been configured
    if name in _configured_loggers:
        return logging.getLogger(name)

    # Create logger and set level
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent log messages from being propagated to the root logger

    # Create console handler and set level
    ch = TqdmLoggingHandler()
    ch.setLevel(level)

    # Create formatter and add to handler
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(format)
    ch.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(ch)

    # Configure CloudWatch logging if executing on AWS EC2 instance
    _configure_cloudwatch(logger, name)

    # Add logger to set of configured loggers
    _configured_loggers.add(name)

    return logger


def _configure_cloudwatch(logger: logging.Logger, name: str) -> None:
    """Configures the logger to send logs to CloudWatch if executing in AWS.

    Enables CloudWatch when:
    - EC2 metadata endpoint is reachable, or
    - CLOUDWATCH_LOGS_ENABLED=true (e.g. when running in Docker on EC2).

    Args:
        logger: The logger to configure.
        name: The name of the logger.
    """
    # Enable when explicitly requested (e.g. Docker on EC2)
    env_enabled = os.environ.get("CLOUDWATCH_LOGS_ENABLED", "").lower() in ("true", "1", "yes")

    if env_enabled:
        instance_id = _get_instance_id()
        log_group_name = LOG_GROUP_NAME
        log_stream_name = f"{LOG_STREAM_PREFIX}/{instance_id}/{_EXECUTION_ID}"

        if "AWS_REGION" in os.environ:
            logs_client = boto3.client("logs", region_name=os.environ["AWS_REGION"])
        else:
            logs_client = boto3.client("logs")

        handler = watchtower.CloudWatchLogHandler(
            log_group_name=log_group_name,
            log_stream_name=log_stream_name,
            use_queues=False,
            boto3_client=logs_client,
            log_group_retention_days=LOG_RETENTION_DAYS,
        )

        format = "%(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(format)
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.info(
            "CloudWatch logging enabled: name=%s, log_group=%s log_stream=%s",
            name,
            log_group_name,
            log_stream_name,
        )
