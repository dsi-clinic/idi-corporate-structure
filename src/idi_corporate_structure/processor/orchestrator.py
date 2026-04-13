"""Pipeline Orchestrator - Runs the corporate structure pipeline for a specified input file.

Uses the `SubsidiaryPipeline` to process the input file and save the results to the output file.
GPT extraction is performed in a separate thread.

The orchestrator is responsible for running the pipeline.
"""

# Standard imports
import argparse
import datetime
import os

# Application imports
from idi_corporate_structure.common.api import SecClient
from idi_corporate_structure.processor.extractor import GptExtractor
from idi_corporate_structure.processor.pipeline import SubsidiaryPipeline
from idi_corporate_structure.processor.types import PipelineConfig


def get_args() -> argparse.Namespace:
    """Get command line arguments."""
    parser = argparse.ArgumentParser(description="Corporate Structure Pipeline Orchestrator")
    parser.add_argument("--input-file", type=str, required=True, help="Input file path")
    parser.add_argument("--output-file", type=str, required=True, help="Output file path")
    parser.add_argument("--failure-file", type=str, required=True, help="Failure file path")
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key (falls back to OPENAI_API_KEY env var)",
    )
    parser.add_argument("--rate-limit", type=float, default=0.2, help="Rate limit")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers")
    return parser.parse_args()


def main() -> None:
    """Main function to run the pipeline orchestrator."""
    start = datetime.datetime.now()
    args = get_args()

    if not args.openai_api_key:
        msg = "OpenAI API key required via --openai-api-key or OPENAI_API_KEY env var"
        raise SystemExit(msg)

    config = PipelineConfig(
        input_file=args.input_file,
        output_file=args.output_file,
        failure_file=args.failure_file,
        rate_limit=args.rate_limit,
        num_workers=args.num_workers,
        openai_api_key=args.openai_api_key,
    )
    sec_client = SecClient(rate_limit=config.rate_limit)
    extractor = GptExtractor(openai_api_key=config.openai_api_key)
    pipeline = SubsidiaryPipeline(config=config, sec_client=sec_client, extractor=extractor)
    pipeline.run()

    end = datetime.datetime.now()
    print(f"Elasped time: {end - start}")


if __name__ == "__main__":
    main()
