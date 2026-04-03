"""Extractors for parsing subsidiary data from SEC exhibit documents."""

# Standard imports
import json
import os
from abc import ABC, abstractmethod

# Application imports
from idi_corporate_structure.common.api import OpenAiClient
from idi_corporate_structure.processor.types import Filing, Subsidiary


class DocumentError(Exception):
    """Exception raised for document-specific errors."""
    pass


class Extractor(ABC):
    """Interface for extracting subsidiaries from a single exhibit document."""

    @abstractmethod
    def extract(self, filing: Filing, document: dict) -> list[Subsidiary]:
        """Extract subsidiaries from a single exhibit document.

        Args:
            filing: The filing the document belongs to.
            document: Dict with 'url' and 'data' keys for the exhibit content.

        Returns:
            List of extracted Subsidiary objects.
        """
        ...


class GptExtractor(Extractor):
    """Extracts subsidiaries from a single exhibit document using GPT."""

    _OPENAI_CLIENT = OpenAiClient(api_key=os.environ.get("OPENAI_API_KEY", ""))
    _SYSTEM_PROMPT = """
    Given a table of a company's subsidiaries (in Markdown or raw text, previously converted from PDF), format them as a JSON, like

    ```json
    {
    "subsidiaries": [
        {"name": "XXX", "in": YYY}
    ]
    }
    ```

    objects, where `"XXX"` is the name of the subsidiary and `YYY` is the place of incorporation or other location, or `null` if not provided.

    Include all of the subsidiaries, but ignore any nested structure and ignore any data unrelated to subsidiaries.
    """.strip()

    def _get_request_data_json(self, document: str) -> dict:
        return {
            "model": "gpt-4.1-nano",
            "messages": [
                {
                    "role": "system",
                    "content": self._SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": document,
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "list_of_subsidiaries",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "subsidiaries": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "in": {"type": ["string", "null"]},
                                    },
                                    "required": ["name", "in"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "additionalProperties": False,
                    },
                },
            },
        }

    def _summarize(self, document: str) -> str:
        """Summarize the document using GPT.

        Args:
            document: The document to summarize.

        Returns:
            A dictionary of the summarized document.
        """
        post_data = self._get_request_data_json(document)
        response = self._OPENAI_CLIENT.query_endpoint(post_data)

        if "error" in response:
            if response.get("status_code") == 400:
                raise DocumentError(response["error"])
            raise RuntimeError(response["error"])

        content = response["data"]["choices"][0]["message"]["content"]
        return json.loads(content)

    def extract(self, filing: Filing, document: dict) -> list[Subsidiary]:
        """Use GPT to extract structured subsidiary data from a single document."""
        summary = self._summarize(document["data"])
        return [
            Subsidiary(
                parent_cik=filing.cik,
                parent_name=filing.company_name,
                parent_location=filing.location,
                filing_date=filing.filing_date,
                form_type=filing.form_type,
                accession_number=filing.accession_number,
                exhibit_url=document["url"],
                name=sub["name"],
                location=sub.get("in") or "",
            )
            for sub in summary["subsidiaries"]
        ]
