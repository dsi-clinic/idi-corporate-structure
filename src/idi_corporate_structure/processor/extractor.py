"""Extractors for parsing subsidiary data from SEC exhibit documents."""

# Standard imports
from abc import ABC, abstractmethod

# Application imports
from idi_corporate_structure.processor.types import Filing, Subsidiary


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

    def extract(self, filing: Filing, document: dict) -> list[Subsidiary]:
        """Use GPT to extract structured subsidiary data from a single document."""
        ## TODO: Implement GPT extraction
        return [
            Subsidiary(
                parent_cik=filing.cik,
                name="",
                location="",
                filing_date=filing.filing_date,
                form_type=filing.form_type,
                accession_number=filing.accession_number,
                exhibit_url=document["url"],
            )
        ]
