"""Extractors for parsing subsidiary data from SEC exhibit documents."""

# Standard imports
from abc import ABC, abstractmethod

# Application imports
from idi_corporate_structure.processor.types import Filing, Subsidiary


class Extractor(ABC):
    """Interface for extracing subsidiaries from documents."""

    @abstractmethod
    def extract(self, documents: list, filing: Filing) -> list[Subsidiary]:
        """Extact subsidiaries from documents."""
        ...


class GptExtractor(Extractor):
    """Extracts subsidiaries using GPT."""

    def extract(self, filing: Filing, documents: list[str]) -> list[Subsidiary]:
        """Use GPT to extract structured subsidiary data."""
        ## TODO: Implement GPT extraction

        subsidiaries = []
        for document in documents:
            subsidiaries.append(
                Subsidiary(
                    parent_cik=filing.cik,
                    name="",
                    location="",
                    filing_date=filing.filing_date,
                    form_type=filing.form_type,
                    accession_number=filing.accession_number,
                    exhibit_url=document["url"],
                )
            )

        return subsidiaries
