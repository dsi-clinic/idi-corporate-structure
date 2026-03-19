
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

    def extract(self, documents: list, filing: Filing) -> list[Subsidiary]:
        """Use GPT to extract structured subsidiary data."""
        pass