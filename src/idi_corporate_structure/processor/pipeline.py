"""Pipeline for extracting subsidiary data from SEC 10-K Exhibit 21 filings."""

# Standard application imports
import json
import queue
import re
import threading
import zipfile
from abc import ABC, abstractmethod

# Third party imports
from tqdm import tqdm

# Application imports
from idi_corporate_structure.common.api import SecClient
from idi_corporate_structure.common.failures import FailureRegistry
from idi_corporate_structure.common.logs import get_logger
from idi_corporate_structure.common.storage import open_zip
from idi_corporate_structure.processor.extractor import GptExtractor
from idi_corporate_structure.processor.failures import (
    CorporateStructureFailureClassifier,
    FailureType,
)
from idi_corporate_structure.processor.types import (
    Filing,
    PipelineConfig,
    PipelineStats,
    Subsidiary,
)


class Pipeline(ABC):
    """Baseline class for processing piplines."""

    def __init__(
        self, config: PipelineConfig, sec_client: SecClient, extractor: GptExtractor
    ) -> None:
        """Initialize the pipeline with config, SEC client, and extractor."""
        self.config = config
        self.extractor = extractor
        self.logger = get_logger("Pipeline")
        self.sec_client = sec_client
        self.stats = PipelineStats()

    @abstractmethod
    def load_input(self) -> list:
        """Load input data and return a list of items to process."""
        ...

    @abstractmethod
    def process(self, input_list: list) -> list:
        """Process each item in the input list and return a list of results."""
        ...

    @abstractmethod
    def save_output(self, processed_list: list) -> None:
        """Saves processed items list."""
        ...

    @abstractmethod
    def display_stats(self) -> None:
        """Display processing stats."""

    def run(self) -> None:
        """Run the pipeline."""
        input_data = self.load_input()
        self.logger.info("Located %d input data items.", len(input_data))

        results = self.process(input_data)
        self.logger.info("Located %d result data items.", len(results))
        for r in results:
            print(r)

        self.save_output(results)
        self.display_stats()


class SubsidiaryPipeline(Pipeline):
    """Pipeline that fetches Exhibit 21 filings from SEC EDGAR and extracts subsidiary data."""

    EX = re.compile(r"\BEX")
    IS_10K = re.compile("10-?K")
    IS_DATE = re.compile("[0-9]{4}-[0-9]{2}-[0-9]{2}")
    TWENTYONE = re.compile("[^0-9]21")

    _INPUT_SAMPLE_SIZE = 10  # TODO: Remove after done testing

    def __init__(
        self, config: PipelineConfig, sec_client: SecClient, extractor: GptExtractor
    ) -> None:
        """Initialize the subsidiary pipeline with failure registry."""
        super().__init__(config, sec_client, extractor)
        self.failure_registry = FailureRegistry(
            config.failure_file,
            classifier=CorporateStructureFailureClassifier(),
            flush_every=config.failure_flush_every,
        )
        self.rows = []

    def _zip_file_data(self, data: dict, filename: str, cik: str) -> zip | None:
        """Locate and returned zipped file data.

        Args:
            data: Data to retrieve specific file data from
            filename: Name of file data was retrieved from
            cik: Identifier of file data was retrieved from
        """
        forms = data.get("filings", {}).get("recent", {}).get("form", [])
        accession_numbers = data.get("filings", {}).get("recent", {}).get("accessionNumber", [])
        primary_documents = data.get("filings", {}).get("recent", {}).get("primaryDocument", [])
        filing_dates = data.get("filings", {}).get("recent", {}).get("filingDate", [])

        if (
            len({len(forms), len(accession_numbers), len(primary_documents), len(filing_dates)})
            != 1
        ):
            self.logger.error("Filename: %s has forms with mismatched data lengths.", filename)
            self.stats.increment("failed_filings")
            self.failure_registry.add(cik, filename, failure_type=FailureType.MISMATCHED_LENGTHS)
            return None

        if not any([forms, accession_numbers, primary_documents, filing_dates]):
            self.logger.debug("Filename: %s has forms without data.", filename)
            self.stats.increment("failed_filings")
            self.failure_registry.add(cik, filename, failure_type=FailureType.NO_FORM_DATA)
            return None

        return zip(forms, accession_numbers, primary_documents, filing_dates)

    def _create_filing(
        self, accession_number: str, primary_document: str, cik: str, filing_date: str, form: str
    ) -> Filing:
        """Create Filing object with form data.

        Args:
            accession_number: String taken from form data
            primary_document: String document URL from form data
            cik: String CIK identifier
            filing_date: String date of filing
            form: String name of form
        """
        accession = accession_number.replace("-", "")
        directory = f"{self.sec_client.SEC_URL}/{cik}/{accession}/index.json"

        if primary_document != "" and primary_document.split(".")[-1].upper() in ("HTM", "HTML"):
            primary = f"{self.sec_client.SEC_URL}/{cik}/{accession}/{primary_document}"
        else:
            primary = ""

        filing = Filing(
            cik=cik,
            filing_date=filing_date,
            form_type=form,
            accession_number=accession_number,
            directory=directory,
            primary_document=primary,
        )
        return filing

    def _parse_file(self, zf: zipfile.ZipFile, filename: str) -> list[Filing]:
        """Parse file contents and save as a row in the rows list.

        Parses: cik, date, form, accession number, directory, and primary document

        Args:
            zf: Open ZipFile to read from.
            filename: The filename to parse data from.

        Returns:
            List of Filing objects with form data
        """
        filings = []
        if filename.startswith("CIK") and filename.endswith(".json"):
            with zf.open(filename) as file:
                data = json.load(file)

            cik = filename[3:-5]
            data_zip = self._zip_file_data(data, filename, cik)

            if data_zip:
                for form, accession_number, primary_document, filing_date in data_zip:
                    if self.IS_10K.match(form):
                        filings.append(
                            self._create_filing(
                                accession_number=accession_number,
                                primary_document=primary_document,
                                cik=cik,
                                filing_date=filing_date,
                                form=form,
                            )
                        )
                        self.stats.increment("total_filing")

                    else:
                        self.logger.debug("Filename: %s does not have a 10K form.", filename)
                        self.stats.increment("failed_filings")
                        self.failure_registry.add(
                            cik, filename, failure_type=FailureType.NO_10K_FILINGS
                        )

        return filings

    def load_input(self) -> list[Filing]:
        """Load input data from the SEC and return a list of filings.

        Returns:
            A list of Filing objects
        """
        filings = []
        with open_zip(self.config.input_file, headers=self.sec_client.SEC_HEADERS) as zf:
            namelist = zf.namelist()
            self.logger.info("Total # of files to process: %d", len(namelist))

            count = 0
            for filename in tqdm(namelist):
                filings.extend(self._parse_file(zf, filename))
                count += 1
                if count == self._INPUT_SAMPLE_SIZE:
                    break

        return filings

    def _extract_worker(self, work_queue: queue.Queue, results_queue: queue.Queue) -> None:
        """Extract worker for the pipeline.

        Args:
            work_queue: Queue of filings and exhibit contents
            results_queue: Queue of results
        """
        while True:
            filing, exhibit_contents = work_queue.get()
            try:
                subsidiaries = self.extractor.extract(filing, exhibit_contents)
                results_queue.put(subsidiaries)

            except Exception as _:
                self.logger.error(
                    "Error extracting subsidiaries from filing: %s - %s - %s",
                    filing.cik,
                    filing.accession_number,
                    filing.filing_date,
                )
                self.stats.increment("failed_subsidiaries")
                self.failure_registry.add(
                    filing.cik, filing.accession_number, FailureType.EXTRACTION_FAILED
                )

            finally:
                work_queue.task_done()

    def _results_worker(self, results_queue: queue.Queue, subsidiaries: list[Subsidiary]) -> None:
        """Results worker for the pipeline.

        Args:
            results_queue: Queue of results
            subsidiaries: List of subsidiaries to add to
        """
        while True:
            result = results_queue.get()
            subsidiaries.extend(result)
            self.stats.increment("total_subsidiaries", len(result))
            results_queue.task_done()

    def _fetch_directory(self, filing: Filing) -> list[dict]:
        """Fetch directory data from the SEC.

        Args:
            filing: Filing object to fetch directory data from

        Returns:
            Directory response items
        """
        directory_response = self.sec_client.query_endpoint(filing.directory)
        if "directory" not in directory_response.get("data", {}).keys():
            self.logger.error(
                "Filing: %s - %s - %s does not have a directory listing.",
                filing.cik,
                filing.accession_number,
                filing.filing_date,
            )
            self.stats.increment("failed_subsidiaries")
            self.failure_registry.add(
                filing.cik, filing.accession_number, FailureType.NO_FILING_DIRECTORY
            )
            return {}
        self.sec_client.rate_limit()
        return directory_response.get("data", {}).get("directory", {}).get("item", [])

    def _fetch_exhibit_content(self, filing: Filing, item: dict) -> dict:
        """Fetch exhibit content from the SEC.

        Args:
            filing: Filing object to fetch exhibit content from
            item: Item object to fetch exhibit content from

        Returns:
            Dict of SEC response that containsexhibit content
        """
        name = item["name"].upper()
        accession = filing.accession_number.replace("-", "")

        exhibit_content = {}
        if (
            self.EX.search(name) and (name.startswith("21") or self.TWENTYONE.search(name))
        ) or "SUB" in name:
            sec_url = f"{self.sec_client.SEC_URL}/{filing.cik}/{accession}/{item['name']}"
            item_response = self.sec_client.query_endpoint(sec_url=sec_url, return_json=False)
            if item_response:
                exhibit_content = item_response

            else:
                self.logger.error(
                    "Exhibit    : %s - %s - %s does not have content.",
                    name,
                    filing.cik,
                    filing.accession_number,
                    filing.filing_date,
                )
                self.stats.increment("failed_subsidiaries")
                self.failure_registry.add(
                    filing.cik, filing.accession_number, FailureType.NO_EXHIBIT_CONTENT
                )

            self.sec_client.rate_limit()
        return exhibit_content

    def _fetch_exhibit(self, filing: Filing) -> list[str]:
        """Fetch exhibit data from the SEC.

        Args:
            filing: Filing object to fetch exhibit data from

        Returns:
            List of Strings of exhibit content
        """
        directory_items = self._fetch_directory(filing)
        exhibit_content = []
        for item in directory_items:
            exhibits = self._fetch_exhibit_content(filing, item)
            if exhibits:
                exhibit_content.append(exhibits)
        return exhibit_content

    def process(self, input_list: list[Filing]) -> list[Subsidiary]:
        """Process input filing list to retrieve subsidiary data."""
        # Queues to store exhibit data and subsidiary data
        work_queue = queue.Queue(maxsize=self.config.num_workers * 2)
        result_queue = queue.Queue()
        subsidiaries = []

        # Start extract and results workers
        extract_workers = [
            threading.Thread(
                target=self._extract_worker,
                args=(work_queue, result_queue),
                daemon=True,
                name=f"extract-worker-{i}",
            )
            for i in range(self.config.num_workers)
        ]
        for worker in extract_workers:
            worker.start()

        results_worker = threading.Thread(
            target=self._results_worker,
            args=(result_queue, subsidiaries),
            daemon=True,
            name="results-worker",
        )
        results_worker.start()

        # SEC operations to fetch exhibit data
        for filing in tqdm(input_list):
            exhibit_contents = self._fetch_exhibit(filing)
            work_queue.put((filing, exhibit_contents))

        # Shutdown workers as all work is done
        work_queue.join()
        result_queue.join()

        return subsidiaries

    def save_output(self, processed_list: list[Subsidiary]) -> None:
        """Save subsidiary list output."""

    def display_stats(self) -> None:
        """Log pipeline stats on completion."""
        self.logger.info(
            "Stats: total_filings=%d failed_filings=%d total_subsidiaries=%d failed_subsidiaries=%d",
            self.stats.total_filing,
            self.stats.failed_filings,
            self.stats.total_subsidiaries,
            self.stats.failed_subsidiaries,
        )

    def run(self) -> None:
        """Run the pipeline, flushing any buffered failures on completion."""
        try:
            super().run()
        finally:
            self.failure_registry.flush()


if __name__ == "__main__":
    # uv run python3 -m src.idi_corporate_structure.processor.pipeline
    import datetime

    start = datetime.datetime.now()

    config = PipelineConfig(
        # input_file="https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip",
        input_file="/Users/ntebaldi/Documents/workspace/11hour/ftm2j/data/corporate-struct/input/submissions.zip",
        failure_file="/Users/ntebaldi/Documents/workspace/11hour/ftm2j/data/corporate-struct/failures/failures.json",
        rate_limit=0.12,
        num_workers=10,
    )

    sec_client = SecClient(config.rate_limit)

    extractor = GptExtractor()

    sub_pipeline = SubsidiaryPipeline(config=config, sec_client=sec_client, extractor=extractor)

    sub_pipeline.run()

    end = datetime.datetime.now()
    print(f"Elasped time: {end - start}")
