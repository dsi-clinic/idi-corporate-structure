"""Pipeline for extracting subsidiary data from SEC 10-K Exhibit 21 filings."""

# Standard application imports
import dataclasses
import datetime
import io
import json
import os
import queue
import re
import threading
import zipfile
from abc import ABC, abstractmethod

# Third party imports
import pandas as pd
import pdfplumber
from idi_ftm2j_shared.failures import FailureRegistry
from idi_ftm2j_shared.logs import get_logger
from idi_ftm2j_shared.storage import open_zip
from tqdm import tqdm

# Application imports
from idi_corporate_structure.api import SecClient
from idi_corporate_structure.extractor import (
    DocumentError,
    ExtractionTimeoutError,
    ExtractionTruncatedError,
    GptExtractor,
    html_to_text,
)
from idi_corporate_structure.failures import (
    CorporateStructureFailureClassifier,
    FailureType,
)
from idi_corporate_structure.normalization import (
    normalize_parent_location,
    normalize_subsidiary_location,
)
from idi_corporate_structure.types import (
    SUPPORTED_EXHIBIT_EXTENSIONS,
    Filing,
    PipelineConfig,
    PipelineStats,
    Subsidiary,
)


class Pipeline(ABC):
    """Baseline class for processing piplines."""

    def __init__(
        self,
        config: PipelineConfig,
        sec_client: SecClient,
        extractor: GptExtractor,
    ) -> None:
        """Initialize the pipeline with config, SEC client, and extractor.

        Args:
            config: Pipeline configuration including input/output paths and tuning
                parameters.
            sec_client: Configured SEC EDGAR API client used for fetching filings.
            extractor: Extractor instance responsible for parsing subsidiary data
                from exhibit documents.
        """
        self.config = config
        self.extractor = extractor
        self.sec_client = sec_client
        self.stats = PipelineStats()
        self.logger = get_logger(type(self).__name__)

    @abstractmethod
    def load_input(self) -> list:
        """Load input data and return a list of items to process.

        Returns:
            List of input items. The concrete element type is defined by each
            subclass (e.g. ``list[Filing]``).
        """
        ...

    @abstractmethod
    def process(self, input_list: list) -> list:
        """Process each item in the input list and return a list of results.

        Args:
            input_list: Items returned by :meth:`load_input`.

        Returns:
            List of processed results. The concrete element type is defined by
            each subclass (e.g. ``list[Subsidiary]``).
        """
        ...

    @abstractmethod
    def save_output(self, processed_list: list) -> None:
        """Persist the processed results to the configured output destination.

        Args:
            processed_list: Items returned by :meth:`process`.

        Returns:
            None
        """
        ...

    @abstractmethod
    def display_stats(self) -> None:
        """Log or display a summary of pipeline processing statistics.

        Returns:
            None
        """

    def run(self) -> None:
        """Execute the full pipeline: load → process → save → display stats.

        Calls :meth:`load_input`, :meth:`process`, :meth:`save_output`, and
        :meth:`display_stats` in sequence, then logs the total elapsed time.

        Returns:
            None
        """
        start_time = datetime.datetime.now()

        input_data = self.load_input()

        if input_data:
            results = self.process(input_data)
            self.save_output(results)
            self.display_stats()
        else:
            self.logger.info("No input data found, skipping pipeline")

        end_time = datetime.datetime.now()
        self.logger.info("Elasped time: %s", end_time - start_time)


class SubsidiaryPipeline(Pipeline):
    """Pipeline that fetches Exhibit 21 filings from SEC EDGAR and extracts subsidiary data."""

    EX = re.compile(r"EX[-\d]", re.IGNORECASE)
    IS_10K = re.compile("10-?K")
    IS_20F = re.compile("20-?F")
    IS_DATE = re.compile("[0-9]{4}-[0-9]{2}-[0-9]{2}")
    TWENTYONE = re.compile("[^0-9]21")
    EIGHT = re.compile("[^0-9]8")
    IS_OVERFLOW = re.compile(r"-submissions-\d+\.json$")

    _INPUT_SAMPLE_SIZE = int(os.environ.get("INPUT_SAMPLE_SIZE", 0))

    def __init__(
        self, config: PipelineConfig, sec_client: SecClient, extractor: GptExtractor
    ) -> None:
        """Initialize the subsidiary pipeline with failure registry.

        Args:
            config: Pipeline configuration including input/output paths, rate limit,
                worker count, and failure flush threshold.
            sec_client: Configured SEC EDGAR API client.
            extractor: Extractor used to parse subsidiary data from exhibit documents.
        """
        super().__init__(config, sec_client, extractor)
        self.failure_registry = FailureRegistry(
            config.failure_file,
            classifier=CorporateStructureFailureClassifier(),
            flush_every=config.failure_flush_every,
        )
        self.rows = []

    def _retrieve_overflow_filings(
        self, data: dict, cik: str, zf: zipfile.ZipFile
    ) -> tuple[list, list, list, list]:
        """Retrieve overflow filings from the SEC submissions JSON.

        Args:
            data: Data to retrieve overflow filings from
            cik: Identifier of file data was retrieved from
            zf: Open ZipFile to read from.

        Returns:
            Tuple of lists of forms, accession numbers, primary documents, and filing dates.
        """
        forms = []
        accession_numbers = []
        primary_documents = []
        filing_dates = []
        for entry in data.get("filings", {}).get("files", []):
            overflow_filename = entry.get("name", "")
            if not overflow_filename:
                continue

            if (cik, overflow_filename) in self.failure_registry:
                self.logger.debug(
                    "Skipping overflow file — permanent failure recorded: %s", overflow_filename
                )
                self.stats.increment("skipped_filings")
                continue

            try:
                with zf.open(overflow_filename) as of:
                    overflow = json.load(of)

                forms += overflow.get("form", [])
                accession_numbers += overflow.get("accessionNumber", [])
                primary_documents += overflow.get("primaryDocument", [])
                filing_dates += overflow.get("filingDate", [])

            except KeyError:
                self._record_failure(
                    (cik, overflow_filename),
                    FailureType.NO_OVERFLOW_FILINGS,
                    "error",
                    "Overflow file not found: %s",
                    overflow_filename,
                    stat_keys=("failed_filings",),
                )

        return forms, accession_numbers, primary_documents, filing_dates

    def _zip_file_data(
        self, data: dict, filename: str, cik: str, zf: zipfile.ZipFile
    ) -> zip | None:
        """Locate and return zipped filing data for a single CIK JSON file.

        Combines recent filings from ``data`` with any overflow filings referenced
        in ``data["filings"]["files"]``, then validates that all four parallel lists
        (forms, accession numbers, primary documents, filing dates) have the same
        length before zipping them together.

        Args:
            data: Parsed CIK submissions JSON with ``filings.recent`` and optionally
                ``filings.files`` overflow references.
            filename: Source filename, used for failure registry entries and logging.
            cik: CIK identifier for the company, used for failure registry entries.
            zf: Open :class:`zipfile.ZipFile` from which overflow files are read.

        Returns:
            A ``zip`` iterator of ``(form, accession_number, primary_document,
            filing_date)`` tuples, or ``None`` if the data is missing or inconsistent.
        """
        # Retrieve recent filings
        forms = data.get("filings", {}).get("recent", {}).get("form", [])
        accession_numbers = data.get("filings", {}).get("recent", {}).get("accessionNumber", [])
        primary_documents = data.get("filings", {}).get("recent", {}).get("primaryDocument", [])
        filing_dates = data.get("filings", {}).get("recent", {}).get("filingDate", [])

        # Retrieve overflow filings
        o_forms, o_accession_numbers, o_primary_documents, o_filing_dates = (
            self._retrieve_overflow_filings(data, cik, zf)
        )
        forms.extend(o_forms)
        accession_numbers.extend(o_accession_numbers)
        primary_documents.extend(o_primary_documents)
        filing_dates.extend(o_filing_dates)

        if (
            len({len(forms), len(accession_numbers), len(primary_documents), len(filing_dates)})
            != 1
        ):
            self._record_failure(
                (cik, filename),
                FailureType.MISMATCHED_LENGTHS,
                "error",
                "Filename: %s has forms with mismatched data lengths.",
                filename,
                stat_keys=("failed_filings",),
            )
            return None

        if not any([forms, accession_numbers, primary_documents, filing_dates]):
            self._record_failure(
                (cik, filename),
                FailureType.NO_FORM_DATA,
                "debug",
                "Filename: %s has forms without data.",
                filename,
                stat_keys=("failed_filings",),
            )
            return None

        return zip(forms, accession_numbers, primary_documents, filing_dates)

    def _process_filings_zip(
        self,
        filings_zip: zip,
        company_data: dict,
        source_filename: str,
    ) -> list[Filing]:
        """Iterate a filings zip and collect 10-K Filing objects.

        Args:
            filings_zip: zip of (form, accession_number, primary_document, filing_date) tuples.
            company_data: Dict with cik, name, location keys.
            source_filename: Filename used for failure registry entries and debug logging.

        Returns:
            List of Filing objects for 10-K forms only.
        """
        filings = []
        for form, accession_number, primary_document, filing_date in filings_zip:
            if self.IS_10K.match(form) or self.IS_20F.match(form):
                filings.append(
                    self._create_filing(
                        accession_number=accession_number,
                        primary_document=primary_document,
                        filing_date=filing_date,
                        form=form,
                        company_data=company_data,
                    )
                )
                self.stats.increment("total_filing")

            else:
                self.logger.debug("Filename: %s skipping non-10K form: %s.", source_filename, form)

        if not filings:
            self.stats.increment("failed_filings")
            self.failure_registry.add(
                (company_data["cik"], source_filename), failure_type=FailureType.NO_10K_FILINGS
            )

        return filings

    def _create_filing(
        self,
        accession_number: str,
        primary_document: str,
        filing_date: str,
        form: str,
        company_data: dict,
    ) -> Filing:
        """Create Filing object with form data.

        Args:
            accession_number: String taken from form data
            primary_document: String document URL from form data
            filing_date: String date of filing
            form: String name of form
            company_data: Dictionary with company data from SEC submissions JSON

        Returns:
            Filing object with form data
        """
        accession = accession_number.replace("-", "")
        directory = f"{self.sec_client.SEC_URL}/{company_data['cik']}/{accession}/index.json"

        if (
            primary_document != ""
            and primary_document.split(".")[-1].upper() in SUPPORTED_EXHIBIT_EXTENSIONS
        ):
            primary = (
                f"{self.sec_client.SEC_URL}/{company_data['cik']}/{accession}/{primary_document}"
            )
        else:
            primary = ""

        return Filing(
            cik=company_data["cik"],
            filing_date=filing_date,
            form_type=form,
            accession_number=accession_number,
            directory=directory,
            primary_document=primary,
            company_name=company_data["name"],
            location=company_data["location"],
            filename=company_data["filename"],
        )

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

        # Overflow files have no company metadata; loaded below with primary file
        if self.IS_OVERFLOW.search(filename):
            return filings

        if not (filename.startswith("CIK") and filename.endswith(".json")):
            return filings

        with zf.open(filename) as file:
            data = json.load(file)

        company_data = {
            "cik": data.get("cik", ""),
            "name": data.get("name", ""),
            "location": data.get("stateOfIncorporation", ""),
            "filename": filename,
        }

        # Skip files that had a permanent failure on a previous run
        if (company_data["cik"], filename) in self.failure_registry:
            self.logger.debug(
                "Skipping permanent failure for filing: %s - %s", company_data["cik"], filename
            )
            self.stats.increment("skipped_filings")
            return filings

        # Process filings
        filings_zip = self._zip_file_data(data, filename, company_data["cik"], zf)
        if filings_zip:
            filings.extend(self._process_filings_zip(filings_zip, company_data, filename))

        return filings

    def _filter_already_processed(self, filings: list[Filing]) -> list[Filing]:
        """Drop filings already represented in the output parquet.

        Reads the existing output parquet and removes any filing whose
        ``(parent_cik, accession_number)`` key is already present. Returns
        ``filings`` unchanged if no parquet exists yet.

        Args:
            filings: Filings returned by `load_input`.

        Returns:
            Filings that still need to be processed.
        """
        try:
            existing_df = pd.read_parquet(self.config.output_file)
        except FileNotFoundError:
            return filings

        if "parent_cik" not in existing_df.columns or "accession_number" not in existing_df.columns:
            return filings

        existing_keys = set(zip(existing_df["parent_cik"], existing_df["accession_number"]))
        unprocessed = [
            f
            for f in filings
            if (f.cik, f.accession_number) not in existing_keys
            and (f.cik, f.filename) not in self.failure_registry
        ]
        self.logger.info(
            "Filter: %d filings loaded, %d skipped as already processed, %d remaining",
            len(filings),
            len(filings) - len(unprocessed),
            len(unprocessed),
        )
        return unprocessed

    def load_input(self) -> list[Filing]:
        """Load input data from the SEC and return a list of filings.

        Returns:
            A list of Filing objects
        """
        filings = []
        with open_zip(self.config.input_file, headers=self.sec_client.sec_headers) as zf:
            namelist = zf.namelist()
            if self._INPUT_SAMPLE_SIZE:
                namelist = namelist[: self._INPUT_SAMPLE_SIZE]
            self.logger.info("Total # of files to process: %d", len(namelist))

            for filename in tqdm(namelist, desc="Retrieving filings"):
                filings_for_file = self._parse_file(zf, filename)
                if filings_for_file:
                    filings.extend(filings_for_file)

        self.logger.info(
            "Located %d filings for %d forms",
            len(filings),
            len(namelist) - self.stats.skipped_filings,
        )
        self.logger.info("Skipped %d files", self.stats.skipped_filings)

        return self._filter_already_processed(filings)

    def _record_failure(
        self,
        key: tuple[str, str],
        failure_type: FailureType,
        log_level: str,
        message: str,
        *log_args: object,
        stat_keys: tuple[str, ...] = ("failed_subsidiaries",),
    ) -> None:
        """Log a failure, increment stats, and register it in the failure registry.

        Args:
            key: Registry key tuple, typically ``(cik, filename)``.
            failure_type: Classified failure type.
            log_level: Logger method name (``"warning"`` or ``"error"``).
            message: ``%s``-style log message.
            *log_args: Arguments to substitute into ``message``.
            stat_keys: Stat field names to increment (default: ``("failed_subsidiaries",)``).
        """
        getattr(self.logger, log_level)(message, *log_args)
        for key_ in stat_keys:
            self.stats.increment(key_)
        self.failure_registry.add(key, failure_type)

    def _report_extraction(
        self,
        num_chunks: int,
        ungrounded_name: int,
        ungrounded_location: int,
        num_subsidiaries: int,
        filing: Filing,
    ) -> None:
        """Track stats on extraction operations.

        Args:
            num_chunks: The number of chunks and exhibit may be split up in
            ungrounded_name: The number of instances where name check failed
            ungrounded_location: The number of instances where location check failed
            num_subsidiaries: The number of subsidiaries extracted
            filing: The Filing object the subsidiaries were extracted for
        """
        if num_chunks > 1:
            self.stats.increment("chunked_extractions")

        if ungrounded_name:
            self.stats.increment("ungrounded_name", ungrounded_name)

        if ungrounded_location:
            self.stats.increment("ungrounded_location", ungrounded_location)

        if num_subsidiaries == 0:
            self._record_failure(
                (filing.cik, filing.filename),
                FailureType.NO_SUBSIDIARIES,
                "warning",
                "No subsidiaries found for filing: %s - %s - %s",
                filing.cik,
                filing.accession_number,
                filing.filing_date,
                stat_keys=("zero_subsidiaries",),
            )

    def _extract_worker(self, work_queue: queue.Queue, results_queue: queue.Queue) -> None:
        """Worker thread that extracts subsidiaries from queued exhibit documents.

        Runs as a daemon thread, consuming ``(filing, exhibit_contents)`` tuples from
        ``work_queue`` and posting extracted ``list[Subsidiary]`` results to
        ``results_queue``. Extraction errors are caught, logged, and recorded in the
        failure registry so the worker loop continues.

        Args:
            work_queue: Queue of ``(Filing, dict)`` tuples to process. Each dict has
                ``"url"`` and ``"data"`` keys for the exhibit content.
            results_queue: Queue to which extracted ``list[Subsidiary]`` results are
                posted.

        Returns:
            None
        """
        while True:
            filing, exhibit_contents = work_queue.get()
            try:
                subsidiaries, ungrounded_name, ungrounded_location, num_chunks = (
                    self.extractor.extract(filing, exhibit_contents)
                )
                self._report_extraction(
                    num_chunks=num_chunks,
                    ungrounded_name=ungrounded_name,
                    ungrounded_location=ungrounded_location,
                    num_subsidiaries=len(subsidiaries),
                    filing=filing,
                )
                results_queue.put(subsidiaries)

            except DocumentError as e:
                self._record_failure(
                    (filing.cik, filing.filename),
                    FailureType.DOCUMENT_ERROR,
                    "error",
                    "Document error for filing: %s - %s - %s: %s",
                    filing.cik,
                    filing.accession_number,
                    filing.filing_date,
                    e,
                )

            except ExtractionTimeoutError:
                self._record_failure(
                    (filing.cik, filing.filename),
                    FailureType.TIMEOUT_ERROR,
                    "error",
                    "Timeout extracting subsidiaries from filing: %s - %s - %s",
                    filing.cik,
                    filing.accession_number,
                    filing.filing_date,
                    stat_keys=("failed_subsidiaries", "timeout_subsidiaries"),
                )

            except ExtractionTruncatedError as e:
                self._record_failure(
                    (filing.cik, filing.filename),
                    FailureType.TRUNCATED_ERROR,
                    "error",
                    "Truncated extraction for filing: %s - %s - %s: %s",
                    filing.cik,
                    filing.accession_number,
                    filing.filing_date,
                    e,
                    stat_keys=("failed_subsidiaries", "truncated_extractions"),
                )

            except Exception:
                self._record_failure(
                    (filing.cik, filing.filename),
                    FailureType.EXTRACTION_FAILED,
                    "error",
                    "Error extracting subsidiaries from filing: %s - %s - %s",
                    filing.cik,
                    filing.accession_number,
                    filing.filing_date,
                )

            finally:
                work_queue.task_done()

    def _results_worker(
        self,
        results_queue: queue.Queue,
        subsidiaries: list[Subsidiary],
        extract_bar: tqdm | None = None,
    ) -> None:
        """Worker thread that collects extracted subsidiaries from the results queue.

        Runs as a daemon thread, consuming ``list[Subsidiary]`` batches from
        ``results_queue`` and appending them to the shared ``subsidiaries`` list.
        Optionally advances a tqdm progress bar on each batch completion.

        Args:
            results_queue: Queue of ``list[Subsidiary]`` batches produced by
                :meth:`_extract_worker`.
            subsidiaries: Shared list to which extracted subsidiaries are appended.
                Must be safe for single-threaded append (only this worker writes).
            extract_bar: Optional tqdm progress bar incremented by one for each
                completed extraction task.

        Returns:
            None
        """
        while True:
            result = results_queue.get()
            subsidiaries.extend(result)
            self.stats.increment("total_subsidiaries", len(result))

            if extract_bar is not None:
                extract_bar.update(1)

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
            self._record_failure(
                (filing.cik, filing.filename),
                FailureType.NO_FILING_DIRECTORY,
                "error",
                "Filing: %s - %s - %s does not have a directory listing.",
                filing.cik,
                filing.accession_number,
                filing.filing_date,
            )
            return []
        return directory_response.get("data", {}).get("directory", {}).get("item", [])

    def _fetch_pdf_content(self, filing: Filing, item: dict, sec_url: str) -> dict:
        """Fetch PDF content from the SEC.

        Args:
            filing: Filing object to fetch PDF content from
            item: Item object to fetch PDF content from
            sec_url: URL of the PDF file

        Returns:
            Dict with 'url' and 'data' keys
        """
        self.logger.warning("PDF exhibit found: %s / %s", filing.cik, item["name"])
        item_response = self.sec_client.query_endpoint(sec_url=sec_url, return_bytes=True)
        exhibit_content = {}
        if item_response.get("data"):
            try:
                with pdfplumber.open(io.BytesIO(item_response["data"])) as pdf:
                    text = "\n\n".join(page.extract_text() or "" for page in pdf.pages)
                exhibit_content = {"url": sec_url, "data": text}
            except Exception:
                self._record_failure(
                    (filing.cik, filing.filename),
                    FailureType.NO_EXHIBIT_CONTENT,
                    "error",
                    "Failed to extract PDF content: %s",
                    sec_url,
                )
        return exhibit_content

    def _fetch_html_content(self, name: str, filing: Filing, sec_url: str) -> dict:
        """Fetch HTM/HTML content from the SEC and convert it to plain text.

        Mirrors ``_fetch_pdf_content``: fetches the document and pre-processes
        it (HTML → text) so the extractor works on the same string it hands to
        the model and uses for grounding.

        Args:
            name: Name of the item
            filing: Filing object to fetch HTML content from
            sec_url: URL of the HTML file

        Returns:
            Dict with ``"url"`` and ``"data"`` (plain text) keys.
        """
        item_response = self.sec_client.query_endpoint(sec_url=sec_url, return_json=False)
        if item_response.get("data"):
            exhibit_content = {"url": sec_url, "data": html_to_text(item_response["data"])}
        else:
            exhibit_content = {}
            self._record_failure(
                (filing.cik, filing.filename),
                FailureType.NO_EXHIBIT_CONTENT,
                "error",
                "Exhibit %s - %s - %s does not have content.",
                name,
                filing.cik,
                filing.accession_number,
            )
        return exhibit_content

    def _fetch_other_content(self, name: str, filing: Filing, sec_url: str) -> dict:
        """Fetch plain-text exhibit content from the SEC.

        Args:
            name: Name of the item
            filing: Filing object to fetch other content from
            sec_url: URL of the other content

        Returns:
            Dict with 'url' and 'data' keys
        """
        item_response = self.sec_client.query_endpoint(sec_url=sec_url, return_json=False)
        if item_response.get("data"):
            exhibit_content = {"url": sec_url, "data": item_response["data"]}
        else:
            exhibit_content = {}
            self._record_failure(
                (filing.cik, filing.filename),
                FailureType.NO_EXHIBIT_CONTENT,
                "error",
                "Exhibit %s - %s - %s does not have content.",
                name,
                filing.cik,
                filing.accession_number,
            )
        return exhibit_content

    def _fetch_exhibit_content(self, filing: Filing, item: dict) -> dict:
        """Fetch exhibit content from the SEC.

        Supports HTM, HTML, TXT, and PDF files. PDFs are extracted via pdfplumber.
        Unsupported file types are skipped. PDF instances are logged as warnings
        since they are rare and may require manual review.

        Args:
            filing: Filing object to fetch exhibit content from
            item: Item object to fetch exhibit content from

        Returns:
            Dict with 'url' and 'data' keys, or empty dict if not an exhibit 21 file
            or the file type is unsupported.
        """
        name = item["name"].upper()
        accession = filing.accession_number.replace("-", "")

        num = filing.exhibit_type
        num_re = self.TWENTYONE if num == "21" else self.EIGHT

        if not (
            (self.EX.search(name) and (name.startswith(num) or num_re.search(name)))
            or "SUB" in name
        ):
            return {}

        ext = item["name"].rsplit(".", 1)[-1].upper() if "." in item["name"] else ""
        if ext not in SUPPORTED_EXHIBIT_EXTENSIONS:
            self.logger.warning("Unsupported exhibit extension: %s", ext)
            return {}

        sec_url = f"{self.sec_client.SEC_URL}/{filing.cik}/{accession}/{item['name']}"
        self.stats.increment(f"{ext.lower()}_exhibits")
        if ext == "PDF":
            exhibit_content = self._fetch_pdf_content(filing, item, sec_url)
        elif ext in ("HTM", "HTML"):
            exhibit_content = self._fetch_html_content(name, filing, sec_url)
        else:
            exhibit_content = self._fetch_other_content(name, filing, sec_url)

        return exhibit_content

    def _fetch_exhibit(self, filing: Filing) -> list[dict]:
        """Fetch exhibit data from the SEC.

        Args:
            filing: Filing object to fetch exhibit data from

        Returns:
            List of dicts with 'url' and 'data' keys
        """
        directory_items = self._fetch_directory(filing)
        exhibit_content = []
        for item in directory_items:
            exhibits = self._fetch_exhibit_content(filing, item)
            if exhibits:
                exhibit_content.append(exhibits)
        return exhibit_content

    def process(self, input_list: list[Filing]) -> list[Subsidiary]:
        """Fetch exhibit content and extract subsidiaries from each filing.

        Exhibit fetching (SEC HTTP calls) runs on the main thread; extraction is
        parallelised across :attr:`~PipelineConfig.num_workers` daemon threads.
        Progress is reported via two tqdm bars (fetching and extraction).

        Args:
            input_list: List of :class:`Filing` objects returned by
                :meth:`load_input`.

        Returns:
            Deduplicated list of :class:`Subsidiary` objects extracted across all
            filings.
        """
        self.logger.info("Located %d filings with exhibits to process", len(input_list))
        # Queues to store exhibit data and subsidiary data
        work_queue = queue.Queue(maxsize=self.config.num_workers * 2)
        result_queue = queue.Queue()
        subsidiaries = []

        with (
            tqdm(
                total=len(input_list), desc="Fetching exhibits", position=0, leave=True
            ) as fetch_bar,
            tqdm(total=0, desc="Extracting subsidiaries", position=1, leave=True) as extract_bar,
        ):
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
                args=(result_queue, subsidiaries, extract_bar),
                daemon=True,
                name="results-worker",
            )
            results_worker.start()

            # SEC operations to fetch exhibit data — one task per document
            for filing in input_list:
                exhibit_contents = self._fetch_exhibit(filing)
                if not exhibit_contents:
                    self.failure_registry.add(
                        (filing.cik, filing.filename), FailureType.NO_EXHIBIT_FOUND
                    )
                for exhibit_content in exhibit_contents:
                    work_queue.put((filing, exhibit_content))
                    extract_bar.total += 1
                    extract_bar.refresh()
                fetch_bar.update(1)

            # Wait for all extraction to complete
            work_queue.join()
            result_queue.join()

        return subsidiaries

    def save_output(self, processed_list: list[Subsidiary]) -> None:
        """Deduplicate and persist extracted subsidiaries as a Parquet file.

        Merges new rows with any existing parquet, drops duplicates keyed on
        ``(parent_cik, accession_number, name)``, and stamps a UTC ``date_added``
        column before writing.

        Args:
            processed_list: List of :class:`Subsidiary` objects returned by
                :meth:`process`.

        Returns:
            None
        """
        # Save processed subsidiaries to a DataFrame
        subsidiaries_df = pd.DataFrame([dataclasses.asdict(s) for s in processed_list])

        try:
            existing_subsidiaries_df = pd.read_parquet(self.config.output_file)

            # Merge the existing subsidiaries with the new subsidiaries
            self.logger.info(
                "Merging existing %d subsidiaries with %d new subsidiaries",
                len(existing_subsidiaries_df),
                len(subsidiaries_df),
            )
            combined_subsidiaries_df = pd.concat(
                [existing_subsidiaries_df, subsidiaries_df], ignore_index=True
            )

        except FileNotFoundError:
            self.logger.info("No existing subsidiaries found, creating new file")
            combined_subsidiaries_df = subsidiaries_df

        # Canonicalize jurisdiction strings so the same place yields the same
        # value across filings. Applied to merged historic + new rows so that
        # alias-dict updates retroactively normalize older data on next write.
        combined_subsidiaries_df["location"] = (
            combined_subsidiaries_df["location"].fillna("").map(normalize_subsidiary_location)
        )
        combined_subsidiaries_df["parent_location"] = (
            combined_subsidiaries_df["parent_location"].fillna("").map(normalize_parent_location)
        )

        # Drop duplicate rows keyed on (parent_cik, accession_number, name)
        combined_subsidiaries_df = combined_subsidiaries_df.drop_duplicates(
            subset=["parent_cik", "accession_number", "name"]
        )

        # Add a date_added column if it doesn't exist and set the value to the current UTC timestamp
        if "date_added" not in combined_subsidiaries_df.columns:
            combined_subsidiaries_df["date_added"] = pd.NA
        combined_subsidiaries_df.loc[
            combined_subsidiaries_df["date_added"].isna(), "date_added"
        ] = datetime.datetime.now(datetime.UTC).isoformat()

        # Save the combined subsidiaries to the output file
        combined_subsidiaries_df.to_parquet(self.config.output_file)
        self.logger.info(
            "Saved %d subsidiaries to %s", len(combined_subsidiaries_df), self.config.output_file
        )

    def display_stats(self) -> None:
        """Log a formatted summary of pipeline statistics on completion.

        Writes filing totals (total, skipped, failed) and subsidiary totals
        (total, failed) to the logger at INFO level.

        Returns:
            None
        """
        self.logger.info("=" * 40)
        self.logger.info("Pipeline Stats")
        self.logger.info("=" * 40)
        self.logger.info("  Filings")
        self.logger.info("    Total:    %d", self.stats.total_filing)
        self.logger.info("    Skipped:  %d", self.stats.skipped_filings)
        self.logger.info("    Failed:   %d", self.stats.failed_filings)
        self.logger.info("  Subsidiaries")
        self.logger.info("    Total:    %d", self.stats.total_subsidiaries)
        self.logger.info("    Failed:   %d", self.stats.failed_subsidiaries)
        self.logger.info("    Timeouts: %d", self.stats.timeout_subsidiaries)
        self.logger.info("    Truncated: %d", self.stats.truncated_extractions)
        self.logger.info("    Chunked:   %d", self.stats.chunked_extractions)
        self.logger.info("    Zero:     %d", self.stats.zero_subsidiaries)
        self.logger.info("    Ungrounded name:     %d", self.stats.ungrounded_name)
        self.logger.info("    Ungrounded location: %d", self.stats.ungrounded_location)
        self.logger.info("    Dropped:             %d", self.stats.dropped_subsidiaries)
        self.logger.info("  Exhibits by type")
        self.logger.info("    HTM:  %d", self.stats.htm_exhibits)
        self.logger.info("    HTML: %d", self.stats.html_exhibits)
        self.logger.info("    TXT:  %d", self.stats.txt_exhibits)
        self.logger.info("    PDF:  %d", self.stats.pdf_exhibits)
        self.logger.info("=" * 40)

    def run(self) -> None:
        """Run the pipeline, flushing any buffered failures on completion."""
        try:
            super().run()
        finally:
            self.failure_registry.flush()
