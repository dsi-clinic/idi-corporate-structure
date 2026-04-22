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
from tqdm import tqdm

# Application imports
from idi_corporate_structure.common.api import SecClient
from idi_corporate_structure.common.failures import FailureRegistry
from idi_corporate_structure.common.logs import get_logger
from idi_corporate_structure.common.storage import open_zip
from idi_corporate_structure.processor.extractor import DocumentError, GptExtractor
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
    _SUPPORTED_EXHIBIT_EXTENSIONS = frozenset({"HTM", "HTML", "TXT", "PDF"})

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
        self.logger = get_logger("SubsidiaryPipeline")
        self.failure_registry = FailureRegistry(
            config.failure_file,
            classifier=CorporateStructureFailureClassifier(),
            flush_every=config.failure_flush_every,
        )
        self.rows = []
        self._stale_filing_keys: set[tuple[str, str]] = set()

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
                self.logger.error("Overflow file not found: %s", overflow_filename)
                self.stats.increment("failed_filings")
                self.failure_registry.add(
                    (cik, overflow_filename), failure_type=FailureType.NO_OVERFLOW_FILINGS
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
            self.logger.error("Filename: %s has forms with mismatched data lengths.", filename)
            self.stats.increment("failed_filings")
            self.failure_registry.add((cik, filename), failure_type=FailureType.MISMATCHED_LENGTHS)
            return None

        if not any([forms, accession_numbers, primary_documents, filing_dates]):
            self.logger.debug("Filename: %s has forms without data.", filename)
            self.stats.increment("failed_filings")
            self.failure_registry.add((cik, filename), failure_type=FailureType.NO_FORM_DATA)
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

        if primary_document != "" and primary_document.split(".")[-1].upper() in ("HTM", "HTML"):
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

    def load_input(self) -> list[Filing]:
        """Load input data from the SEC and return a list of filings.

        Returns:
            A list of Filing objects
        """
        filings = []
        with open_zip(self.config.input_file, headers=self.sec_client.SEC_HEADERS) as zf:
            namelist = zf.namelist()
            if self._INPUT_SAMPLE_SIZE:
                namelist = namelist[: self._INPUT_SAMPLE_SIZE]
            self.logger.info("Total # of files to process: %d", len(namelist))

            for filename in tqdm(namelist, desc="Retrieving filings"):
                filings_for_file = self._parse_file(zf, filename)
                if filings_for_file:
                    filings.extend(filings_for_file)

        self.logger.info(
            "Located %d filings for %d files",
            len(filings),
            len(namelist) - self.stats.skipped_filings,
        )
        self.logger.info("Skipped %d files", self.stats.skipped_filings)

        return self._filter_already_processed(filings)

    def _filter_already_processed(self, filings: list[Filing]) -> list[Filing]:
        """Drop filings already represented in the output parquet, unless stale.

        Reads the existing output parquet and partitions its ``(parent_cik,
        accession_number)`` keys into fresh (within ``stale_threshold_days``) and
        stale (older). Fresh filings are removed from the returned list; stale
        filing keys are recorded on ``self._stale_filing_keys`` so that
        `save_output` can drop their old rows before merging the new ones.

        If ``stale_threshold_days`` is ``None`` or no parquet exists yet, returns
        ``filings`` unchanged.

        Args:
            filings: Filings returned by `load_input`.

        Returns:
            Filings that still need to be processed (new + stale).
        """
        if self.config.stale_threshold_days is None:
            return filings

        try:
            existing_df = pd.read_parquet(self.config.output_file)
        except FileNotFoundError:
            return filings

        if existing_df.empty or "date_added" not in existing_df.columns:
            return filings

        # Get the last date_added for each (parent_cik, accession_number) group
        last_added = (
            pd.to_datetime(existing_df["date_added"], utc=True, errors="coerce")
            .groupby([existing_df["parent_cik"], existing_df["accession_number"]])
            .max()
        )
        threshold = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=self.config.stale_threshold_days)

        # Filter filings to only include those that have a date_added greater than or equal to the threshold
        fresh_keys = {key for key, ts in last_added.items() if pd.notna(ts) and ts >= threshold}

        # Filter filings to only include those that have a date_added less than the threshold
        self._stale_filing_keys = {
            key for key, ts in last_added.items() if pd.notna(ts) and ts < threshold
        }

        # Filter filings to only include those that are not in the fresh_keys set
        unprocessed = [
            f
            for f in filings
            if (f.cik, f.accession_number) not in fresh_keys
            and (f.cik, f.filename) not in self.failure_registry
        ]
        self.logger.info(
            "Filter: %d filings loaded, %d skipped as fresh, %d stale to reprocess, %d remaining",
            len(filings),
            len(filings) - len(unprocessed),
            len(self._stale_filing_keys),
            len(unprocessed),
        )
        return unprocessed

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
                subsidiaries = self.extractor.extract(filing, exhibit_contents)
                results_queue.put(subsidiaries)

                if len(subsidiaries) == 0:
                    self.logger.warning(
                        "No subsidiaries found for filing: %s - %s - %s",
                        filing.cik,
                        filing.accession_number,
                        filing.filing_date,
                    )
                    self.stats.increment("zero_subsidiaries")
                    self.failure_registry.add(
                        (filing.cik, filing.filename), FailureType.NO_SUBSIDIARIES
                    )

            except DocumentError as e:
                self.logger.error(
                    "Document error for filing: %s - %s - %s: %s",
                    filing.cik,
                    filing.accession_number,
                    filing.filing_date,
                    e,
                )
                self.stats.increment("failed_subsidiaries")
                self.failure_registry.add((filing.cik, filing.filename), FailureType.DOCUMENT_ERROR)

            except Exception as _:
                self.logger.error(
                    "Error extracting subsidiaries from filing: %s - %s - %s",
                    filing.cik,
                    filing.accession_number,
                    filing.filing_date,
                )
                self.stats.increment("failed_subsidiaries")
                self.failure_registry.add(
                    (filing.cik, filing.filename), FailureType.EXTRACTION_FAILED
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
            self.logger.error(
                "Filing: %s - %s - %s does not have a directory listing.",
                filing.cik,
                filing.accession_number,
                filing.filing_date,
            )
            self.stats.increment("failed_subsidiaries")
            self.failure_registry.add(
                (filing.cik, filing.filename), FailureType.NO_FILING_DIRECTORY
            )
            return []
        self.sec_client.rate_limit()
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
                self.logger.error("Failed to extract PDF content: %s", sec_url)
                self.stats.increment("failed_subsidiaries")
                self.failure_registry.add(
                    (filing.cik, filing.filename), FailureType.NO_EXHIBIT_CONTENT
                )
        return exhibit_content

    def _fetch_other_content(self, name: str, filing: Filing, sec_url: str) -> dict:
        """Fetch other content from the SEC including HTM, HTML, and TXT files.

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
            self.logger.error(
                "Exhibit %s - %s - %s does not have content.",
                name,
                filing.cik,
                filing.accession_number,
            )
            self.stats.increment("failed_subsidiaries")
            self.failure_registry.add((filing.cik, filing.filename), FailureType.NO_EXHIBIT_CONTENT)
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
        if ext not in self._SUPPORTED_EXHIBIT_EXTENSIONS:
            self.logger.warning("Unsupported exhibit extension: %s", ext)
            return {}

        sec_url = f"{self.sec_client.SEC_URL}/{filing.cik}/{accession}/{item['name']}"
        if ext == "PDF":
            exhibit_content = self._fetch_pdf_content(filing, item, sec_url)
        else:
            exhibit_content = self._fetch_other_content(name, filing, sec_url)

        self.sec_client.rate_limit()
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
        self.logger.info("Located %d subsidiaries", len(input_list))
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

    def _drop_stale_rows(self, existing_subsidiaries_df: pd.DataFrame) -> pd.DataFrame:
        """Drop stale rows from the subsidiaries DataFrame.

        Args:
            existing_subsidiaries_df: DataFrame of existing subsidiaries

        Returns:
            DataFrame of subsidiaries with stale rows dropped
        """
        # Create a mask of stale rows
        if self._stale_filing_keys and not existing_subsidiaries_df.empty:
            stale_mask = pd.Series(
                list(
                    zip(
                        existing_subsidiaries_df["parent_cik"],
                        existing_subsidiaries_df["accession_number"],
                    )
                ),
                index=existing_subsidiaries_df.index,
            ).isin(self._stale_filing_keys)
            # Drop rows for stale filings so re-extracted subsidiaries replace them
            dropped = int(stale_mask.sum())
            if dropped:
                self.logger.info("Dropping %d stale rows before merge", dropped)
                existing_subsidiaries_df = existing_subsidiaries_df.loc[~stale_mask]

        return existing_subsidiaries_df

    def save_output(self, processed_list: list[Subsidiary]) -> None:
        """Deduplicate and persist extracted subsidiaries as a Parquet file.

        Drops duplicate rows keyed on ``(parent_cik, accession_number, name)`` and
        appends a UTC ``date_added`` timestamp column before writing.

        Args:
            processed_list: List of :class:`Subsidiary` objects returned by
                :meth:`process`.

        Returns:
            None
        """
        # Save processed subsidiaries to a DataFrame
        subsidiaries_df = pd.DataFrame([dataclasses.asdict(s) for s in processed_list])

        try:
            # Read existing subsidiaries from the output file and drop stale rows
            existing_subsidiaries_df = pd.read_parquet(self.config.output_file)
            existing_subsidiaries_df = self._drop_stale_rows(existing_subsidiaries_df)

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
        self.logger.info("    Zero:     %d", self.stats.zero_subsidiaries)
        self.logger.info("=" * 40)

    def run(self) -> None:
        """Run the pipeline, flushing any buffered failures on completion."""
        try:
            super().run()
        finally:
            self.failure_registry.flush()
