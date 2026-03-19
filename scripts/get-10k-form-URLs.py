import json
import os
import re
import zipfile

import pandas as pd
from tqdm import tqdm

is_10k = re.compile("10-?K")
is_date = re.compile("[0-9]{4}-[0-9]{2}-[0-9]{2}")

rows = []

with zipfile.ZipFile(os.path.expanduser("~/Box/dsi-core/11th-hour/idi-corporate-structure/submissions.zip")) as zf:
    namelist = list(zf.namelist())
    for filename in tqdm(namelist):
        if filename.startswith("CIK") and filename.endswith(".json"):
            cik = filename[3:-5]

            with zf.open(filename) as file:
                data = json.load(file)

            forms = data.get("filings", {}).get("recent", {}).get("form", [])
            accessionNumbers = data.get("filings", {}).get("recent", {}).get("accessionNumber", [])
            primaryDocuments = data.get("filings", {}).get("recent", {}).get("primaryDocument", [])
            filingDates = data.get("filings", {}).get("recent", {}).get("filingDate", [])
            assert len(forms) == len(accessionNumbers)
            assert len(forms) == len(primaryDocuments)
            assert len(forms) == len(filingDates)

            for form, accessionNumber, primaryDocument, filingDate in zip(forms, accessionNumbers, primaryDocuments, filingDates):
                if is_10k.match(form):
                    accession = accessionNumber.replace("-", "")
                    directory = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/index.json"
                    if primaryDocument != "" and primaryDocument.split(".")[-1].upper() in ("HTM", "HTML"):
                        primary = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primaryDocument}"
                    else:
                        primary = None

                    rows.append({"cik": cik, "date": filingDate, "form": form, "accessionNumber": accessionNumber, "directory": directory, "primary": primary})

pd.DataFrame(rows).sort_values(["cik", "date", "form"]).to_csv("form-directories.csv", index=False)
