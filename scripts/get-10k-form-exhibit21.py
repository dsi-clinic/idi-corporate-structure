import json
import os
import re
import time

import requests
# from tqdm import tqdm

def tqdm(iterable, **kwargs):
    yield from iterable

RATE_LIMIT = 0.2  # seconds per request
EX = re.compile(r"\BEX")
TWENTYONE = re.compile("[^0-9]21")

for directory in tqdm(os.listdir("form-directories"), miniters=1):
    outdir = f"form-exhibit21/{directory.split('.')[0]}"
    if os.path.exists(outdir):
        continue

    with open(f"form-directories/{directory}") as file:
        items = json.load(file)["directory"]["item"]

    assert directory.endswith(".json")
    date, cik, accessionNumber = directory[:-5].split("_")
    accession = accessionNumber.replace("-", "")

    urls = {}
    for item in items:
        name = item["name"].upper()
        if (EX.search(name) and (name.startswith("21") or TWENTYONE.search(name))) or "SUB" in name:
            urls[item["name"]] = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{item['name']}"

    os.mkdir(outdir)

    for filename, url in urls.items():
        start_time = time.time()
        response = requests.get(url, headers={"User-Agent": "Jim Pivarski jpivarski@uchicago.edu"})
        with open(f"{outdir}/{filename}", "wb") as file:
            file.write(response.content)

        stop_time = time.time()
        wait_time = RATE_LIMIT - (stop_time - start_time)
        if wait_time > 0:
            time.sleep(wait_time)
