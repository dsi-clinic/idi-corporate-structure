import json
import os
import time

import requests
import pandas as pd
# from tqdm import tqdm

def tqdm(iterable, **kwargs):
    yield from iterable

RATE_LIMIT = 0.2  # seconds per request

df = pd.read_csv("form-directories.csv", dtype=str)
selected = df.groupby(["cik", "form"]).last().sort_values("date", ascending=False).reset_index()

for index, row in tqdm(selected.iterrows(), total=len(selected), miniters=1):
    filename = f"form-directories/{row['date']}_{row['cik']}_{row['accessionNumber']}.json"
    if os.path.exists(filename):
        continue

    start_time = time.time()
    response = requests.get(row["directory"], headers={"User-Agent": "Jim Pivarski jpivarski@uchicago.edu"})
    try:
        test_json = json.loads(response.content)
    except Exception as err:
        print(response.text)
        raise err
    assert "directory" in test_json

    with open(filename, "w") as file:
        json.dump(test_json, file, separators=(",", ":"))

    stop_time = time.time()
    wait_time = RATE_LIMIT - (stop_time - start_time)
    if wait_time > 0:
        time.sleep(wait_time)
