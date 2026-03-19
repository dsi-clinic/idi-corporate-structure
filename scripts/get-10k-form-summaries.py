import glob
import json
import os
import zipfile
import queue
import threading
import csv

import numpy as np
import pandas as pd
import pdfplumber
import requests
from html2text import html2text
from tqdm import tqdm

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

NUM_THREADS = 16
NUM_RETRIES = 5

exhibits = {}

for exhibit_number in [21, 8]:
    for filename in tqdm(
        glob.glob(f"form-exhibit{exhibit_number}/*/*"),
        desc=f"exhibit {exhibit_number}",
    ):
        key = tuple(filename.split("/")[1].split("_")) + (exhibit_number,)
        if key not in exhibits:
            exhibits[key] = []

        fnup = filename.upper()
        if fnup.endswith(".TXT"):
            with open(filename) as file:
                exhibits[key].append(file.read())

        elif fnup.endswith(".HTM") or fnup.endswith(".HTML"):
            with open(filename) as file:
                exhibits[key].append(html2text(file.read()))

        elif fnup.endswith(".PDF"):
            with pdfplumber.open(filename) as file:
                exhibits[key].append(
                    "\n\n".join(page.extract_text() for page in file.pages)
                )


def summarize(text, countdown, err=None):
    if countdown == 0:
        raise err

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}",
            },
            json={
                "model": "gpt-4.1-nano",
                "messages": [
                    {
                        "role": "system",
                        "content": """
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
""".strip(),
                    },
                    {
                        "role": "user",
                        "content": text,
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
            },
        )
    except Exception as err:
        return summarize(text, countdown - 1, err)

    if response.status_code != 200:
        return summarize(text, countdown - 1, Exception(response.text))

    try:
        summary = json.loads(response.json()["choices"][0]["message"]["content"])
    except Exception as err:
        return summarize(text, countdown - 1, err)

    if all(item["name"] in text for item in summary["subsidiaries"]):
        return summary
    else:
        return summarize(text, countdown - 1, Exception("name not in original text"))


pbar = tqdm(total=len(exhibits), miniters=1)
pbar.set_lock(threading.RLock())

tasks = queue.Queue()
for key, texts in exhibits.items():
    tasks.put((key, texts))

for _ in range(NUM_THREADS):
    tasks.put(None)


def worker(i):
    with open(f"outputs/subsidiaries-from-exhibits-21-8-thread{i:02d}.csv", "w") as file:
        writer = csv.writer(file)

        while True:
            task = tasks.get()
            if task is None:
                break

            key, texts = task

            text = "\n\n".join(texts)
            try:
                summary = summarize(text, NUM_RETRIES)
            except Exception as err:
                with open(f"failures/{'_'.join(map(str, key))}", "w") as file2:
                    file2.write(f"{type(err).__name__}: {str(err)}")
            else:
                for item in summary["subsidiaries"]:
                    writer.writerow(list(key) + [item["name"], item.get("in")])
                    file.flush()

            pbar.update(1)


threads = [threading.Thread(target=worker, args=(i,)) for i in range(NUM_THREADS)]

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

print("DONE")
