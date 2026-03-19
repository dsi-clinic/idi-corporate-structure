import json
import logging
import os
import sys
import time

import requests

REQUESTS_PER_DAY = 5000
PAUSE_BETWEEN_REQUESTS = 5
NUM_ATTEMPTS = 3
LSEG_API_KEY = os.environ["LSEG_API_KEY"]
LOGFILE = None if len(sys.argv) == 1 else sys.argv[1]


def do_search(cik, filename, attempt):
    attempt += 1

    time.sleep(PAUSE_BETWEEN_REQUESTS)
    try:
        response = requests.get(
            f"https://api-eit.refinitiv.com/permid/search?q=CIK:{cik}&format=json&access-token={LSEG_API_KEY}"
        )
    except Exception as err:
        if attempt >= NUM_ATTEMPTS:
            with open(filename, "w") as file:
                json.dump(
                    {"requests_error": type(err).__name__, "message": str(err)}, file
                )
            return attempt
        else:
            logging.warning(f"{cik = } FAILED with {type(err).__name__}: {str(err)}")
            return do_search(cik, filename, attempt)

    if response.status_code != 200:
        if attempt >= NUM_ATTEMPTS:
            with open(filename, "w") as file:
                json.dump(
                    {
                        "requests_error": f"code {response.status_code}",
                        "message": response.content.decode(),
                    },
                    file,
                )
            return attempt
        else:
            logging.warning(
                f"{cik = } FAILED with status code {response.status_code}: {response.content.decode()}"
            )
            return do_search(cik, filename, attempt)

    with open(filename, "wb") as file:
        file.write(response.content)
    logging.info(f"{cik=}")
    return attempt


def do_mdass(cik, permid, filename, attempt):
    attempt += 1

    time.sleep(PAUSE_BETWEEN_REQUESTS)
    try:
        response = requests.get(
            f"https://permid.org/api/mdaas/getEntityById/{permid}?access-token={LSEG_API_KEY}"
        )
    except Exception as err:
        if attempt >= NUM_ATTEMPTS:
            with open(filename, "a") as file:
                json.dump(
                    {
                        "PERM ID": permid,
                        "requests_error": type(err).__name__,
                        "message": str(err),
                    },
                    file,
                )
            return attempt
        else:
            logging.warning(
                f"{cik = } {permid = } FAILED with {type(err).__name__}: {str(err)}"
            )
            return do_mdass(cik, permid, filename, attempt)

    if response.status_code != 200:
        if attempt >= NUM_ATTEMPTS:
            with open(filename, "a") as file:
                json.dump(
                    {
                        "PERM ID": permid,
                        "requests_error": f"code {response.status_code}",
                        "message": response.content.decode(),
                    },
                    file,
                )
            return attempt
        else:
            logging.warning(
                f"{cik = } {permid = } FAILED with status code {response.status_code}: {response.content.decode()}"
            )
            return do_mdass(cik, permid, filename, attempt)

    data = response.json()
    with open(filename, "a") as file:
        file.write(json.dumps(data))

    logging.info(f"{cik=} {permid=} lei={data.get('LEI', '?')}")
    return attempt


if __name__ == "__main__":
    logging_args = {
        "level": logging.INFO,
        "format": "%(asctime)s %(levelname)s: %(message)s",
    }
    if LOGFILE is None:
        logging_args["stream"] = sys.stdout
    else:
        logging_args["filename"] = LOGFILE
    logging.basicConfig(**logging_args)

    with open("data/unique-ciks.txt") as file:
        unique_ciks = [x.rstrip() for x in file]

    for cik in unique_ciks:
        num_api_calls = 0
        filename_search = f"OUTPUT-SEARCH/CIK{cik}.json"
        filename_mdass = f"OUTPUT-MDASS/CIK{cik}.jsonl"

        start_time = time.time()

        if not os.path.exists(filename_search):
            with open(filename_search, "w"):
                pass  # create an empty file to "own" this CIK

            num_api_calls += do_search(cik, filename_search, 1)

        if not os.path.exists(filename_mdass) and os.path.getsize(filename_search) != 0:
            with open(filename_mdass, "w"):
                pass  # create an empty file to "own" this CIK

            logging.info(f"staring MDASS for {cik}")

            try:
                with open(filename_search) as file:
                    search_data = json.load(file)
            except Exception:
                # the search file was nonempty, but not full; let another process take it
                os.unlink(filename_mdass)
            else:
                permids = [
                    x.get("@id", "").split("/")[-1]
                    for x in search_data.get("result", {})
                    .get("organizations", {})
                    .get("entities", [])
                ]
                logging.info(f"number of permids for {cik} is {len(permids)}")
                for permid in permids:
                    if permid != "":
                        num_api_calls += do_mdass(cik, permid, filename_mdass, 1)

        actual_time = time.time() - start_time

        if num_api_calls != 0:
            target_time_between_requests = (24 * 60 * 60) / REQUESTS_PER_DAY
            time_to_wait = 1 + max(0, target_time_between_requests - actual_time)
            logging.info(f"waiting {time_to_wait} seconds")
            time.sleep(time_to_wait)
