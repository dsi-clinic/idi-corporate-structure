import json
import time

import requests
import pandas as pd
import boto3

ORCHESTRATION_SQS_QUEUE = ""
NUM_SQS_RETRIES = 5
USER_AGENT = "Jim Pivarski jpivarski@uchicago.edu"
OUTPUT_S3_BUCKET = ""
OUTPUT_S3_PREFIX = ""

RATE_LIMIT = 0.2  # seconds per request
EX = re.compile(r"\BEX")
N21 = re.compile("[^0-9]21")
N8 = re.compile("[^0-9]8")

def request_with_wait(url: str, *, expect_json: bool, retry: int = 5) -> requests.Response | None:
    if retry == 0:
        return None

    start_time = time.time()
    response = requests.get(url, headers={"User-Agent": USER_AGENT})

    if response.status_code != 200:
        return request_with_wait(url, expect_json, retry - 1)

    if expect_json:
        try:
            json.loads(response.content)
        except Exception:
            return request_with_wait(url, expect_json, retry - 1)

    stop_time = time.time()

    wait_time = RATE_LIMIT - (stop_time - start_time)
    if wait_time > 0:
        time.sleep(wait_time)

    return response


if __name__ == "__main__":
    print("connecting to S3")
    s3 = boto.client("s3")

    print("connecting to SQS")
    sqs = boto.client("sqs")

    retry_countdown = NUM_SQS_RETRIES
    while True:
        sqs_response = sqs.receive_message(
            QueueUrl=ORCHESTRATION_SQS_QUEUE,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=10,
            VisibilityTimeout=600,
        )
        messages = sqs_response.get("Messages", [])
        if len(messages) == 0:
            print(f"no response; retrying {retry_countdown} more times")
            if retry_countdown == 0:
                break
            else:
                retry_countdown -= 1
                continue

        # if we ever see a message, the retry_countdown gets reset
        retry_countdown = NUM_SQS_RETRIES

        message_data = json.loads(messages[0]["Body"])
        cik = message_data["cik"]
        date = message_data["date"]
        form = message_data["form"]
        is_10k = message_data["is_10k"]
        accessionNumber = message_data["accessionNumber"]
        directory_url = message_data["directory_url"]

        response = request_with_wait(directory_url, expect_json=True)
        if response is not None:
            form_urls = []
            for item in response.json()["directory"]["item"]:
                name = item["name"].upper()
                is_text = (
                    name.endswith(".TXT")
                    or name.endswith(".HTM")
                    or name.endswith(".HTML")
                    or name.endswith(".PDF")
                )
                has_ex = EX.search(name)
                good_10k = has_ex and (name.startswith("21") or N21.search(name))
                good_20f = has_ex and (name.startswith("8") or N8.search(name))
                has_sub = "SUB" in name
                if (is_text and (
                    has_sub or (is_10k and good_10k) or (not is_10k and good_20f)
                )):
                    form_urls.append(
                        (item["name"], f"https://www.sec.gov/Archives/edgar/data/{cik}/{accessionNumber}/{item['name']}")
                    )

            for filename, form_url in form_urls:
                response = request_with_wait(form_url, expect_json=False)
                if response is not None:
                    s3.put_object(
                        Bucket=OUTPUT_S3_BUCKET,
                        Key=f"{OUTPUT_S3_PREFIX}/{cik}_{accessionNumber}/{filename}",
                        Body=response.content,
                    )
                    s3.put_object(
                        Bucket=OUTPUT_S3_BUCKET,
                        Key=f"{OUTPUT_S3_PREFIX}/{cik}_{accessionNumber}_meta/{filename}.json",
                        Body=json.dumps({
                            "cik": cik,
                            "date": date,
                            "form": form,
                            "is_10k": is_10k,
                            "accessionNumber": accessionNumber,
                            "directory": directory_url,
                        }).encode(),
                    )

        # tell SQS that we have successfully handled this message
        sqs.delete_message(
            QueueUrl=ORCHESTRATION_SQS_QUEUE,
            ReceiptHandle=messages[0]["ReceiptHandle"],
        )
