import io
import json
import re
import zipfile

import boto3

ORCHESTRATION_SQS_QUEUE = ""
SUBMISSIONS_S3_BUCKET = ""
SUBMISSIONS_S3_KEY = ""

LOCAL_ZIPFILE = "tmp.zip"
IS_10K = re.compile("10-?K")
IS_20F = re.compile("20-?F")
IS_DATE = re.compile("[0-9]{4}-[0-9]{2}-[0-9]{2}")

if __name__ == "__main__":
    print("connecting to S3")
    s3 = boto.client("s3")

    print("connecting to SQS")
    sqs = boto.client("sqs")

    print(f"downloading s3://{SUBMISSIONS_S3_BUCKET}/{SUBMISSIONS_S3_KEY}")
    s3.download_file(SUBMISSIONS_S3_BUCKET, SUBMISSIONS_S3_KEY, LOCAL_ZIPFILE)

    print("opening as zipfile")
    with zipfile.ZipFile(LOCAL_ZIPFILE) as zf:
        namelist = list(zf.namelist())
        for filename in namelist:
            if filename.startswith("CIK") and filename.endswith(".json"):
                cik = filename[3:-5]

                with zf.open(filename) as file:
                    data = json.load(file)

                forms = data.get("filings", {}).get("recent", {}).get("form", [])
                accessionNumbers = data.get("filings", {}).get("recent", {}).get("accessionNumber", [])
                filingDates = data.get("filings", {}).get("recent", {}).get("filingDate", [])
                assert len(forms) == len(accessionNumbers)
                assert len(forms) == len(filingDates)

                count = 0
                for form, accessionNumber, filingDate in zip(forms, accessionNumbers, filingDates):
                    is_10k = IS_10K.match(form)
                    is_20f = IS_20F.match(form)

                    if is_10k or is_20f:
                        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accessionNumber.replace('-', '')}/index.json"

                        sqs.send_message(
                            QueueUrl=ORCHESTRATION_SQS_QUEUE,
                            MessageBody=json.dumps({
                                "cik": cik,
                                "date": filingDate,
                                "form": form,
                                "is_10k": is_10k,
                                "accessionNumber": accessionNumber,
                                "directory_url": url,
                            }),
                        )

                print(f"CIK {cik} has {count} out of {len(forms)} 10-K/20-F forms")

    print("done")
