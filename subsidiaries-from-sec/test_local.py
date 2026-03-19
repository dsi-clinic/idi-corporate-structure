import time
import json
import os

import local


def dispatcher():
    print("hello")

    sqs = boto3.client("sqs")

    for i in range(10):
        body = json.dumps({"i": i})
        print(f"write {body!r}")
        sqs.send_message(QueueUrl="dummy", MessageBody=body)

        if json.loads(body)["i"] == 8:
            raise Exception("ouch")

    print("good bye")


def worker():
    print(f"hi, I'm {os.environ['AWS_BATCH_JOB_ARRAY_INDEX']}")

    sqs = boto3.client("sqs")

    while True:
        sqs_response = sqs.receive_message(
            QueueUrl="", MaxNumberOfMessages=1, WaitTimeSeconds=2
        )
        messages = sqs_response.get("Messages", [])
        if len(messages) == 0:
            break

        body = messages[0]["Body"]
        print(f"read {body!r}")

        if json.loads(body)["i"] == 4:
            raise Exception("ouch")

        time.sleep(0.2)

    print("see ya")


if __name__ == "__main__":
    local.run(dispatcher, worker, 3)
