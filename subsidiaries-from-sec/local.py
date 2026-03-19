"""
This module provides an infrastructure for testing distributed job orchestration
using multiprocessing. It mocks AWS S3 and SQS clients (via boto3 interface),
and provides functions to launch dispatcher and worker functions in separate
processes, communicating via a multiprocessing.Queue. Useful for simulating
cloud workflow patterns locally without requiring real AWS resources.

Exports:
    run -- Entry point function to orchestrate running a dispatcher and multiple workers.
"""

import builtins
import multiprocessing
import os
import time
import traceback
from typing import Callable

message_queue = multiprocessing.Queue()
message_count = multiprocessing.Value("i", 0)


class MockClient:
    """Base class for mock AWS clients."""

    pass


class MockBoto3:
    """
    Mock replacement for the boto3 library, supporting 's3' and 'sqs' clients.

    Args:
        modified_print (Callable): Print function to use for output.
        message_queue (multiprocessing.Queue): Inter-process queue for messages.
        message_count (multiprocessing.Value): Shared message counter.
    """

    def __init__(self, modified_print, message_queue, message_count):
        self.modified_print = modified_print
        self.message_queue = message_queue
        self.message_count = message_count

    def client(self, name: str) -> MockClient:
        if name == "s3":
            return MockS3(self)
        elif name == "sqs":
            return MockSQS(self)
        else:
            raise Exception(f"MockBoto3.client({name = })")


class MockS3(MockClient):
    """
    Mock implementation of the AWS S3 boto3 client.
    """

    def __init__(self, boto3):
        self.boto3 = boto3

    def download_file(self, bucket, remote_filename, local_filename):
        # TODO
        raise NotImplementedError


class MockSQS(MockClient):
    """
    Mock implementation of the AWS SQS boto3 client.
    """

    def __init__(self, boto3):
        self.boto3 = boto3

    def send_message(self, *, QueueUrl: str = None, MessageBody: str = None):
        """
        Simulate sending a message to an SQS queue.

        Args:
            QueueUrl (str, optional): The queue URL (ignored).
            MessageBody (str, optional): The message content.
        """

        self.boto3.message_queue.put(MessageBody)
        self.boto3.message_count.value += 1

    def receive_message(
        self,
        *,
        QueueUrl: str = None,
        MaxNumberOfMessages: int = None,
        WaitTimeSeconds: float = None,
        VisibilityTimeout: float = None,
    ) -> dict:
        """
        Simulate receiving messages from an SQS queue.

        Args:
            QueueUrl (str, optional): The queue URL (ignored).
            MaxNumberOfMessages (int, optional): Maximum messages to receive.
            WaitTimeSeconds (float, optional): Wait time for message retrieval.
            VisibilityTimeout (float, optional): Visibility timeout (ignored).

        Returns:
            dict: Dictionary with a "Messages" key containing the received messages.
        """

        self.boto3.modified_print(f"{self.boto3.message_count.value} remaining")

        out = []
        for _ in range(MaxNumberOfMessages):
            try:
                body = self.boto3.message_queue.get(timeout=WaitTimeSeconds)
                self.boto3.message_count.value -= 1
                out.append({"Body": body, "ReceiptHandle": None})
            except multiprocessing.queues.Empty:
                pass
        return {"Messages": out}

    def delete_message(
        self,
        *,
        QueueUrl: str = None,
        ReceiptHandle: str = None,
    ):
        """
        Simulate deleting a message from an SQS queue.

        Args:
            QueueUrl (str, optional): The queue URL (ignored).
            ReceiptHandle (str, optional): The message's receipt handle (ignored).
        """

        pass


def modify_print(identifier: str) -> Callable:
    """
    Create a print function that prefixes output with time and identifier.

    Args:
        identifier (str): Identifier to prefix messages with.

    Returns:
        Callable: A print-like function with custom prefixing.
    """

    def modified_print(*args, **kwargs):
        args = (time.strftime("%H:%M:%S"), identifier + ":") + args
        kwargs["flush"] = True
        return builtins.print(*args, **kwargs)

    return modified_print


def run_dispatcher(
    dispatcher: Callable,
    message_queue: multiprocessing.Queue,
    message_count: multiprocessing.Value,
):
    """
    Execute the dispatcher function in isolation, providing it with mock 'boto3'
    and an overridden print function. Catches and logs exceptions.

    Args:
        dispatcher (Callable): The dispatcher function to run.
        message_queue (multiprocessing.Queue): Queue for dispatching messages.
        message_count (multiprocessing.Value): Shared message count.
    """

    # replace the 'print' function in the dispatcher
    dispatcher.__globals__["print"] = modified_print = modify_print("DISPATCHER")
    # replace the 'boto3' module in the dispatcher to mock interfaces
    dispatcher.__globals__["boto3"] = MockBoto3(
        modified_print, message_queue, message_count
    )

    modified_print("BEGIN")

    try:
        dispatcher()

    except Exception:
        for line in traceback.format_exc().split("\n"):
            modified_print(line)
        modified_print("END WITH ERROR 💀")

    else:
        modified_print("END")


def run_worker(
    identifier: str,
    worker: Callable,
    message_queue: multiprocessing.Queue,
    message_count: multiprocessing.Value,
):
    """
    Execute a worker function with a specific identifier, providing mock 'boto3'
    and print. Sets AWS_BATCH_JOB_ARRAY_INDEX and logs exceptions.

    Args:
        identifier (str): Unique identifier for the worker process.
        worker (Callable): Worker function to run.
        message_queue (multiprocessing.Queue): Queue for receiving messages.
        message_count (multiprocessing.Value): Shared message count.
    """

    # replace the 'print' function in the worker to distinguish each worker's output
    worker.__globals__["print"] = modified_print = modify_print(f"WORKER-{identifier}")
    # replace the 'boto3' module in the dispatcher to mock interfaces
    worker.__globals__["boto3"] = MockBoto3(
        modified_print, message_queue, message_count
    )

    # replace the 'AWS_BATCH_JOB_ARRAY_INDEX' environment variable for this worker
    os.environ["AWS_BATCH_JOB_ARRAY_INDEX"] = identifier

    modified_print("BEGIN")

    try:
        worker()

    except Exception as err:
        for line in traceback.format_exc().split("\n"):
            modified_print(line)
        modified_print("END WITH ERROR 💀")

    modified_print("END")


def run(
    dispatcher: Callable,
    worker: Callable,
    num_workers: int,
    *,
    workers_after_dispatcher_ends: bool = False,
):
    """
    Orchestrate running the dispatcher and worker processes.
    Sets up IPC and launches processes. Optionally runs workers only after dispatcher ends.

    Args:
        dispatcher (Callable): Function to launch as dispatcher.
        worker (Callable): Function to launch as each worker.
        num_workers (int): Number of worker processes to launch.
        workers_after_dispatcher_ends (bool, optional): If True, start workers only after the dispatcher finishes.
    """

    dispatcher_process = multiprocessing.Process(
        target=run_dispatcher, args=(dispatcher, message_queue, message_count)
    )

    fmt = f"%0{len(str(num_workers))}d"
    worker_processes = [
        multiprocessing.Process(
            target=run_worker, args=(fmt % i, worker, message_queue, message_count)
        )
        for i in range(num_workers)
    ]

    if workers_after_dispatcher_ends:
        dispatcher_process.start()
        dispatcher_process.join()

        for worker in worker_processes:
            worker.start()
        for worker in worker_processes:
            worker.join()

    else:
        dispatcher_process.start()
        for worker in worker_processes:
            worker.start()

        dispatcher_process.join()
        for worker in worker_processes:
            worker.join()


__all__ = ["run"]
