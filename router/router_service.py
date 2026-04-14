import queue
import threading
import time
import uuid
import zmq
from data_model.data_types import TaggedRequest, BatchRequest
from model_logic.model_endpoint.grpc_client import ModelServiceClient

MLP_PULL_ADDR  = "tcp://localhost:5556"
GRPC_TARGET    = "localhost:50051"
MAX_BATCH_SIZE = 4
BATCH_TIMEOUT  = 0.05  # seconds


def receiver_thread(zmq_ctx: zmq.Context, shared_queue: queue.Queue):
    sock = zmq_ctx.socket(zmq.PULL)
    sock.connect(MLP_PULL_ADDR)
    print(f"[Router:recv] connected to {MLP_PULL_ADDR}")
    while True:
        raw = sock.recv_json()
        tagged = TaggedRequest(**raw)
        shared_queue.put(tagged)
        print(f"[Router:recv] enqueued {tagged.request_id} task={tagged.task_type}")


def batcher_thread(shared_queue: queue.Queue, client: ModelServiceClient):
    print("[Router:batch] started")
    while True:
        batch = []
        deadline = time.monotonic() + BATCH_TIMEOUT
        while len(batch) < MAX_BATCH_SIZE:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = shared_queue.get(timeout=remaining)
                batch.append(item)
            except queue.Empty:
                break
        if not batch:
            continue
        batch_id = str(uuid.uuid4())
        br = BatchRequest(batch_id=batch_id, requests=batch)
        print(f"[Router:batch] sending batch {batch_id} size={len(batch)}")
        resp = client.prefill(
            batch_id=batch_id,
            request_ids=[r.request_id for r in br.requests],
            prompts=[r.prompt for r in br.requests],
            max_tokens=[r.max_tokens for r in br.requests],
        )
        print(f"[Router:batch] gRPC Prefill status={resp.status}")


def main():
    zmq_ctx = zmq.Context()
    client = ModelServiceClient(GRPC_TARGET)
    q = queue.Queue()
    t1 = threading.Thread(target=receiver_thread, args=(zmq_ctx, q), daemon=True, name="recv")
    t2 = threading.Thread(target=batcher_thread, args=(q, client), daemon=True, name="batch")
    t1.start()
    t2.start()
    print("[Router] both threads running")
    t1.join()
    t2.join()


if __name__ == "__main__":
    main()
