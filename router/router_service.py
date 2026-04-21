"""
Router service — Orca-style iteration-level scheduling.

Two threads:
  receiver_thread  — ZMQ PULL from MLP classifier → pending_queue
  scheduler_thread — tight loop:
      1. DECODE   : advance every active batch by one token
      2. ADMIT    : if capacity available, prefill new requests from pending_queue
      3. IDLE     : sleep only when both queues are empty

Results are pushed via ZMQ PUSH to RESULT_PUSH_ADDR so the app server (or any
subscriber) can collect them keyed by request_id.
"""

import queue
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import zmq

from data_model.data_types import TaggedRequest, ModelResponse
from model_logic.model_endpoint.grpc_client import ModelServiceClient


# ── LRU adapter cache ──────────────────────────────────────────────────────────

class AdapterCache:
    """Keeps at most `max_loaded` adapters resident in the model server.

    Before each prefill, call ensure_loaded() with the adapter IDs in that
    batch.  If a required adapter is not loaded and the cache is full, the
    least-recently-used adapter is evicted first via UnloadAdapter RPC, then
    the new adapter is fetched via LoadAdapter RPC.

    Pass max_loaded=None to disable eviction (load everything, never evict).
    """

    def __init__(self, client: ModelServiceClient,
                 adapter_dirs: dict[str, str],
                 max_loaded: int | None = None):
        self.client = client
        self.adapter_dirs = adapter_dirs          # all known: name → path
        self.max_loaded = max_loaded or len(adapter_dirs)
        self._loaded: OrderedDict[str, None] = OrderedDict()  # end = MRU

    def ensure_loaded(self, adapter_ids: list[str]):
        """Load any missing adapters, evicting LRU entries if at capacity."""
        for aid in dict.fromkeys(adapter_ids):    # unique, preserve order
            if not aid or aid not in self.adapter_dirs:
                continue                           # unknown adapter / base model
            if aid in self._loaded:
                self._loaded.move_to_end(aid)     # mark as recently used
                continue
            if len(self._loaded) >= self.max_loaded:
                lru_id, _ = self._loaded.popitem(last=False)
                self.client.unload_adapter(lru_id)
                print(f"[AdapterCache] evicted {lru_id!r} "
                      f"(max={self.max_loaded})")
            self.client.load_adapter(
                adapter_id=aid, adapter_path=self.adapter_dirs[aid])
            self._loaded[aid] = None
            print(f"[AdapterCache] loaded {aid!r} "
                  f"({len(self._loaded)}/{self.max_loaded} slots used)")

# ── Addresses ──────────────────────────────────────────────────────────────────
MLP_PULL_ADDR    = "tcp://localhost:5556"   # receive tagged requests
RESULT_PUSH_ADDR = "tcp://*:5557"           # push completed ModelResponse objects
GRPC_TARGET      = "localhost:50051"

# ── Scheduling knobs ───────────────────────────────────────────────────────────
MAX_BATCH_SIZE      = 4    # max requests per prefill call
MAX_ACTIVE_REQUESTS = 16   # total concurrent requests across all active batches
ADMIT_TIMEOUT       = 0.05 # seconds to wait for the first request in an empty system
STEP_SLEEP          = 0.0  # sleep between decode iterations (0 = as fast as possible)
IDLE_SLEEP          = 0.01 # sleep when nothing is running and nothing is pending


# ── Per-batch bookkeeping ──────────────────────────────────────────────────────

@dataclass
class BatchTracker:
    batch_id: str
    request_ids: list[str]
    adapter_ids: list[str]
    max_tokens: list[int]
    generated_texts: list[str] = field(default_factory=list)
    finished: list[bool] = field(default_factory=list)
    tokens_generated: list[int] = field(default_factory=list)

    def __post_init__(self):
        n = len(self.request_ids)
        if not self.generated_texts:
            self.generated_texts = [""] * n
        if not self.finished:
            self.finished = [False] * n
        if not self.tokens_generated:
            self.tokens_generated = [0] * n

    @property
    def all_done(self) -> bool:
        return all(self.finished)

    @property
    def active_count(self) -> int:
        return sum(1 for f in self.finished if not f)


# ── Thread 1: receive tagged requests from MLP classifier ─────────────────────

def receiver_thread(zmq_ctx: zmq.Context, pending: queue.Queue):
    sock = zmq_ctx.socket(zmq.PULL)
    sock.connect(MLP_PULL_ADDR)
    print(f"[Router:recv] connected to {MLP_PULL_ADDR}")
    while True:
        raw = sock.recv_json()
        req = TaggedRequest(**raw)
        pending.put(req)
        print(f"[Router:recv] queued {req.request_id} task={req.task_type}")


# ── Thread 2: Orca-style scheduler ────────────────────────────────────────────

def scheduler_thread(pending: queue.Queue,
                     client: ModelServiceClient,
                     zmq_ctx: zmq.Context,
                     adapter_cache: AdapterCache | None = None):
    result_sock = zmq_ctx.socket(zmq.PUSH)
    result_sock.bind(RESULT_PUSH_ADDR)
    print(f"[Router:sched] result socket bound to {RESULT_PUSH_ADDR}")

    # batch_id → BatchTracker for every batch currently being decoded
    active: dict[str, BatchTracker] = {}

    while True:

        # ── STEP 1: DECODE ────────────────────────────────────────────────────
        # Advance every active batch by exactly one token and collect completions.
        finished_batches = []
        for batch_id, tracker in active.items():
            try:
                resp = client.decode(batch_id)
            except Exception as e:
                print(f"[Router:sched] Decode error batch={batch_id}: {e}")
                _deliver_errors(tracker, result_sock, str(e))
                finished_batches.append(batch_id)
                continue

            # Update tracker from gRPC response
            for i, (text, done) in enumerate(zip(resp.generated_texts, resp.is_finished)):
                tracker.generated_texts[i] = text
                if not tracker.finished[i] and done:
                    tracker.tokens_generated[i] += 1
                tracker.finished[i] = done

            if tracker.all_done:
                finished_batches.append(batch_id)

        # Deliver results and remove completed batches
        for batch_id in finished_batches:
            tracker = active.pop(batch_id)
            _deliver_results(tracker, result_sock)
            print(f"[Router:sched] batch {batch_id} done "
                  f"({len(tracker.request_ids)} requests)")

        # ── STEP 2: ADMIT ─────────────────────────────────────────────────────
        # Pull new requests from pending_queue if we have spare capacity.
        n_active = sum(t.active_count for t in active.values())
        capacity = MAX_ACTIVE_REQUESTS - n_active

        if capacity > 0:
            new_reqs = _drain(pending,
                              limit=min(MAX_BATCH_SIZE, capacity),
                              timeout=ADMIT_TIMEOUT if not active else 0)

            if new_reqs:
                _prefill_batch(new_reqs, client, active, result_sock,
                               adapter_cache)

        # ── STEP 3: IDLE ──────────────────────────────────────────────────────
        if not active and pending.empty():
            time.sleep(IDLE_SLEEP)
        elif STEP_SLEEP > 0:
            time.sleep(STEP_SLEEP)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _drain(pending: queue.Queue, limit: int, timeout: float) -> list[TaggedRequest]:
    """Pull up to `limit` items from the queue; wait up to `timeout` for the first."""
    items = []
    deadline = time.monotonic() + timeout
    while len(items) < limit:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            items.append(pending.get(timeout=remaining if not items else 0))
        except queue.Empty:
            break
    return items


def _prefill_batch(reqs: list[TaggedRequest],
                   client: ModelServiceClient,
                   active: dict[str, BatchTracker],
                   result_sock: "zmq.Socket",
                   adapter_cache: AdapterCache | None = None):
    """Prefill a list of requests and register the resulting batch as active."""
    # Ensure every adapter needed by this batch is loaded, evicting LRU if needed
    if adapter_cache is not None:
        try:
            adapter_cache.ensure_loaded([r.task_type for r in reqs])
        except Exception as e:
            print(f"[Router:sched] AdapterCache error: {e}")
            _deliver_errors_for_reqs(reqs, result_sock, str(e))
            return

    batch_id = str(uuid.uuid4())
    try:
        resp = client.prefill(
            batch_id=batch_id,
            request_ids=[r.request_id for r in reqs],
            prompts=[r.prompt for r in reqs],
            max_tokens=[r.max_tokens for r in reqs],
            adapter_ids=[r.task_type for r in reqs],
        )
    except Exception as e:
        print(f"[Router:sched] Prefill error: {e}")
        _deliver_errors_for_reqs(reqs, result_sock, str(e))
        return

    if resp.status != "accepted":
        print(f"[Router:sched] Prefill rejected: {resp.message}")
        _deliver_errors_for_reqs(reqs, result_sock, resp.message)
        return

    tracker = BatchTracker(
        batch_id=batch_id,
        request_ids=[r.request_id for r in reqs],
        adapter_ids=[r.task_type for r in reqs],
        max_tokens=[r.max_tokens for r in reqs],
    )
    active[batch_id] = tracker
    print(f"[Router:sched] admitted batch={batch_id} "
          f"n={len(reqs)} adapters={[r.task_type for r in reqs]}")


def _deliver_results(tracker: BatchTracker, result_sock: zmq.Socket):
    """Push one ModelResponse per completed request onto the result socket."""
    for req_id, text, n_gen, max_tok in zip(
            tracker.request_ids, tracker.generated_texts,
            tracker.tokens_generated, tracker.max_tokens):
        # Infer reason: if we stopped before hitting the limit → EOS ("stop")
        finish_reason = "length" if n_gen >= max_tok else "stop"
        response = ModelResponse(
            request_id=req_id,
            generated_text=text,
            finish_reason=finish_reason,
        )
        result_sock.send_json(response.model_dump())
        print(f"[Router:sched] delivered {req_id} "
              f"finish={finish_reason} tokens={n_gen}")


def _deliver_errors(tracker: BatchTracker, result_sock: zmq.Socket, error: str):
    """Push error ModelResponse for all requests in a failed batch."""
    for req_id in tracker.request_ids:
        result_sock.send_json(ModelResponse(
            request_id=req_id,
            generated_text="",
            finish_reason=f"error: {error}",
        ).model_dump())
        print(f"[Router:sched] error delivered {req_id}: {error}")


def _deliver_errors_for_reqs(reqs: list[TaggedRequest],
                              result_sock: zmq.Socket, error: str):
    """Push error ModelResponse for a list of TaggedRequests (pre-batch)."""
    for req in reqs:
        result_sock.send_json(ModelResponse(
            request_id=req.request_id,
            generated_text="",
            finish_reason=f"error: {error}",
        ).model_dump())
        print(f"[Router:sched] prefill error {req.request_id}: {error}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    zmq_ctx = zmq.Context()
    client = ModelServiceClient(GRPC_TARGET)
    pending: queue.Queue[TaggedRequest] = queue.Queue()

    t_recv = threading.Thread(
        target=receiver_thread,
        args=(zmq_ctx, pending),
        daemon=True, name="recv",
    )
    t_sched = threading.Thread(
        target=scheduler_thread,
        args=(pending, client, zmq_ctx),
        daemon=True, name="sched",
    )

    t_recv.start()
    t_sched.start()
    print("[Router] receiver + Orca scheduler running")
    t_recv.join()
    t_sched.join()


if __name__ == "__main__":
    main()
