"""
Integration test: one simulated client, all services in-process threads.

Services used:
  - SmartStubBackend  gRPC server  (no real model; returns fake text after 1 decode step)
  - Stub MLP thread               (tags every request as "diagnosis", no real BERT model)
  - Router (Orca scheduler)       (real code, test ports)
  - FastAPI app server            (real code, test ports)
  - httpx client                  (POST /generate  →  GET /result/{id})

Ports (non-default so they don't clash with any live services):
  HTTP     8010
  gRPC     50061
  ZMQ app→mlp       5565
  ZMQ mlp→router    5566
  ZMQ result        5567
"""

import sys
import os
import time
import queue
import threading
from concurrent import futures

import grpc
import pytest
import httpx
import zmq

# ── Project root on sys.path ───────────────────────────────────────────────────
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from model_logic.protos import model_service_pb2, model_service_pb2_grpc
from test.grpc_server_factory.servicer import DelegatingModelServicer
from data_model.data_types import TaggedRequest

# ── Test-specific ports ────────────────────────────────────────────────────────
HTTP_PORT       = 8010
GRPC_PORT       = 50061
ZMQ_APP_PORT    = 5565   # app server → MLP
ZMQ_MLP_PORT    = 5566   # MLP → router
ZMQ_RESULT_PORT = 5567   # router → app server (results)

READY_TIMEOUT = 10.0  # seconds to wait for each service to become ready


# ── Smart stub gRPC backend ────────────────────────────────────────────────────

class SmartStubBackend:
    """Accepts prefill, returns fake generated text on first decode call."""

    def __init__(self):
        self._batches: dict[str, list[str]] = {}  # batch_id → request_ids

    def load_adapter(self, request):
        return model_service_pb2.LoadAdapterResponse(
            adapter_id=request.adapter_id, status="loaded")

    def unload_adapter(self, request):
        return model_service_pb2.UnloadAdapterResponse(
            adapter_id=request.adapter_id, status="unloaded")

    def prefill(self, request):
        n = len(request.request_ids)
        if not n == len(request.prompts) == len(request.max_tokens) == len(request.adapter_ids):
            return model_service_pb2.PrefillResponse(
                batch_id=request.batch_id, status="error",
                message="length mismatch")
        self._batches[request.batch_id] = list(request.request_ids)
        return model_service_pb2.PrefillResponse(
            batch_id=request.batch_id, status="accepted", message="")

    def decode(self, request):
        request_ids = self._batches.pop(request.batch_id, [])
        n = len(request_ids)
        return model_service_pb2.DecodeResponse(
            batch_id=request.batch_id,
            request_ids=request_ids,
            generated_texts=[f"stub response for {rid}" for rid in request_ids],
            is_finished=[True] * n,
        )


# ── Stub MLP service (no real BERT) ───────────────────────────────────────────

def _run_stub_mlp(zmq_ctx: zmq.Context, ready: threading.Event):
    """Pull raw requests from app server, tag task_type='diagnosis', push to router."""
    pull = zmq_ctx.socket(zmq.PULL)
    pull.connect(f"tcp://localhost:{ZMQ_APP_PORT}")
    pull.setsockopt(zmq.RCVTIMEO, 300)

    push = zmq_ctx.socket(zmq.PUSH)
    push.bind(f"tcp://*:{ZMQ_MLP_PORT}")

    ready.set()
    while True:
        try:
            raw = pull.recv_json()
        except zmq.error.Again:
            continue
        except zmq.error.ZMQError:
            break
        tagged = TaggedRequest(task_type="diagnosis", **raw)
        push.send_json(tagged.model_dump())


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def services():
    """Start all services in background threads; yield; nothing to tear down (daemon)."""

    # 1. Patch router + app server module constants to use test ports
    import router.router_service as rrs
    rrs.MLP_PULL_ADDR    = f"tcp://localhost:{ZMQ_MLP_PORT}"
    rrs.RESULT_PUSH_ADDR = f"tcp://*:{ZMQ_RESULT_PORT}"
    rrs.GRPC_TARGET      = f"localhost:{GRPC_PORT}"

    import application_endpoint.server as app_mod
    app_mod.ZMQ_PUSH_ADDR   = f"tcp://*:{ZMQ_APP_PORT}"
    app_mod.ZMQ_RESULT_ADDR = f"tcp://localhost:{ZMQ_RESULT_PORT}"

    zmq_ctx = zmq.Context()

    # 2. Start gRPC stub server
    _start_grpc_server(zmq_ctx)

    # 3. Start stub MLP
    mlp_ready = threading.Event()
    threading.Thread(
        target=_run_stub_mlp, args=(zmq_ctx, mlp_ready), daemon=True, name="stub-mlp"
    ).start()
    assert mlp_ready.wait(READY_TIMEOUT), "stub MLP did not become ready"

    # 4. Start router (receiver + scheduler threads)
    _start_router(zmq_ctx)

    # 5. Start FastAPI app server (uvicorn in background thread)
    app_ready = _start_app_server(app_mod)

    assert app_ready.wait(READY_TIMEOUT), "app server did not become ready"
    time.sleep(0.3)  # let ZMQ sockets settle

    yield


def _start_grpc_server(zmq_ctx: zmq.Context):
    backend = SmartStubBackend()
    servicer = DelegatingModelServicer(backend)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    model_service_pb2_grpc.add_ModelServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{GRPC_PORT}")
    server.start()
    # server.wait_for_termination() would block; don't call it — it runs in background


def _start_router(zmq_ctx: zmq.Context):
    import router.router_service as rrs
    from model_logic.model_endpoint.grpc_client import ModelServiceClient

    pending: queue.Queue[TaggedRequest] = queue.Queue()
    client = ModelServiceClient(rrs.GRPC_TARGET)

    threading.Thread(
        target=rrs.receiver_thread, args=(zmq_ctx, pending),
        daemon=True, name="router-recv",
    ).start()
    threading.Thread(
        target=rrs.scheduler_thread, args=(pending, client, zmq_ctx),
        daemon=True, name="router-sched",
    ).start()


def _start_app_server(app_mod) -> threading.Event:
    import uvicorn

    ready = threading.Event()
    config = uvicorn.Config(
        app_mod.app,
        host="127.0.0.1",
        port=HTTP_PORT,
        log_level="warning",
        loop="asyncio",
    )
    server = uvicorn.Server(config)

    # uvicorn sets server.started when ready
    original_startup = server.startup

    async def _startup_hook(sockets=None):
        await original_startup(sockets)
        ready.set()

    server.startup = _startup_hook

    threading.Thread(
        target=server.run, daemon=True, name="app-server"
    ).start()
    return ready


# ── Test ───────────────────────────────────────────────────────────────────────

BASE_URL = f"http://127.0.0.1:{HTTP_PORT}"
POLL_INTERVAL = 0.3
POLL_TIMEOUT  = 15.0


def _poll_result(client: httpx.Client, request_id: str) -> dict:
    deadline = time.monotonic() + POLL_TIMEOUT
    while time.monotonic() < deadline:
        resp = client.get(f"{BASE_URL}/result/{request_id}", timeout=5.0)
        if resp.status_code == 200:
            return resp.json()
        assert resp.status_code == 202, f"unexpected status {resp.status_code}: {resp.text}"
        time.sleep(POLL_INTERVAL)
    pytest.fail(f"Result for {request_id} not ready after {POLL_TIMEOUT}s")


def test_health_check(services):
    with httpx.Client() as client:
        resp = client.get(f"{BASE_URL}/health", timeout=5.0)
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_single_client_generate_and_receive(services):
    """One client sends a prompt and gets back generated text via polling."""
    prompt = "Describe the symptoms of a common cold."

    with httpx.Client() as client:
        # 1. Send request
        post_resp = client.post(
            f"{BASE_URL}/generate",
            json={"prompt": prompt, "max_tokens": 64},
            timeout=5.0,
        )
    assert post_resp.status_code == 200
    body = post_resp.json()
    assert body["status"] == "queued"
    request_id = body["request_id"]
    assert request_id  # non-empty UUID

    # 2. Poll until result arrives
    with httpx.Client() as client:
        result = _poll_result(client, request_id)

    # 3. Assertions on result shape
    assert result["request_id"] == request_id
    assert isinstance(result["generated_text"], str)
    assert len(result["generated_text"]) > 0
    assert result["finish_reason"] in ("stop", "length")


def test_result_contains_stub_text(services):
    """The stub backend returns a predictable response we can assert on."""
    with httpx.Client() as client:
        post_resp = client.post(
            f"{BASE_URL}/generate",
            json={"prompt": "Hello world", "max_tokens": 32},
            timeout=5.0,
        )
        assert post_resp.status_code == 200
        request_id = post_resp.json()["request_id"]
        result = _poll_result(client, request_id)

    assert f"stub response for {request_id}" == result["generated_text"]


def test_result_consumed_after_poll(services):
    """Once a result has been retrieved it is removed from the store (no double-delivery)."""
    with httpx.Client() as client:
        post_resp = client.post(
            f"{BASE_URL}/generate",
            json={"prompt": "Test idempotency", "max_tokens": 16},
            timeout=5.0,
        )
        request_id = post_resp.json()["request_id"]

        # First poll — should return 200
        result = _poll_result(client, request_id)
        assert result["request_id"] == request_id

        # Second poll — result was deleted, should return 202 (pending) or 404
        resp2 = client.get(f"{BASE_URL}/result/{request_id}", timeout=5.0)
    assert resp2.status_code in (202, 404)
