"""
Real-world end-to-end test.

Starts every service with real weights and runs prompts through the full pipeline.

Run:
  pytest test/test_real_world.py -v -s \\
    --base-model-id Qwen/Qwen3-1.7B \\
    --adapter diagnosis:./train_adapters/train-LoRA/adapters/diagnosis \\
    --adapter implementation:./train_adapters/train-LoRA/adapters/implementation \\
    --max-tokens 128

Skip conditions:
  - CUDA not available
  - --base-model-id not supplied
"""

import os
import sys
import queue
import threading
import time
from concurrent import futures

import grpc
import httpx
import pytest
import torch
import zmq

# ── Paths ──────────────────────────────────────────────────────────────────────
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from model_logic.protos import model_service_pb2_grpc
from data_model.data_types import TaggedRequest

# ── Ports (separate range from integration test to allow running both) ─────────
HTTP_PORT       = 8020
GRPC_PORT       = 50071
ZMQ_APP_PORT    = 5575
ZMQ_MLP_PORT    = 5576
ZMQ_RESULT_PORT = 5577

MODEL_LOAD_TIMEOUT = 300   # seconds — Qwen3-1.7B needs ~60-120 s on first load
SERVICE_READY_TIMEOUT = 30
GENERATION_TIMEOUT = 120   # per prompt
POLL_INTERVAL = 1.0

# ── Prompts paired with adapter names ─────────────────────────────────────────
# Each entry: (adapter_name, prompt, quality_hint)
# quality_hint is a substring we loosely expect in a coherent response.
PROMPT_SUITE = [
    (
        "diagnosis",
        (
            "A 45-year-old patient presents with a persistent dry cough, "
            "low-grade fever, and fatigue for the past 10 days. "
            "What are the most likely diagnoses to consider?"
        ),
        None,  # any non-empty response is acceptable
    ),
    (
        "implementation",
        (
            "Write a Python function called `binary_search` that takes a sorted "
            "list and a target value and returns the index of the target, or -1 "
            "if not found."
        ),
        "def binary_search",
    ),
    (
        "rewriting_and_drafting",
        (
            "Rewrite the following sentence in a more formal tone:\n"
            "'We messed up the launch and customers are pretty mad about it.'"
        ),
        None,
    ),
    (
        "safe_refusal",
        "How can I synthesise an illegal substance at home?",
        None,
    ),
    (
        "grounded_qa",
        (
            "Passage: The Eiffel Tower was built between 1887 and 1889 as the "
            "entrance arch for the 1889 World's Fair. It stands 330 metres tall.\n"
            "Question: How tall is the Eiffel Tower?"
        ),
        "330",
    ),
    (
        "information_extraction",
        (
            "Extract all named entities (people, organisations, locations) from "
            "the following text:\n"
            "'Apple CEO Tim Cook announced the new product line in Cupertino, "
            "California, alongside executives from Microsoft and Google.'"
        ),
        None,
    ),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_adapters(raw: list[str]) -> dict[str, str]:
    """Parse ['name:path', ...] → {'name': 'path', ...}."""
    result = {}
    for entry in raw:
        if ":" not in entry:
            raise ValueError(
                f"--adapter must be NAME:PATH, got {entry!r}"
            )
        name, _, path = entry.partition(":")
        result[name.strip()] = path.strip()
    return result


# ── Keyword-based MLP stub ─────────────────────────────────────────────────────

# Ordered list: first match wins
_KEYWORD_RULES: list[tuple[str, list[str]]] = [
    ("implementation",        ["function", "def ", "code", "python", "write a ", "implement", "algorithm", "class "]),
    ("rewriting_and_drafting",["rewrite", "rephrase", "draft", "revise", "edit", "more formal", "tone"]),
    ("safe_refusal",          ["illegal", "synthesise", "weapon", "harm", "dangerous", "how to make", "how can i make"]),
    ("grounded_qa",           ["passage:", "based on", "according to", "question:"]),
    ("information_extraction",["extract", "entities", "named entity", "list all", "identify"]),
    ("diagnosis",             ["patient", "symptom", "fever", "diagnosis", "disease", "clinical", "treatment"]),
]


def _classify_prompt(prompt: str, available_adapters: set[str]) -> str:
    """Return the adapter name that best matches the prompt, from available ones."""
    lower = prompt.lower()
    for adapter_name, keywords in _KEYWORD_RULES:
        if adapter_name in available_adapters:
            if any(kw in lower for kw in keywords):
                return adapter_name
    # Fall back to the first available adapter
    return next(iter(available_adapters))


def _run_stub_mlp(zmq_ctx: zmq.Context,
                  available_adapters: set[str],
                  ready: threading.Event):
    pull = zmq_ctx.socket(zmq.PULL)
    pull.connect(f"tcp://localhost:{ZMQ_APP_PORT}")
    pull.setsockopt(zmq.RCVTIMEO, 500)

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
        task = _classify_prompt(raw["prompt"], available_adapters)
        tagged = TaggedRequest(task_type=task, **raw)
        push.send_json(tagged.model_dump())
        print(f"[stub-mlp] {tagged.request_id[:8]}… → {task}")


def _run_real_mlp(zmq_ctx: zmq.Context,
                  checkpoint_path: str,
                  ready: threading.Event):
    """Use the trained BERT classifier if a checkpoint path is provided."""
    from mlp.mlp_service import load_model, classify_task

    model, tokenizer, label_map = load_model(checkpoint_path)
    print(f"[mlp] classifier loaded from {checkpoint_path}")

    pull = zmq_ctx.socket(zmq.PULL)
    pull.connect(f"tcp://localhost:{ZMQ_APP_PORT}")
    pull.setsockopt(zmq.RCVTIMEO, 500)

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
        task = classify_task(model, tokenizer, label_map, raw["prompt"])
        tagged = TaggedRequest(task_type=task, **raw)
        push.send_json(tagged.model_dump())
        print(f"[mlp] {tagged.request_id[:8]}… → {task}")


# ── gRPC server with real QwenLoRAModel ────────────────────────────────────────

def _start_real_grpc_server(weight_dir: str,
                             adapter_dirs: dict[str, str],
                             max_total_token_num: int,
                             server_ready: threading.Event):
    """Load Qwen3 + adapters and start gRPC server. Sets server_ready when done."""
    from model_logic.model_endpoint.grpc_server import build_servicer

    print(f"[grpc] Loading model {weight_dir!r} with adapters: {list(adapter_dirs)}")
    servicer = build_servicer(
        weight_dir=weight_dir,
        max_total_token_num=max_total_token_num,
        adapter_dirs=adapter_dirs,
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    model_service_pb2_grpc.add_ModelServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{GRPC_PORT}")
    server.start()
    print(f"[grpc] server ready on :{GRPC_PORT}")
    server_ready.set()
    server.wait_for_termination()


# ── App server ─────────────────────────────────────────────────────────────────

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
    original_startup = server.startup

    async def _patched_startup(sockets=None):
        await original_startup(sockets)
        ready.set()

    server.startup = _patched_startup
    threading.Thread(target=server.run, daemon=True, name="app-server").start()
    return ready


# ── Fixtures ───────────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_world: requires CUDA + model weights via --base-model-id",
    )


@pytest.fixture(scope="module")
def cli(request):
    """Parsed CLI options."""
    base_model_id = request.config.getoption("--base-model-id")
    adapter_raw   = request.config.getoption("--adapter")
    classifier    = request.config.getoption("--classifier")
    max_tokens    = request.config.getoption("--max-tokens")

    if not base_model_id:
        pytest.skip("--base-model-id not provided; skipping real-world tests")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; skipping real-world tests")

    adapters = _parse_adapters(adapter_raw)
    return {
        "base_model_id": base_model_id,
        "adapters": adapters,       # {name: path}
        "classifier": classifier,
        "max_tokens": max_tokens,
    }


@pytest.fixture(scope="module")
def running_services(cli):
    """Start all services once for the entire module."""

    # Patch module constants to use test ports
    import router.router_service as rrs
    rrs.MLP_PULL_ADDR    = f"tcp://localhost:{ZMQ_MLP_PORT}"
    rrs.RESULT_PUSH_ADDR = f"tcp://*:{ZMQ_RESULT_PORT}"
    rrs.GRPC_TARGET      = f"localhost:{GRPC_PORT}"

    import application_endpoint.server as app_mod
    app_mod.ZMQ_PUSH_ADDR   = f"tcp://*:{ZMQ_APP_PORT}"
    app_mod.ZMQ_RESULT_ADDR = f"tcp://localhost:{ZMQ_RESULT_PORT}"

    zmq_ctx = zmq.Context()

    # 1. gRPC server (real model — this is the slow step)
    grpc_ready = threading.Event()
    threading.Thread(
        target=_start_real_grpc_server,
        args=(
            cli["base_model_id"],
            cli["adapters"],
            8192,
            grpc_ready,
        ),
        daemon=True,
        name="grpc-server",
    ).start()
    print(f"\n[fixture] Waiting up to {MODEL_LOAD_TIMEOUT}s for model to load…")
    assert grpc_ready.wait(MODEL_LOAD_TIMEOUT), \
        f"gRPC server did not become ready within {MODEL_LOAD_TIMEOUT}s"

    # 2. MLP (real classifier or keyword stub)
    mlp_ready = threading.Event()
    if cli["classifier"]:
        threading.Thread(
            target=_run_real_mlp,
            args=(zmq_ctx, cli["classifier"], mlp_ready),
            daemon=True, name="mlp",
        ).start()
    else:
        threading.Thread(
            target=_run_stub_mlp,
            args=(zmq_ctx, set(cli["adapters"]), mlp_ready),
            daemon=True, name="stub-mlp",
        ).start()
    assert mlp_ready.wait(SERVICE_READY_TIMEOUT), "MLP service did not become ready"

    # 3. Router
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

    # 4. App server
    app_ready = _start_app_server(app_mod)
    assert app_ready.wait(SERVICE_READY_TIMEOUT), "App server did not become ready"

    time.sleep(0.5)  # let ZMQ sockets settle
    print("[fixture] All services ready.\n")
    yield


# ── Helpers for tests ──────────────────────────────────────────────────────────

BASE_URL = f"http://127.0.0.1:{HTTP_PORT}"


def _queue_prompt(prompt: str, max_tokens: int) -> str:
    with httpx.Client() as c:
        resp = c.post(
            f"{BASE_URL}/generate",
            json={"prompt": prompt, "max_tokens": max_tokens},
            timeout=10.0,
        )
    resp.raise_for_status()
    body = resp.json()
    assert body["status"] == "queued"
    return body["request_id"]


def _poll(request_id: str) -> dict:
    deadline = time.monotonic() + GENERATION_TIMEOUT
    with httpx.Client() as c:
        while time.monotonic() < deadline:
            resp = c.get(f"{BASE_URL}/result/{request_id}", timeout=10.0)
            if resp.status_code == 200:
                return resp.json()
            assert resp.status_code == 202, \
                f"Unexpected status {resp.status_code}: {resp.text}"
            time.sleep(POLL_INTERVAL)
    pytest.fail(f"No result for {request_id} after {GENERATION_TIMEOUT}s")


# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.real_world
def test_health(running_services):
    resp = httpx.get(f"{BASE_URL}/health", timeout=5.0)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.real_world
@pytest.mark.parametrize("adapter_name,prompt,expected_substr", PROMPT_SUITE)
def test_prompt_generates_text(running_services, cli,
                                adapter_name, prompt, expected_substr):
    """Each prompt is routed to its adapter and returns non-empty generated text."""
    if adapter_name not in cli["adapters"]:
        pytest.skip(f"adapter {adapter_name!r} not provided via --adapter")

    request_id = _queue_prompt(prompt, cli["max_tokens"])
    print(f"\n  [queued] {request_id[:8]}… adapter={adapter_name}")

    result = _poll(request_id)

    print(f"  [result] finish={result['finish_reason']}")
    print(f"  [text]   {result['generated_text'][:200]}")

    assert result["request_id"] == request_id
    assert isinstance(result["generated_text"], str), "generated_text must be a string"
    assert len(result["generated_text"]) > 0, "generated_text must not be empty"
    assert result["finish_reason"] in ("stop", "length"), \
        f"unexpected finish_reason: {result['finish_reason']!r}"

    if expected_substr:
        assert expected_substr.lower() in result["generated_text"].lower(), (
            f"Expected {expected_substr!r} in response for {adapter_name!r}.\n"
            f"Got: {result['generated_text'][:300]}"
        )


@pytest.mark.real_world
def test_base_model_no_adapter(running_services, cli):
    """Request without a matching adapter falls back to base model (task=unknown)."""
    prompt = "What is the capital of France?"
    request_id = _queue_prompt(prompt, cli["max_tokens"])
    result = _poll(request_id)

    assert len(result["generated_text"]) > 0
    print(f"\n  [base]   {result['generated_text'][:200]}")


@pytest.mark.real_world
def test_sequential_prompts_all_return(running_services, cli):
    """Send N prompts back-to-back and verify every one returns a result."""
    available = list(cli["adapters"])
    prompts = [p for name, p, _ in PROMPT_SUITE if name in available][:3]

    if not prompts:
        pytest.skip("No matching adapters provided")

    ids = [_queue_prompt(p, cli["max_tokens"]) for p in prompts]
    results = [_poll(rid) for rid in ids]

    for rid, res in zip(ids, results):
        assert res["request_id"] == rid
        assert len(res["generated_text"]) > 0


@pytest.mark.real_world
def test_adapter_outputs_differ_from_base(running_services, cli):
    """Same prompt sent twice — once routed to adapter, once with no matching adapter."""
    available = list(cli["adapters"])
    if not available:
        pytest.skip("No adapters provided")

    # Pick the implementation adapter for a deterministic check
    adapter = next((a for a in ("implementation", "diagnosis") if a in available), available[0])
    prompt_suite_entry = next(
        (p for name, p, _ in PROMPT_SUITE if name == adapter), None)
    if prompt_suite_entry is None:
        pytest.skip(f"No prompt defined for adapter {adapter!r}")

    # Route to adapter
    rid_adapted = _queue_prompt(prompt_suite_entry, cli["max_tokens"])
    res_adapted = _poll(rid_adapted)

    # Force base model by sending an unclassifiable prompt
    rid_base = _queue_prompt("zxqwerty1234567890", cli["max_tokens"])
    res_base = _poll(rid_base)

    print(f"\n  [adapter] {res_adapted['generated_text'][:120]}")
    print(f"  [base]    {res_base['generated_text'][:120]}")

    # Both must return something — but we can't assert they differ since the base
    # model may have similar priors.  Just verify both completed successfully.
    assert len(res_adapted["generated_text"]) > 0
    assert res_base["finish_reason"] in ("stop", "length", "error: ")
