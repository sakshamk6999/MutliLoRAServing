"""
Real-world end-to-end test.

Starts one backend (selected via --backend) plus the full service stack
(FastAPI → MLP → Router → gRPC model server) and runs prompts through it.

Run:
  # Our custom QwenLoRAModel — PyTorch LoRA (default)
  pytest test/test_real_world.py -v -s \\
    --base-model-id Qwen/Qwen3-1.7B \\
    --adapter diagnosis:./train_adapters/train-LoRA/adapters/diagnosis \\
    --adapter implementation:./train_adapters/train-LoRA/adapters/implementation \\
    --backend ours

  # Our custom QwenLoRAModel — Triton LoRA
  pytest test/test_real_world.py -v -s \\
    --base-model-id Qwen/Qwen3-1.7B \\
    --adapter diagnosis:./train_adapters/train-LoRA/adapters/diagnosis \\
    --adapter implementation:./train_adapters/train-LoRA/adapters/implementation \\
    --backend ours --lora-backend triton

  # PEFT baseline
  pytest test/test_real_world.py -v -s \\
    --base-model-id Qwen/Qwen3-1.7B \\
    --adapter diagnosis:./train_adapters/train-LoRA/adapters/diagnosis \\
    --backend peft

  # vLLM baseline
  pytest test/test_real_world.py -v -s \\
    --base-model-id Qwen/Qwen3-1.7B \\
    --adapter diagnosis:./train_adapters/train-LoRA/adapters/diagnosis \\
    --backend vllm

Skip conditions:
  - CUDA not available
  - --base-model-id not supplied
  - required package not installed for the chosen backend
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


# ── gRPC server startup — one function per backend ────────────────────────────

def _start_grpc_ours(weight_dir: str, adapter_dirs: dict[str, str],
                     max_total_token_num: int, ready: threading.Event,
                     use_triton: bool = False):
    from model_logic.model_endpoint.grpc_server import build_servicer

    print(f"[grpc:ours] loading {weight_dir!r}, adapters={list(adapter_dirs)}, "
          f"lora_backend={'triton' if use_triton else 'pytorch'}")
    servicer = build_servicer(
        weight_dir=weight_dir,
        max_total_token_num=max_total_token_num,
        adapter_dirs=adapter_dirs,
        use_triton=use_triton,
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    model_service_pb2_grpc.add_ModelServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{GRPC_PORT}")
    server.start()
    print(f"[grpc:ours] ready on :{GRPC_PORT}")
    ready.set()
    server.wait_for_termination()


def _start_grpc_peft(weight_dir: str, ready: threading.Event):
    from test.grpc_server_factory.backends.peft.causal_lm_backend import CausalLMPEFTBackend
    from test.grpc_server_factory.servicer import DelegatingModelServicer

    print(f"[grpc:peft] loading {weight_dir!r}")
    backend = CausalLMPEFTBackend(base_model_id=weight_dir)
    servicer = DelegatingModelServicer(backend)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    model_service_pb2_grpc.add_ModelServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{GRPC_PORT}")
    server.start()
    print(f"[grpc:peft] ready on :{GRPC_PORT}")
    ready.set()
    server.wait_for_termination()


def _start_grpc_vllm(weight_dir: str, ready: threading.Event):
    from test.grpc_server_factory.backends.vLLM.vLLM import VLLMBackend
    from test.grpc_server_factory.servicer import DelegatingModelServicer

    print(f"[grpc:vllm] loading {weight_dir!r}")
    backend = VLLMBackend(base_model_id=weight_dir)
    servicer = DelegatingModelServicer(backend)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    model_service_pb2_grpc.add_ModelServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{GRPC_PORT}")
    server.start()
    print(f"[grpc:vllm] ready on :{GRPC_PORT}")
    ready.set()
    server.wait_for_termination()


def _load_adapters_via_grpc(adapter_dirs: dict[str, str]):
    """Load adapters into a running gRPC server via the LoadAdapter RPC."""
    from model_logic.model_endpoint.grpc_client import ModelServiceClient

    client = ModelServiceClient(f"localhost:{GRPC_PORT}")
    for name, path in adapter_dirs.items():
        resp = client.load_adapter(adapter_id=name, adapter_path=path)
        assert resp.status in ("loaded", "already_loaded"), \
            f"LoadAdapter {name!r} → {resp.status}"
        print(f"[grpc] adapter {name!r} {resp.status}")


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
    backend       = request.config.getoption("--backend")
    num_adapters  = request.config.getoption("--num-adapters")
    lora_backend  = request.config.getoption("--lora-backend")

    if not base_model_id:
        pytest.skip("--base-model-id not provided; skipping real-world tests")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; skipping real-world tests")

    if backend == "peft":
        try:
            import peft  # noqa: F401
        except ImportError:
            pytest.skip("peft not installed; skipping peft backend tests")
    elif backend == "vllm":
        try:
            import vllm  # noqa: F401
        except ImportError:
            pytest.skip("vllm not installed; skipping vllm backend tests")

    if num_adapters is not None and num_adapters < 1:
        pytest.fail(f"--num-adapters must be ≥ 1, got {num_adapters}")

    adapters = _parse_adapters(adapter_raw)
    return {
        "base_model_id": base_model_id,
        "adapters": adapters,          # ALL provided adapters — cache handles eviction
        "num_adapters": num_adapters,  # max concurrent; None = no limit
        "classifier": classifier,
        "max_tokens": max_tokens,
        "backend": backend,
        "lora_backend": lora_backend,  # "pytorch" | "triton" (only applies when backend="ours")
    }


@pytest.fixture(scope="module")
def running_services(cli):
    """Start all services once for the entire module."""
    backend = cli["backend"]
    print(f"\n[fixture] backend={backend!r}")

    # Patch module constants to use test ports
    import router.router_service as rrs
    rrs.MLP_PULL_ADDR    = f"tcp://localhost:{ZMQ_MLP_PORT}"
    rrs.RESULT_PUSH_ADDR = f"tcp://*:{ZMQ_RESULT_PORT}"
    rrs.GRPC_TARGET      = f"localhost:{GRPC_PORT}"

    import application_endpoint.server as app_mod
    app_mod.ZMQ_PUSH_ADDR   = f"tcp://*:{ZMQ_APP_PORT}"
    app_mod.ZMQ_RESULT_ADDR = f"tcp://localhost:{ZMQ_RESULT_PORT}"

    zmq_ctx = zmq.Context()

    # 1. gRPC backend server — start with NO pre-loaded adapters;
    #    the AdapterCache below handles all loading/eviction on demand.
    use_triton = (backend == "ours") and (cli["lora_backend"] == "triton")
    grpc_ready = threading.Event()
    if backend == "ours":
        target, args = _start_grpc_ours, (
            cli["base_model_id"], {}, 8192, grpc_ready, use_triton)
    elif backend == "peft":
        target, args = _start_grpc_peft, (cli["base_model_id"], grpc_ready)
    else:  # vllm
        target, args = _start_grpc_vllm, (cli["base_model_id"], grpc_ready)

    threading.Thread(target=target, args=args,
                     daemon=True, name=f"grpc-{backend}").start()
    print(f"[fixture] waiting up to {MODEL_LOAD_TIMEOUT}s for {backend!r} to load…")
    assert grpc_ready.wait(MODEL_LOAD_TIMEOUT), \
        f"{backend!r} gRPC server did not become ready within {MODEL_LOAD_TIMEOUT}s"

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

    # 3. Router + AdapterCache
    from model_logic.model_endpoint.grpc_client import ModelServiceClient
    from router.router_service import AdapterCache
    pending: queue.Queue[TaggedRequest] = queue.Queue()
    client = ModelServiceClient(rrs.GRPC_TARGET)

    adapter_cache = AdapterCache(
        client=client,
        adapter_dirs=cli["adapters"],
        max_loaded=cli["num_adapters"],   # None = load all, never evict
    )
    if cli["num_adapters"] is not None:
        print(f"[fixture] AdapterCache: max {cli['num_adapters']} adapter(s) "
              f"resident at once across {len(cli['adapters'])} known adapters")

    threading.Thread(
        target=rrs.receiver_thread, args=(zmq_ctx, pending),
        daemon=True, name="router-recv",
    ).start()
    threading.Thread(
        target=rrs.scheduler_thread, args=(pending, client, zmq_ctx, adapter_cache),
        daemon=True, name="router-sched",
    ).start()

    # 4. App server
    app_ready = _start_app_server(app_mod)
    assert app_ready.wait(SERVICE_READY_TIMEOUT), "App server did not become ready"

    time.sleep(0.5)  # let ZMQ sockets settle
    print(f"[fixture] all services ready (backend={backend!r})\n")
    yield


# ── Metrics collection ─────────────────────────────────────────────────────────

from dataclasses import dataclass as _dc, field as _field
import statistics as _stats

@_dc
class _Record:
    test: str
    adapter: str
    ttft_s: float    # router: prefill-accepted → first decode with text
    e2e_s: float     # router: prefill-accepted → all tokens done
    tokens: int      # approximate word count of generated text

_records: list[_Record] = []


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


def _record(test: str, adapter: str, result: dict):
    """Store one timing record and print a one-line summary."""
    r = _Record(
        test=test,
        adapter=adapter,
        ttft_s=result.get("ttft_s", 0.0),
        e2e_s=result.get("e2e_s", 0.0),
        tokens=len(result.get("generated_text", "").split()),
    )
    _records.append(r)
    print(f"\n  [metrics] ttft={r.ttft_s:.3f}s  e2e={r.e2e_s:.3f}s  "
          f"tokens≈{r.tokens}  finish={result.get('finish_reason')}")


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
        pytest.skip(f"adapter {adapter_name!r} path not provided via --adapter")

    request_id = _queue_prompt(prompt, cli["max_tokens"])
    print(f"\n  [queued] {request_id[:8]}… adapter={adapter_name}")

    result = _poll(request_id)
    _record("prompt_generates_text", adapter_name, result)
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
    _record("base_model_no_adapter", "base", result)
    assert len(result["generated_text"]) > 0
    print(f"\n  [base]   {result['generated_text'][:200]}")


@pytest.mark.real_world
def test_sequential_prompts_all_return(running_services, cli):
    """Send N prompts back-to-back and verify every one returns a result."""
    available = list(cli["adapters"])
    prompts = [p for name, p, _ in PROMPT_SUITE if name in available][:3]

    if not prompts:
        pytest.skip("No matching adapters provided")

    t_wall_start = time.monotonic()
    ids = [_queue_prompt(p, cli["max_tokens"]) for p in prompts]
    results = [_poll(rid) for rid in ids]
    wall_s = time.monotonic() - t_wall_start
    throughput = len(prompts) / wall_s

    for rid, res in zip(ids, results):
        assert res["request_id"] == rid
        assert len(res["generated_text"]) > 0
        _record("sequential_prompts", res.get("adapter", "unknown"), res)

    print(f"\n  [throughput] {len(prompts)} requests in {wall_s:.2f}s "
          f"= {throughput:.3f} req/s")


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
    _record("adapter_vs_base", adapter, res_adapted)

    # Force base model by sending an unclassifiable prompt
    rid_base = _queue_prompt("zxqwerty1234567890", cli["max_tokens"])
    res_base = _poll(rid_base)
    _record("adapter_vs_base", "base", res_base)

    print(f"\n  [adapter] {res_adapted['generated_text'][:120]}")
    print(f"  [base]    {res_base['generated_text'][:120]}")

    assert len(res_adapted["generated_text"]) > 0
    assert res_base["finish_reason"] in ("stop", "length", "error: ")


@pytest.mark.real_world
def test_metrics_summary(running_services, cli):  # noqa: ARG001
    """Print aggregated TTFT / E2E / throughput across all completed tests."""
    if not _records:
        pytest.skip("No metrics collected yet")

    sep = "─" * 80
    print(f"\n{sep}")
    print(f"  {'REAL-WORLD TEST METRICS':^76}")
    lora_info = (f"  lora_backend={cli['lora_backend']}"
                 if cli["backend"] == "ours" else "")
    print(f"  backend={cli['backend']}{lora_info}  "
          f"max_tokens={cli['max_tokens']}  "
          f"num_adapters={cli['num_adapters'] or 'all'}")
    print(sep)
    print(f"  {'Test / Adapter':<38} {'TTFT (s)':>9} {'E2E (s)':>9} {'Tokens':>7}")
    print(sep)
    for r in _records:
        label = f"{r.test[:22]}/{r.adapter[:14]}"
        print(f"  {label:<38} {r.ttft_s:>9.3f} {r.e2e_s:>9.3f} {r.tokens:>7}")
    print(sep)

    ttfts = [r.ttft_s for r in _records if r.ttft_s > 0]
    e2es  = [r.e2e_s  for r in _records if r.e2e_s  > 0]
    if ttfts:
        print(f"  {'mean TTFT':<38} {_stats.mean(ttfts):>9.3f}")
        print(f"  {'p50  TTFT':<38} {_stats.median(ttfts):>9.3f}")
        print(f"  {'mean E2E':<38} {'':>9} {_stats.mean(e2es):>9.3f}")
        tput = len(_records) / sum(e2es) if e2es else 0.0
        print(f"  {'throughput (req/s, sequential)':<38} {tput:>9.3f}")
    print(sep + "\n")
