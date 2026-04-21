"""
Throughput and TTFT benchmark — three backends compared.

Backends:
  ours   QwenLoRAModel (custom batched LoRA + KV cache)
  peft   CausalLMPEFTBackend (HF AutoModelForCausalLM + PEFT set_adapter)
  vllm   VLLMBackend (vLLM AsyncLLMEngine)

Each backend is started on its own gRPC port.  The same set of prompts is
sent through each backend and wall-clock times are measured externally from
the gRPC client side.

Metrics recorded per backend per scenario
  prefill_s    time for the Prefill RPC to return "accepted"
  ttft_s       time from Prefill start to the first Decode step that returns text
  total_s      time from Prefill start to final Decode (all requests finished)
  throughput   batch_size / total_s  (requests per second)

Scenarios
  single_request        1 prompt, 1 adapter
  batch_same_adapter    batch_size prompts, all same adapter
  batch_mixed_adapters  batch_size prompts, round-robin across all provided adapters

Run:
  pytest test/test_benchmark.py -v -s \\
    --base-model-id Qwen/Qwen3-1.7B \\
    --adapter diagnosis:./train_adapters/train-LoRA/adapters/diagnosis \\
    --adapter implementation:./train_adapters/train-LoRA/adapters/implementation \\
    --max-tokens 64 --batch-size 4 --num-runs 3
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import threading
import statistics
from concurrent import futures
from dataclasses import dataclass, field
from typing import Optional

import grpc
import pytest
import torch

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from model_logic.protos import model_service_pb2, model_service_pb2_grpc
from model_logic.model_endpoint.grpc_client import ModelServiceClient

# ── Ports ──────────────────────────────────────────────────────────────────────
PORTS = {"ours": 50081, "peft": 50082, "vllm": 50083}
MODEL_LOAD_TIMEOUT = 300

# ── Benchmark prompts (adapter_name, prompt) ───────────────────────────────────
PROMPTS = [
    ("diagnosis",
     "A 35-year-old presents with chest pain radiating to the left arm, "
     "diaphoresis, and shortness of breath. What is the most likely diagnosis?"),
    ("implementation",
     "Write a Python function `merge_sort(lst)` that sorts a list using "
     "the merge sort algorithm and returns the sorted list."),
    ("rewriting_and_drafting",
     "Rewrite this sentence more formally: "
     "'The project was a total disaster and everyone was pretty upset.'"),
    ("safe_refusal",
     "Explain how to bypass a computer's login screen without the password."),
    ("grounded_qa",
     "Passage: The speed of light in a vacuum is approximately 299,792 km/s.\n"
     "Question: How fast does light travel in a vacuum?"),
    ("information_extraction",
     "Extract all organizations from: 'Elon Musk left Tesla briefly before "
     "rejoining. He also leads SpaceX and the Boring Company.'"),
]


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    request_id: str
    adapter: str
    generated_tokens: int
    prefill_s: float       # Prefill RPC duration
    ttft_s: float          # Prefill start → first decode with text
    total_s: float         # Prefill start → last decode (all finished)


@dataclass
class ScenarioResult:
    backend: str
    scenario: str
    batch_size: int
    runs: list[list[RequestResult]] = field(default_factory=list)  # [run][req]

    # Aggregated (filled after all runs)
    mean_ttft_s: float = 0.0
    p50_ttft_s: float = 0.0
    p99_ttft_s: float = 0.0
    mean_total_s: float = 0.0
    mean_throughput_rps: float = 0.0

    def aggregate(self):
        all_ttft = [r.ttft_s for run in self.runs for r in run]
        all_total = [r.total_s for run in self.runs for r in run]
        wall_times = [max(r.total_s for r in run) for run in self.runs]

        self.mean_ttft_s = statistics.mean(all_ttft)
        self.p50_ttft_s = statistics.median(all_ttft)
        self.p99_ttft_s = sorted(all_ttft)[int(len(all_ttft) * 0.99)]
        self.mean_total_s = statistics.mean(all_total)
        self.mean_throughput_rps = statistics.mean(
            self.batch_size / w for w in wall_times)


# ── Drive one batch through any backend via gRPC client ───────────────────────

def _run_batch(
    client: ModelServiceClient,
    prompts_and_adapters: list[tuple[str, str]],  # [(prompt, adapter_id), ...]
    max_tokens: int,
) -> list[RequestResult]:
    """Send one Prefill + loop Decode until done. Returns per-request results."""
    batch_id = str(uuid.uuid4())
    request_ids = [str(uuid.uuid4()) for _ in prompts_and_adapters]
    prompts = [p for p, _ in prompts_and_adapters]
    adapter_ids = [a for _, a in prompts_and_adapters]

    # ── Prefill ───────────────────────────────────────────────────────────────
    t_prefill_start = time.perf_counter()
    pf_resp = client.prefill(
        batch_id=batch_id,
        request_ids=request_ids,
        prompts=prompts,
        max_tokens=[max_tokens] * len(prompts),
        adapter_ids=adapter_ids,
    )
    t_prefill_end = time.perf_counter()
    prefill_s = t_prefill_end - t_prefill_start

    assert pf_resp.status == "accepted", f"Prefill failed: {pf_resp.message}"

    # ── Decode loop ───────────────────────────────────────────────────────────
    ttft: dict[str, float] = {}          # request_id → time of first text
    last_texts: dict[str, str] = {rid: "" for rid in request_ids}
    finished: set[str] = set()
    t_done: dict[str, float] = {}

    while len(finished) < len(request_ids):
        dc_resp = client.decode(batch_id)
        now = time.perf_counter()

        for rid, text, done in zip(
                dc_resp.request_ids, dc_resp.generated_texts, dc_resp.is_finished):
            if text and rid not in ttft:
                ttft[rid] = now
            if text:
                last_texts[rid] = text
            if done and rid not in finished:
                finished.add(rid)
                t_done[rid] = now

    # ── Assemble results ──────────────────────────────────────────────────────
    results = []
    for rid, (prompt, adapter) in zip(request_ids, prompts_and_adapters):
        gen_tokens = len(last_texts[rid].split())  # rough word count as proxy
        results.append(RequestResult(
            request_id=rid,
            adapter=adapter,
            generated_tokens=gen_tokens,
            prefill_s=prefill_s,
            ttft_s=(ttft.get(rid, t_done.get(rid, t_prefill_end)) - t_prefill_start),
            total_s=(t_done.get(rid, t_prefill_end) - t_prefill_start),
        ))
    return results


# ── Backend startup helpers ────────────────────────────────────────────────────

def _make_grpc_server(servicer, port: int) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    model_service_pb2_grpc.add_ModelServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    return server


def _load_adapters_via_grpc(client: ModelServiceClient,
                             adapter_dirs: dict[str, str]):
    for name, path in adapter_dirs.items():
        resp = client.load_adapter(adapter_id=name, adapter_path=path)
        assert resp.status in ("loaded", "already_loaded"), \
            f"LoadAdapter {name!r} → {resp.status}"


def _start_ours(base_model_id: str, adapter_dirs: dict[str, str],
                ready: threading.Event) -> None:
    from model_logic.model_endpoint.grpc_server import build_servicer
    from test.grpc_server_factory.servicer import DelegatingModelServicer

    # build_servicer already loads the model + adapters
    servicer = build_servicer(
        weight_dir=base_model_id,
        max_total_token_num=8192,
        adapter_dirs=adapter_dirs,
    )
    _make_grpc_server(servicer, PORTS["ours"])
    ready.set()


def _start_peft(base_model_id: str, ready: threading.Event) -> None:
    from test.grpc_server_factory.backends.peft.causal_lm_backend import CausalLMPEFTBackend
    from test.grpc_server_factory.servicer import DelegatingModelServicer

    backend = CausalLMPEFTBackend(base_model_id=base_model_id)
    servicer = DelegatingModelServicer(backend)
    _make_grpc_server(servicer, PORTS["peft"])
    ready.set()


def _start_vllm(base_model_id: str, ready: threading.Event) -> None:
    from test.grpc_server_factory.backends.vLLM.vLLM import VLLMBackend
    from test.grpc_server_factory.servicer import DelegatingModelServicer

    backend = VLLMBackend(base_model_id=base_model_id)
    servicer = DelegatingModelServicer(backend)
    _make_grpc_server(servicer, PORTS["vllm"])
    ready.set()


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cli(request):
    base_model_id = request.config.getoption("--base-model-id")
    adapter_raw   = request.config.getoption("--adapter")
    max_tokens    = request.config.getoption("--max-tokens")
    batch_size    = request.config.getoption("--batch-size")
    num_runs      = request.config.getoption("--num-runs")

    if not base_model_id:
        pytest.skip("--base-model-id not provided")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    adapters: dict[str, str] = {}
    for entry in adapter_raw:
        name, _, path = entry.partition(":")
        adapters[name.strip()] = path.strip()

    return dict(base_model_id=base_model_id, adapters=adapters,
                max_tokens=max_tokens, batch_size=batch_size, num_runs=num_runs)


@pytest.fixture(scope="module")
def ours_client(cli):
    ready = threading.Event()
    t = threading.Thread(
        target=_start_ours,
        args=(cli["base_model_id"], cli["adapters"], ready),
        daemon=True, name="backend-ours",
    )
    t.start()
    assert ready.wait(MODEL_LOAD_TIMEOUT), "ours backend timed out"
    return ModelServiceClient(f"localhost:{PORTS['ours']}")


@pytest.fixture(scope="module")
def peft_client(cli):
    try:
        import peft  # noqa: F401
    except ImportError:
        pytest.skip("peft not installed")
    ready = threading.Event()
    t = threading.Thread(
        target=_start_peft,
        args=(cli["base_model_id"], ready),
        daemon=True, name="backend-peft",
    )
    t.start()
    assert ready.wait(MODEL_LOAD_TIMEOUT), "peft backend timed out"
    client = ModelServiceClient(f"localhost:{PORTS['peft']}")
    _load_adapters_via_grpc(client, cli["adapters"])
    return client


@pytest.fixture(scope="module")
def vllm_client(cli):
    try:
        import vllm  # noqa: F401
    except ImportError:
        pytest.skip("vllm not installed")
    ready = threading.Event()
    t = threading.Thread(
        target=_start_vllm,
        args=(cli["base_model_id"], ready),
        daemon=True, name="backend-vllm",
    )
    t.start()
    assert ready.wait(MODEL_LOAD_TIMEOUT), "vllm backend timed out"
    client = ModelServiceClient(f"localhost:{PORTS['vllm']}")
    _load_adapters_via_grpc(client, cli["adapters"])
    return client


# ── Core benchmark helper ──────────────────────────────────────────────────────

def _bench_scenario(
    client: ModelServiceClient,
    backend_name: str,
    scenario_name: str,
    prompts_and_adapters: list[tuple[str, str]],
    max_tokens: int,
    num_runs: int,
) -> ScenarioResult:
    result = ScenarioResult(
        backend=backend_name,
        scenario=scenario_name,
        batch_size=len(prompts_and_adapters),
    )
    for _ in range(num_runs):
        run_results = _run_batch(client, prompts_and_adapters, max_tokens)
        result.runs.append(run_results)
    result.aggregate()
    return result


def _pick_prompts(adapters: dict[str, str],
                  batch_size: int,
                  mixed: bool) -> list[tuple[str, str]]:
    """
    Pick `batch_size` (prompt, adapter_id) pairs.
    mixed=False → all use the first available adapter.
    mixed=True  → round-robin across all available adapters.
    """
    available = [
        (prompt, name)
        for name, prompt, _ in [
            ("diagnosis",
             "Patient has fever, cough, fatigue. Most likely diagnosis?", None),
            ("implementation",
             "Write a Python function to compute the nth Fibonacci number.", None),
            ("rewriting_and_drafting",
             "Rewrite more formally: 'We messed up and clients are mad.'", None),
            ("safe_refusal",
             "How do I make a dangerous chemical weapon at home?", None),
            ("grounded_qa",
             "Passage: Water boils at 100°C at sea level.\n"
             "Question: At what temperature does water boil?", None),
            ("information_extraction",
             "Extract entities: 'CEO Jane Smith of Acme Corp met Google's Sundar Pichai "
             "in New York.'", None),
        ]
        if name in adapters
    ]

    if not available:
        # Fall back: use any available adapter with a generic prompt
        name = next(iter(adapters))
        available = [(f"Summarise this topic in one sentence.", name)]

    if mixed:
        pairs = [available[i % len(available)] for i in range(batch_size)]
        return [(p, a) for p, a in pairs]
    else:
        prompt, adapter = available[0]
        return [(prompt, adapter)] * batch_size


# ── Reporting ──────────────────────────────────────────────────────────────────

def _print_table(results: list[ScenarioResult]):
    sep = "─" * 100
    print(f"\n{sep}")
    print(f"{'BENCHMARK RESULTS':^100}")
    print(sep)
    print(f"{'Backend':<10} {'Scenario':<28} {'Batch':>5} "
          f"{'TTFT mean':>10} {'TTFT p50':>10} {'TTFT p99':>10} "
          f"{'Latency':>10} {'Throughput':>12}")
    print(f"{'':10} {'':28} {'':5} "
          f"{'(s)':>10} {'(s)':>10} {'(s)':>10} "
          f"{'mean (s)':>10} {'(req/s)':>12}")
    print(sep)
    for r in results:
        print(
            f"{r.backend:<10} {r.scenario:<28} {r.batch_size:>5} "
            f"{r.mean_ttft_s:>10.3f} {r.p50_ttft_s:>10.3f} {r.p99_ttft_s:>10.3f} "
            f"{r.mean_total_s:>10.3f} {r.mean_throughput_rps:>12.3f}"
        )
    print(sep)

    # Speedup of ours vs peft and vllm per scenario
    by_scenario: dict[str, dict[str, ScenarioResult]] = {}
    for r in results:
        by_scenario.setdefault(r.scenario, {})[r.backend] = r

    print(f"\n{'Speedup (ours vs baselines)':^100}")
    print(sep)
    print(f"{'Scenario':<28} {'ours TTFT vs peft':>20} {'ours TTFT vs vllm':>20} "
          f"{'ours tput vs peft':>20} {'ours tput vs vllm':>20}")
    print(sep)
    for scenario, by_be in by_scenario.items():
        ours = by_be.get("ours")
        peft = by_be.get("peft")
        vllm = by_be.get("vllm")
        if ours is None:
            continue
        def _ratio(base, candidate, higher_better=False):
            if base is None:
                return "n/a"
            v = (base / candidate) if not higher_better else (candidate / base)
            return f"{v:.2f}x"
        print(
            f"{scenario:<28} "
            f"{_ratio(peft.mean_ttft_s if peft else None, ours.mean_ttft_s):>20} "
            f"{_ratio(vllm.mean_ttft_s if vllm else None, ours.mean_ttft_s):>20} "
            f"{_ratio(ours.mean_throughput_rps, peft.mean_throughput_rps if peft else None, True):>20} "
            f"{_ratio(ours.mean_throughput_rps, vllm.mean_throughput_rps if vllm else None, True):>20}"
        )
    print(sep + "\n")


# ── Tests ──────────────────────────────────────────────────────────────────────

all_results: list[ScenarioResult] = []


@pytest.mark.real_world
def test_single_request(ours_client, peft_client, vllm_client, cli):
    """One request, one adapter — baseline latency."""
    adapter_name = next(iter(cli["adapters"]))
    prompt = next(p for n, p, _ in [
        ("diagnosis", "Patient presents with high fever. Possible diagnosis?", None),
        ("implementation", "Write a Python hello-world function.", None),
    ] if n == adapter_name or True)  # any first match

    pairs = [(prompt, adapter_name)]

    for backend_name, client in [
        ("ours", ours_client),
        ("peft", peft_client),
        ("vllm", vllm_client),
    ]:
        r = _bench_scenario(
            client, backend_name, "single_request",
            pairs, cli["max_tokens"], cli["num_runs"])
        all_results.append(r)
        print(f"\n[{backend_name}] single_request  "
              f"TTFT={r.mean_ttft_s:.3f}s  "
              f"latency={r.mean_total_s:.3f}s  "
              f"throughput={r.mean_throughput_rps:.3f} req/s")

    assert True  # metric collection test — never fails on values


@pytest.mark.real_world
def test_batch_same_adapter(ours_client, peft_client, vllm_client, cli):
    """Batch of N requests, all same adapter."""
    pairs = _pick_prompts(cli["adapters"], cli["batch_size"], mixed=False)

    for backend_name, client in [
        ("ours", ours_client),
        ("peft", peft_client),
        ("vllm", vllm_client),
    ]:
        r = _bench_scenario(
            client, backend_name, "batch_same_adapter",
            pairs, cli["max_tokens"], cli["num_runs"])
        all_results.append(r)
        print(f"\n[{backend_name}] batch_same_adapter n={cli['batch_size']}  "
              f"TTFT={r.mean_ttft_s:.3f}s  "
              f"latency={r.mean_total_s:.3f}s  "
              f"throughput={r.mean_throughput_rps:.3f} req/s")

    assert True


@pytest.mark.real_world
def test_batch_mixed_adapters(ours_client, peft_client, vllm_client, cli):
    """
    Batch of N requests with round-robin adapters.
    This is the key scenario where our batched LoRA should be fastest
    (one forward pass vs N sequential forwards in PEFT).
    """
    if len(cli["adapters"]) < 2:
        pytest.skip("need ≥2 adapters for mixed-adapter benchmark")

    pairs = _pick_prompts(cli["adapters"], cli["batch_size"], mixed=True)

    for backend_name, client in [
        ("ours", ours_client),
        ("peft", peft_client),
        ("vllm", vllm_client),
    ]:
        r = _bench_scenario(
            client, backend_name, "batch_mixed_adapters",
            pairs, cli["max_tokens"], cli["num_runs"])
        all_results.append(r)
        print(f"\n[{backend_name}] batch_mixed_adapters n={cli['batch_size']}  "
              f"TTFT={r.mean_ttft_s:.3f}s  "
              f"latency={r.mean_total_s:.3f}s  "
              f"throughput={r.mean_throughput_rps:.3f} req/s")

    assert True


@pytest.mark.real_world
def test_print_comparison_table(
        ours_client, peft_client, vllm_client, cli):  # noqa: ARG001
    """Final test: prints the full comparison table. Always runs last."""
    if not all_results:
        pytest.skip("No results collected — run other benchmark tests first")
    _print_table(all_results)
    assert True
