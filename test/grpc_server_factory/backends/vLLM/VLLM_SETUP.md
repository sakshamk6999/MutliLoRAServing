# vLLM backend setup (`VLLMBackend`)

This document describes how to **install**, **configure**, and **run** the gRPC
`ModelService` backend implemented in `vLLM.py`, which wraps vLLM’s
`AsyncLLMEngine` and maps your router’s **Prefill / Decode** RPCs to vLLM’s
**async streaming** API.

## What this backend does (short)

- Your **router** drives inference with explicit **`Prefill`** then repeated
  **`Decode`** calls (`model_logic/protos/model_service.proto`).
- vLLM uses an internal **continuous batching** scheduler and **`generate` →
  async iterator** per request.
- The backend stores **one async generator per request** at Prefill and, on each
  Decode, advances each stream by **one step** (`anext`), filling the proto
  response with cumulative text and `is_finished` flags.

See the module docstring in `vLLM.py` for the full behavioral summary.

## Dependencies

- **Python** with this repo on `PYTHONPATH` (or installed as a package), plus
  **`grpcio`** for the gRPC server (same as other backends).
- **`vllm`** installed in the **same** environment you use to run the server.
  Install a build that matches your **CUDA / PyTorch** stack; follow vLLM’s
  official install docs for your platform.
- A **GPU** is typical for vLLM; CPU may be supported depending on vLLM version
  and model, but is not the primary target.

The backend imports vLLM **inside** `VLLMBackend.__init__`. If `import vllm`
fails, construction raises a clear `ImportError`.

## API compatibility note

`VLLMBackend` targets the classic **`AsyncLLMEngine`** +
**`AsyncEngineArgs`** API (as in vLLM **0.6.x**-style releases). Newer vLLM
versions may refactor names (for example `AsyncLLMEngine` aliasing other types).
If imports or `generate` signatures change, align your vLLM version or adjust
`vLLM.py` accordingly.

## Running the gRPC server (CLI)

From the repo root (or any layout where `test.grpc_server_factory` is
importable):

```bash
python -m test.grpc_server_factory \
  --kind vllm \
  --base-model-id <BASE_MODEL_ID_OR_PATH> \
  --port 50051
```

Or set the base model via environment variable:

```bash
export BASE_MODEL_ID=<BASE_MODEL_ID_OR_PATH>
python -m test.grpc_server_factory --kind vllm --port 50051
```

### Required configuration

| Input | Meaning |
| --- | --- |
| **`--base-model-id` or `BASE_MODEL_ID`** | Hugging Face model id or local path to the **base** model weights vLLM loads at engine startup. **Required** for `--kind vllm`. |

### Optional environment / flags

| Variable / flag | Meaning |
| --- | --- |
| **`GRPC_BIND`** | Full listen address if you do not pass `--bind` (e.g. `[::]:50051`). |
| **`--bind`** | Overrides host:port (overrides `--port` when set). |
| **`--port`** | Used as `[::]:PORT` when `--bind` is omitted (default **50051**). |

The server uses an **insecure** gRPC port (`add_insecure_port`). Use a sidecar
or TLS-terminating proxy if you need encryption on the wire.

## Programmatic construction (`VLLMBackend`)

If you construct `VLLMBackend` yourself (instead of only via `factory.py`), you
can tune engine and runtime behavior.

### Constructor parameters (high level)

| Parameter | Default | Role |
| --- | --- | --- |
| **`base_model_id`** | (required) | Passed to **`AsyncEngineArgs(model=...)`** — base checkpoint. |
| **`max_loras`** | `32` | vLLM **`max_loras`**: max number of LoRA adapters the **engine** is sized for (not set per RPC by adapter id strings). |
| **`max_lora_rank`** | `256` | vLLM **`max_lora_rank`**: max LoRA **rank** the engine supports; your saved adapters must use rank **≤** this. |
| **`request_timeout_sec`** | `600.0` | Timeout when blocking on async work scheduled on the engine loop (`run_coroutine_threadsafe(...).result(...)`). |
| **`extra_engine_args`** | `None` | `dict` merged into **`AsyncEngineArgs`** after the defaults (e.g. tensor parallel size, dtype overrides—**names must match** vLLM’s `AsyncEngineArgs` for your version). |
| **`metrics`** | `None` | Optional shared **`ServingMetrics`** instance (`test/grpc_server_factory/backends/metrics.py`); if omitted, a new one is used. |

**`max_loras` / `max_lora_rank`** are **fixed at engine creation**. They are
**not** derived from router **`adapter_ids`**. Router ids select **which**
registered adapter path to attach per row; these two fields define **capacity**
and **maximum rank** for the LoRA runtime.

Example:

```python
from test.grpc_server_factory.backends.vLLM import VLLMBackend

backend = VLLMBackend(
    base_model_id="meta-llama/Llama-2-7b-hf",
    max_loras=16,
    max_lora_rank=128,
    extra_engine_args={"tensor_parallel_size": 1},
)
```

## Operational flow (router ↔ backend)

1. **`LoadAdapter`**: Records **`adapter_id` → `adapter_path`** in memory only.
   The backend does **not** force-load weights here; vLLM resolves LoRA weights
   when requests run (paths must be valid for vLLM’s LoRA loader).
2. **`UnloadAdapter`**: Removes that id from the registry.
3. **`Prefill`**: For each row, builds a vLLM **`LoRARequest`** from the row’s
   **`adapter_id`** and registered path, calls **`engine.generate(...)`** on the
   dedicated asyncio loop, and stores the resulting **async generator** under
   **`batch_id` → `request_id`**.
4. **`Decode`**: For each active generator in the batch, runs **`anext`** once,
   maps **`RequestOutput`** to **`generated_texts`** / **`is_finished`**, removes
   finished streams, and records **metrics** when a request completes.

Parallel fields in **`PrefillRequest`** must have equal length:
`request_ids`, `prompts`, `max_tokens`, `adapter_ids`.

## Metrics

Completion metrics (throughput, mean latency, TTFT) use the shared
**`ServingMetrics`** helper in `../metrics.py`, same pattern as the PEFT causal
backend. Logs go to **logging** and **stdout** when requests finish.

## Architecture note (threading / asyncio)

`AsyncLLMEngine` runs on a **dedicated thread** with a **single asyncio event
loop** (`loop.run_forever()`). Synchronous gRPC handlers schedule coroutines
onto that loop with **`asyncio.run_coroutine_threadsafe`** so streams stay tied
to the engine’s loop.

## Related files

- Implementation: `vLLM.py`
- Proto: `model_logic/protos/model_service.proto`
- Shared metrics: `../metrics.py`
- Server factory: `../../factory.py`
