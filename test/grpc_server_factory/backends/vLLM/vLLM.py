"""
vLLM `AsyncLLMEngine` adapter for ModelService gRPC.

Maps step-wise ``Prefill`` / ``Decode`` RPCs to vLLM's continuous async scheduler by
keeping one async generator per request and advancing each with ``anext`` on every
``Decode`` call.

Requires a compatible vLLM install (e.g. 0.6.x) with ``AsyncLLMEngine`` and
``AsyncEngineArgs`` as documented in that release.

Metrics: throughput, mean latency, first-token latency (see ``metrics``).

Setup, dependencies, and how to run: ``VLLM_SETUP.md`` (this directory).

This is a test change to the file.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncGenerator, Coroutine
from dataclasses import dataclass
from typing import Any, Dict, Optional

from model_logic.protos import model_service_pb2

from ..metrics import ServingMetrics

DEFAULT_BASE_MODEL = "{{BASE_MODEL_ID_OR_PATH}}"


@dataclass
class _ReqMetrics:
    t_prefill_start: float
    t_prefill_end: float
    t_first_gen_token: Optional[float]
    metrics_emitted: bool


class VLLMBackend:
    """Registry of adapter paths + per-batch async generators over ``AsyncLLMEngine``."""

    def __init__(
        self,
        base_model_id: str = DEFAULT_BASE_MODEL,
        *,
        max_loras: int = 32,
        max_lora_rank: int = 256,
        request_timeout_sec: float = 600.0,
        extra_engine_args: Optional[Dict[str, Any]] = None,
        metrics: Optional[ServingMetrics] = None,
    ) -> None:
        if base_model_id == DEFAULT_BASE_MODEL:
            raise ValueError(
                "Set base_model_id to a real model id or path (placeholder "
                f"{DEFAULT_BASE_MODEL!r} is not loadable)."
            )

        try:
            from vllm import AsyncEngineArgs, SamplingParams
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            from vllm.lora.request import LoRARequest
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "VLLMBackend requires the vllm package. Install vllm in your "
                "environment to use this backend."
            ) from e

        self._SamplingParams = SamplingParams
        self._LoRARequest = LoRARequest

        engine_kw: Dict[str, Any] = {
            "model": base_model_id,
            "enable_lora": True,
            "max_loras": max_loras,
            "max_lora_rank": max_lora_rank,
        }
        if extra_engine_args:
            engine_kw.update(extra_engine_args)

        self._engine_args = AsyncEngineArgs(**engine_kw)
        self._request_timeout_sec = request_timeout_sec

        self._lock = threading.Lock()
        self.adapter_paths: Dict[str, str] = {}
        self._lora_int_ids: Dict[str, int] = {}
        self._next_lora_int: int = 1

        self.active_batches: Dict[str, Dict[str, AsyncGenerator[Any, None]]] = {}
        self._request_metrics: Dict[str, Dict[str, _ReqMetrics]] = {}
        self._metrics = metrics if metrics is not None else ServingMetrics()

        self._loop_ready = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._engine: Optional[Any] = None
        self._init_error: Optional[BaseException] = None

        def _engine_thread_main() -> None:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                self._engine = AsyncLLMEngine.from_engine_args(self._engine_args)
                self._loop_ready.set()
                loop.run_forever()
            except BaseException as exc:  # pragma: no cover - startup failures
                self._init_error = exc
                self._loop_ready.set()

        self._thread = threading.Thread(
            target=_engine_thread_main,
            name="vllm-async-loop",
            daemon=True,
        )
        self._thread.start()
        if not self._loop_ready.wait(timeout=600.0):
            raise RuntimeError("Timed out waiting for AsyncLLMEngine event loop.")
        if self._init_error is not None:
            raise RuntimeError(
                f"AsyncLLMEngine failed to start: {self._init_error!r}"
            ) from self._init_error
        assert self._loop is not None and self._engine is not None

    def _emit_metrics(self, st: _ReqMetrics) -> None:
        if st.metrics_emitted:
            return
        st.metrics_emitted = True
        now = time.monotonic()
        total = now - st.t_prefill_start
        if st.t_first_gen_token is not None:
            ttft = st.t_first_gen_token - st.t_prefill_start
        else:
            ttft = st.t_prefill_end - st.t_prefill_start
        self._metrics.record_completion(
            latency_sec=total,
            first_token_latency_sec=ttft,
        )

    def _lora_int_id_for(self, adapter_id: str) -> int:
        with self._lock:
            if adapter_id not in self._lora_int_ids:
                self._lora_int_ids[adapter_id] = self._next_lora_int
                self._next_lora_int += 1
            return self._lora_int_ids[adapter_id]

    def _run_coro(self, coro: Coroutine[Any, Any, Any]) -> Any:
        assert self._loop is not None
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=self._request_timeout_sec)

    def _make_generate_generator(
        self,
        prompt: str,
        sampling_params: Any,
        request_id: str,
        lora_request: Any | None,
    ) -> AsyncGenerator[Any, None]:
        engine = self._engine

        async def _create():
            return engine.generate(
                prompt,
                sampling_params,
                request_id,
                lora_request=lora_request,
            )

        return self._run_coro(_create())

    @staticmethod
    async def _anext_request_output(
        gen: AsyncGenerator[Any, None],
    ) -> Any | None:
        try:
            return await gen.__anext__()
        except StopAsyncIteration:
            return None

    def _advance_generator(self, gen: AsyncGenerator[Any, None]) -> Any | None:
        return self._run_coro(self._anext_request_output(gen))

    def load_adapter(
        self, request: model_service_pb2.LoadAdapterRequest
    ) -> model_service_pb2.LoadAdapterResponse:
        aid, path = request.adapter_id, request.adapter_path
        with self._lock:
            if aid in self.adapter_paths:
                if self.adapter_paths[aid] == path:
                    return model_service_pb2.LoadAdapterResponse(
                        adapter_id=aid, status="already_loaded"
                    )
            self.adapter_paths[aid] = path
        return model_service_pb2.LoadAdapterResponse(adapter_id=aid, status="loaded")

    def unload_adapter(
        self, request: model_service_pb2.UnloadAdapterRequest
    ) -> model_service_pb2.UnloadAdapterResponse:
        aid = request.adapter_id
        with self._lock:
            if aid not in self.adapter_paths:
                return model_service_pb2.UnloadAdapterResponse(
                    adapter_id=aid, status="not_found"
                )
            del self.adapter_paths[aid]
        return model_service_pb2.UnloadAdapterResponse(
            adapter_id=aid, status="unloaded"
        )

    def prefill(
        self, request: model_service_pb2.PrefillRequest
    ) -> model_service_pb2.PrefillResponse:
        bid = request.batch_id
        ids = list(request.request_ids)
        prompts = list(request.prompts)
        mts = list(request.max_tokens)
        adapters = list(request.adapter_ids)

        n = len(ids)
        if not (n == len(prompts) == len(mts) == len(adapters)):
            return model_service_pb2.PrefillResponse(
                batch_id=bid,
                status="error",
                message=(
                    f"length mismatch: request_ids={len(ids)} "
                    f"prompts={len(prompts)} max_tokens={len(mts)} "
                    f"adapter_ids={len(adapters)}"
                ),
            )

        with self._lock:
            if bid in self.active_batches:
                return model_service_pb2.PrefillResponse(
                    batch_id=bid,
                    status="error",
                    message=f"batch_id {bid!r} already exists",
                )
            missing = [a for a in set(adapters) if a not in self.adapter_paths]
            if missing:
                return model_service_pb2.PrefillResponse(
                    batch_id=bid,
                    status="error",
                    message=f"adapters not registered: {missing}",
                )
            paths_snapshot = {a: self.adapter_paths[a] for a in adapters}

        LoRARequest = self._LoRARequest
        SamplingParams = self._SamplingParams
        batch_gens: Dict[str, AsyncGenerator[Any, None]] = {}
        batch_meta: Dict[str, _ReqMetrics] = {}

        try:
            for i in range(n):
                rid = ids[i]
                prompt = prompts[i]
                adapter_id = adapters[i]
                path = paths_snapshot[adapter_id]
                t_ps = time.monotonic()
                lora_int = self._lora_int_id_for(adapter_id)
                lora_req = LoRARequest(
                    lora_name=adapter_id,
                    lora_int_id=lora_int,
                    lora_path=path,
                )
                sp = SamplingParams(max_tokens=int(mts[i]))
                gen = self._make_generate_generator(prompt, sp, rid, lora_req)
                t_pe = time.monotonic()
                batch_gens[rid] = gen
                batch_meta[rid] = _ReqMetrics(
                    t_prefill_start=t_ps,
                    t_prefill_end=t_pe,
                    t_first_gen_token=None,
                    metrics_emitted=False,
                )
        except Exception as e:
            return model_service_pb2.PrefillResponse(
                batch_id=bid,
                status="error",
                message=f"prefill failed: {e!s}",
            )

        with self._lock:
            if bid in self.active_batches:
                return model_service_pb2.PrefillResponse(
                    batch_id=bid,
                    status="error",
                    message=f"batch_id {bid!r} already exists",
                )
            self.active_batches[bid] = batch_gens
            self._request_metrics[bid] = batch_meta

        return model_service_pb2.PrefillResponse(
            batch_id=bid, status="accepted", message="ok"
        )

    def decode(
        self, request: model_service_pb2.DecodeRequest
    ) -> model_service_pb2.DecodeResponse:
        bid = request.batch_id

        with self._lock:
            batch = self.active_batches.get(bid)

        if not batch:
            return model_service_pb2.DecodeResponse(
                batch_id=bid,
                request_ids=[],
                generated_texts=[],
                is_finished=[],
            )

        outs_ids: list[str] = []
        outs_text: list[str] = []
        outs_fin: list[bool] = []
        finished_ids: list[str] = []
        meta_by_rid: Dict[str, _ReqMetrics] | None = None
        with self._lock:
            meta_by_rid = self._request_metrics.get(bid)

        for rid, gen in list(batch.items()):
            mst = meta_by_rid.get(rid) if meta_by_rid else None
            ro = self._advance_generator(gen)
            if ro is None:
                outs_ids.append(rid)
                outs_text.append("")
                outs_fin.append(True)
                finished_ids.append(rid)
                if mst is not None:
                    self._emit_metrics(mst)
                continue

            if mst is not None and mst.t_first_gen_token is None:
                mst.t_first_gen_token = time.monotonic()

            text = ""
            if ro.outputs:
                text = ro.outputs[0].text or ""
            finished = bool(getattr(ro, "finished", False))
            outs_ids.append(rid)
            outs_text.append(text)
            outs_fin.append(finished)
            if finished:
                finished_ids.append(rid)
                if mst is not None:
                    self._emit_metrics(mst)

        with self._lock:
            b = self.active_batches.get(bid)
            if b:
                for rid in finished_ids:
                    b.pop(rid, None)
                if not b:
                    del self.active_batches[bid]
            rm = self._request_metrics.get(bid)
            if rm:
                for rid in finished_ids:
                    rm.pop(rid, None)
                if not rm:
                    del self._request_metrics[bid]

        return model_service_pb2.DecodeResponse(
            batch_id=bid,
            request_ids=outs_ids,
            generated_texts=outs_text,
            is_finished=outs_fin,
        )
