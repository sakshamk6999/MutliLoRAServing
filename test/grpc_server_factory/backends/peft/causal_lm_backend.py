"""
Causal LM + PEFT LoRA backend for ModelService gRPC.

Decode: one forward per request per step (multi-adapter safe).

Prefill: group by adapter_id; one forward per request; set_adapter per group.

Metrics: throughput, mean latency, first-token latency (see ``metrics``).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_logic.protos import model_service_pb2

from .metrics import ServingMetrics

DEFAULT_BASE_MODEL = "{{BASE_MODEL_ID_OR_PATH}}"


@dataclass
class _ReqState:
    request_id: str
    adapter_id: str
    max_new_tokens: int
    new_tokens: int
    ids: List[int]
    past_key_values: Any
    finished: bool
    next_feed_token: int
    t_prefill_start: float
    t_prefill_end: float
    t_first_gen_token: Optional[float]
    metrics_emitted: bool


@dataclass
class _BatchState:
    requests: List[_ReqState]


class CausalLMPEFTBackend:
    """One base causal LM + PEFT LoRA registry; Prefill/Decode RPCs."""

    def __init__(
        self,
        base_model_id: str = DEFAULT_BASE_MODEL,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        metrics: Optional[ServingMetrics] = None,
    ) -> None:
        if base_model_id == DEFAULT_BASE_MODEL:
            raise ValueError(
                "Set base_model_id to a real model id or path (placeholder "
                f"{DEFAULT_BASE_MODEL!r} is not loadable)."
            )
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if torch_dtype is None:
            torch_dtype = (
                torch.float16 if self.device == "cuda" else torch.float32
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_id, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch_dtype,
            device_map=None,
        )
        self.model.to(self.device)
        self.model.eval()

        self._torch_dtype = torch_dtype
        self._peft_wrapped = False
        self._registry: Dict[str, str] = {}
        self._batches: Dict[str, _BatchState] = {}
        self._metrics = metrics if metrics is not None else ServingMetrics()

    def load_adapter(
        self, request: model_service_pb2.LoadAdapterRequest
    ) -> model_service_pb2.LoadAdapterResponse:
        aid, path = request.adapter_id, request.adapter_path
        if aid in self._registry:
            if self._registry[aid] == path:
                return model_service_pb2.LoadAdapterResponse(
                    adapter_id=aid, status="already_loaded"
                )
        try:
            if not self._peft_wrapped:
                self.model = PeftModel.from_pretrained(
                    self.model,
                    path,
                    adapter_name=aid,
                )
                self._peft_wrapped = True
            else:
                self.model.load_adapter(path, adapter_name=aid)
            self._registry[aid] = path
            self.model.eval()
            return model_service_pb2.LoadAdapterResponse(
                adapter_id=aid, status="loaded"
            )
        except Exception as exc:
            print(f"[CausalLMPEFT] LoadAdapter failed: {exc!r}")
            return model_service_pb2.LoadAdapterResponse(
                adapter_id=aid, status="error"
            )

    def unload_adapter(
        self, request: model_service_pb2.UnloadAdapterRequest
    ) -> model_service_pb2.UnloadAdapterResponse:
        aid = request.adapter_id
        if aid not in self._registry:
            return model_service_pb2.UnloadAdapterResponse(
                adapter_id=aid, status="not_found"
            )
        try:
            if hasattr(self.model, "delete_adapter"):
                self.model.delete_adapter(aid)
            del self._registry[aid]
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            return model_service_pb2.UnloadAdapterResponse(
                adapter_id=aid, status="unloaded"
            )
        except Exception:
            return model_service_pb2.UnloadAdapterResponse(
                adapter_id=aid, status="error"
            )

    def _emit_metrics(self, st: _ReqState) -> None:
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
        if bid in self._batches:
            return model_service_pb2.PrefillResponse(
                batch_id=bid,
                status="error",
                message=f"batch_id {bid!r} already exists",
            )

        missing = [a for a in set(adapters) if a not in self._registry]
        if missing:
            return model_service_pb2.PrefillResponse(
                batch_id=bid,
                status="error",
                message=f"adapters not loaded: {missing}",
            )

        groups: Dict[str, List[int]] = defaultdict(list)
        for i, a in enumerate(adapters):
            groups[a].append(i)

        states: List[Optional[_ReqState]] = [None] * n

        try:
            with torch.no_grad():
                for adapter_id, idxs in groups.items():
                    self.model.set_adapter(adapter_id)
                    for i in idxs:
                        t_ps = time.monotonic()
                        text = prompts[i]
                        enc = self.tokenizer(text, return_tensors="pt")
                        enc = {k: v.to(self.device) for k, v in enc.items()}
                        out = self.model(**enc, use_cache=True)
                        pkv = out.past_key_values
                        row_ids = enc["input_ids"][0].tolist()
                        first_next = int(out.logits[0, -1, :].argmax().item())
                        mt = int(mts[i])
                        done_before_decode = mt <= 0
                        t_pe = time.monotonic()
                        states[i] = _ReqState(
                            request_id=ids[i],
                            adapter_id=adapter_id,
                            max_new_tokens=mt,
                            new_tokens=0,
                            ids=row_ids,
                            past_key_values=pkv,
                            finished=done_before_decode,
                            next_feed_token=(
                                0 if done_before_decode else first_next
                            ),
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

        self._batches[bid] = _BatchState(
            requests=[s for s in states if s is not None]
        )
        return model_service_pb2.PrefillResponse(
            batch_id=bid, status="accepted", message="ok"
        )

    def decode(
        self, request: model_service_pb2.DecodeRequest
    ) -> model_service_pb2.DecodeResponse:
        bid = request.batch_id
        if bid not in self._batches:
            return model_service_pb2.DecodeResponse(
                batch_id=bid,
                request_ids=[],
                generated_texts=[],
                is_finished=[],
            )

        bs = self._batches[bid]
        outs_ids: List[str] = []
        outs_text: List[str] = []
        outs_fin: List[bool] = []

        eos = self.tokenizer.eos_token_id

        with torch.no_grad():
            for st in bs.requests:
                if st.finished:
                    self._emit_metrics(st)
                    outs_ids.append(st.request_id)
                    outs_text.append(
                        self.tokenizer.decode(st.ids, skip_special_tokens=True)
                    )
                    outs_fin.append(True)
                    continue

                self.model.set_adapter(st.adapter_id)

                tok = st.next_feed_token
                inp = torch.tensor(
                    [[tok]], device=self.device, dtype=torch.long
                )
                out = self.model(
                    input_ids=inp,
                    past_key_values=st.past_key_values,
                    use_cache=True,
                )
                st.past_key_values = out.past_key_values
                st.ids.append(tok)
                st.new_tokens += 1
                if st.t_first_gen_token is None:
                    st.t_first_gen_token = time.monotonic()
                st.next_feed_token = int(out.logits[0, -1, :].argmax().item())

                done = False
                if eos is not None and tok == eos:
                    done = True
                if st.new_tokens >= st.max_new_tokens:
                    done = True
                st.finished = done

                outs_ids.append(st.request_id)
                outs_text.append(
                    self.tokenizer.decode(st.ids, skip_special_tokens=True)
                )
                outs_fin.append(st.finished)
                if st.finished:
                    self._emit_metrics(st)

        if all(r.finished for r in bs.requests):
            del self._batches[bid]

        return model_service_pb2.DecodeResponse(
            batch_id=bid,
            request_ids=outs_ids,
            generated_texts=outs_text,
            is_finished=outs_fin,
        )
