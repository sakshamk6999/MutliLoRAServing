import sys
import os
import argparse
import threading
from concurrent import futures
from dataclasses import dataclass, field

import torch
import grpc

# Ensure both the project root and model_logic/ are importable
_here = os.path.dirname(os.path.abspath(__file__))
_model_logic_dir = os.path.dirname(_here)
_project_root = os.path.dirname(_model_logic_dir)
for _d in (_project_root, _model_logic_dir):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from model_logic.protos import model_service_pb2, model_service_pb2_grpc
from model.qwen_lora_model import QwenLoRAModel
from transformers import AutoTokenizer


# ── Per-batch state kept between Prefill and Decode calls ──────────────────────

@dataclass
class BatchState:
    batch_id: str
    request_ids: list[str]
    adapter_ids: list[str]
    max_tokens: list[int]                   # max new tokens per request

    # KV cache location table [batch_size, max_prompt_len + max_new_tokens]
    b_loc: torch.Tensor
    b_seq_len: torch.Tensor                 # current total tokens per request (grows each step)
    b_start_loc: torch.Tensor              # cumulative start positions (prefill only; decode recomputed)
    max_len_in_batch: int

    # Accumulated generated token ids per request
    generated_ids: list[list[int]] = field(default_factory=list)
    tokens_generated: list[int] = field(default_factory=list)
    finished: list[bool] = field(default_factory=list)
    finish_reasons: list[str] = field(default_factory=list)  # "stop" | "length"

    # Next input token ids for the decode step
    next_token_ids: torch.Tensor = None     # [batch_size]


# ── Servicer ───────────────────────────────────────────────────────────────────

class ModelServiceServicer(model_service_pb2_grpc.ModelServiceServicer):

    def __init__(self, model: QwenLoRAModel, tokenizer):
        self._lock = threading.Lock()  # protects _batches across concurrent gRPC calls
        self.model = model
        self.tokenizer = tokenizer
        self._batches: dict[str, BatchState] = {}

    # ── Adapter management ─────────────────────────────────────────────────────

    def LoadAdapter(self, request, context):
        adapter_id = request.adapter_id
        try:
            if self.model.adapter_manager.is_loaded(adapter_id):
                return model_service_pb2.LoadAdapterResponse(
                    adapter_id=adapter_id, status="already_loaded")
            self.model.load_adapter(adapter_id, request.adapter_path)
            print(f"[gRPC] LoadAdapter: {adapter_id} from {request.adapter_path}")
            return model_service_pb2.LoadAdapterResponse(
                adapter_id=adapter_id, status="loaded")
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return model_service_pb2.LoadAdapterResponse(
                adapter_id=adapter_id, status="error")

    def UnloadAdapter(self, request, context):
        adapter_id = request.adapter_id
        try:
            self.model.unload_adapter(adapter_id)
            print(f"[gRPC] UnloadAdapter: {adapter_id}")
            return model_service_pb2.UnloadAdapterResponse(
                adapter_id=adapter_id, status="unloaded")
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return model_service_pb2.UnloadAdapterResponse(
                adapter_id=adapter_id, status="not_found")

    # ── Prefill ────────────────────────────────────────────────────────────────

    def Prefill(self, request, context):
        n = len(request.request_ids)
        if not (n == len(request.prompts) == len(request.max_tokens) == len(request.adapter_ids)):
            return model_service_pb2.PrefillResponse(
                batch_id=request.batch_id, status="error",
                message="request_ids, prompts, max_tokens, adapter_ids length mismatch")

        try:
            batch_id = request.batch_id
            prompts = list(request.prompts)
            adapter_ids = list(request.adapter_ids)
            max_tokens = list(request.max_tokens)

            # Tokenize all prompts (no padding — we pack them flat)
            encoded = [
                self.tokenizer.encode(p, add_special_tokens=True) for p in prompts
            ]
            seq_lens = [len(ids) for ids in encoded]
            total_token_num = sum(seq_lens)
            batch_size = n

            flat_ids = torch.tensor(
                [tok for ids in encoded for tok in ids],
                dtype=torch.long, device="cuda")

            b_seq_len = torch.tensor(seq_lens, dtype=torch.long, device="cuda")
            b_start_loc = torch.zeros(batch_size, dtype=torch.long, device="cuda")
            b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], dim=0)
            max_prompt_len = int(b_seq_len.max().item())

            # b_loc pre-allocated wide enough for prompts + max new tokens
            max_new = max(max_tokens)
            total_cols = max_prompt_len + max_new
            b_loc = torch.zeros(batch_size, total_cols, dtype=torch.long, device="cuda")

            # Run prefill — populates KV cache and returns logits [n, vocab]
            logits = self.model.forward(
                batch_size=batch_size,
                total_token_num=total_token_num,
                max_len_in_batch=max_prompt_len,
                input_ids=flat_ids,
                b_loc=b_loc,
                b_start_loc=b_start_loc,
                b_seq_len=b_seq_len,
                adapter_ids=adapter_ids,
                is_prefill=True,
            )

            # Greedy sample first new token for each request
            next_token_ids = logits.argmax(dim=-1)  # [batch_size]

            state = BatchState(
                batch_id=batch_id,
                request_ids=list(request.request_ids),
                adapter_ids=adapter_ids,
                max_tokens=max_tokens,
                b_loc=b_loc,
                b_seq_len=b_seq_len,
                b_start_loc=b_start_loc,
                max_len_in_batch=max_prompt_len,
                generated_ids=[[] for _ in range(batch_size)],
                tokens_generated=[0] * batch_size,
                finished=[False] * batch_size,
                finish_reasons=[""] * batch_size,
                next_token_ids=next_token_ids,
            )
            with self._lock:
                self._batches[batch_id] = state

            print(f"[gRPC] Prefill: batch_id={batch_id} n={n} prompt_lens={seq_lens}")
            return model_service_pb2.PrefillResponse(
                batch_id=batch_id, status="accepted", message="")

        except Exception as e:
            import traceback
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return model_service_pb2.PrefillResponse(
                batch_id=request.batch_id, status="error", message=str(e))

    # ── Decode ─────────────────────────────────────────────────────────────────

    def Decode(self, request, context):
        batch_id = request.batch_id
        with self._lock:
            if batch_id not in self._batches:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"batch_id {batch_id!r} not found")
                return model_service_pb2.DecodeResponse(batch_id=batch_id)
            state = self._batches[batch_id]

        try:
            batch_size = len(state.request_ids)
            eos_id = self.tokenizer.eos_token_id

            # Commit the previous step's sampled tokens into state
            for i in range(batch_size):
                if state.finished[i]:
                    continue
                tok = state.next_token_ids[i].item()
                state.generated_ids[i].append(tok)
                state.tokens_generated[i] += 1
                if tok == eos_id:
                    state.finished[i] = True
                    state.finish_reasons[i] = "stop"
                elif state.tokens_generated[i] >= state.max_tokens[i]:
                    state.finished[i] = True
                    state.finish_reasons[i] = "length"

            # If all requests are done, skip the forward pass
            all_done = all(state.finished)
            if not all_done:
                # Advance sequence lengths and max_len for unfinished requests
                new_seq_len = state.b_seq_len.clone()
                for i in range(batch_size):
                    if not state.finished[i]:
                        new_seq_len[i] += 1
                state.b_seq_len = new_seq_len
                state.max_len_in_batch = int(state.b_seq_len.max().item())

                # Decode input: one token per request (use the last sampled token)
                decode_input = state.next_token_ids.clone()  # [batch_size]

                # b_start_loc for decode is trivially arange (1 token per request)
                b_start_loc_dec = torch.arange(batch_size, dtype=torch.long, device="cuda")
                total_token_num = batch_size  # one new token per request

                logits = self.model.forward(
                    batch_size=batch_size,
                    total_token_num=total_token_num,
                    max_len_in_batch=state.max_len_in_batch,
                    input_ids=decode_input,
                    b_loc=state.b_loc,
                    b_start_loc=b_start_loc_dec,
                    b_seq_len=state.b_seq_len,
                    adapter_ids=state.adapter_ids,
                    is_prefill=False,
                )

                state.next_token_ids = logits.argmax(dim=-1)

            # Free KV cache if all done
            if all_done:
                self._free_batch(state)
                with self._lock:
                    self._batches.pop(batch_id, None)

            # Build response
            generated_texts = []
            is_finished = []
            finish_reasons = []
            for i in range(batch_size):
                text = self.tokenizer.decode(
                    state.generated_ids[i], skip_special_tokens=True)
                generated_texts.append(text)
                is_finished.append(state.finished[i])
                finish_reasons.append(state.finish_reasons[i])

            return model_service_pb2.DecodeResponse(
                batch_id=batch_id,
                request_ids=state.request_ids,
                generated_texts=generated_texts,
                is_finished=is_finished,
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return model_service_pb2.DecodeResponse(batch_id=batch_id)

    def _free_batch(self, state: BatchState):
        """Return KV cache slots for all requests in the batch."""
        mem = self.model.mem_manager
        for i in range(len(state.request_ids)):
            seq_len = state.b_seq_len[i].item()
            slots = state.b_loc[i, :seq_len]
            mem.free(slots)


# ── Server startup ─────────────────────────────────────────────────────────────

def build_servicer(
    weight_dir: str,
    max_total_token_num: int = 8192,
    mem_adapter_size: int = 0,
    adapter_dirs: dict[str, str] | None = None,
) -> ModelServiceServicer:
    print(f"[server] Loading model from {weight_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(weight_dir, trust_remote_code=True)
    model = QwenLoRAModel(
        weight_dir=weight_dir,
        max_total_token_num=max_total_token_num,
        mem_adapter_size=mem_adapter_size,
        adapter_dirs=adapter_dirs or {},
    )
    print("[server] Model ready.")
    return ModelServiceServicer(model, tokenizer)


def serve(
    weight_dir: str,
    port: int = 50051,
    max_total_token_num: int = 8192,
    mem_adapter_size: int = 0,
    adapter_dirs: dict[str, str] | None = None,
    max_workers: int = 4,
):
    servicer = build_servicer(weight_dir, max_total_token_num, mem_adapter_size, adapter_dirs)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    model_service_pb2_grpc.add_ModelServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"[gRPC server] listening on :{port}")
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3 multi-LoRA gRPC server")
    parser.add_argument("--weight_dir", required=True,
                        help="Path or HF repo ID for Qwen3 base weights")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--max_total_token_num", type=int, default=8192)
    parser.add_argument("--mem_adapter_size", type=int, default=0)
    parser.add_argument("--adapter", nargs=2, metavar=("ID", "PATH"), action="append",
                        default=[], help="--adapter task-A ./adapters/task-A (repeatable)")
    args = parser.parse_args()

    adapter_dirs = {aid: path for aid, path in args.adapter}
    serve(
        weight_dir=args.weight_dir,
        port=args.port,
        max_total_token_num=args.max_total_token_num,
        mem_adapter_size=args.mem_adapter_size,
        adapter_dirs=adapter_dirs,
    )
