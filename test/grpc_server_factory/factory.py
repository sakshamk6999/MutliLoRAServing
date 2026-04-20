from __future__ import annotations

from concurrent import futures
from enum import Enum
import grpc
import os

from model_logic.protos import model_service_pb2_grpc

from .backends.stub import StubBackend
from .servicer import DelegatingModelServicer


class BackendKind(str, Enum):
    STUB = "stub"
    CAUSAL_LM_PEFT = "causal_lm_peft"
    VLLM = "vllm"


def create_model_servicer(
    kind: BackendKind | str = BackendKind.STUB,
    *,
    base_model_id: str | None = None,
) -> DelegatingModelServicer:
    if isinstance(kind, str):
        kind = BackendKind(kind)
    if kind == BackendKind.STUB:
        backend = StubBackend()
    elif kind == BackendKind.CAUSAL_LM_PEFT:
        from .backends.peft import CausalLMPEFTBackend

        mid = base_model_id or os.environ.get("BASE_MODEL_ID")
        if not mid:
            raise ValueError(
                "CAUSAL_LM_PEFT requires base_model_id=... or BASE_MODEL_ID env"
            )
        backend = CausalLMPEFTBackend(base_model_id=mid)
    elif kind == BackendKind.VLLM:
        from .backends.vLLM import VLLMBackend

        mid = base_model_id or os.environ.get("BASE_MODEL_ID")
        if not mid:
            raise ValueError("vllm requires base_model_id=... or BASE_MODEL_ID env")
        backend = VLLMBackend(base_model_id=mid)
    else:
        raise ValueError(f"Unknown backend: {kind}")
    return DelegatingModelServicer(backend)


def serve(
    kind: BackendKind | str = BackendKind.STUB,
    *,
    bind: str | None = None,
    port: int = 50051,
    base_model_id: str | None = None,
) -> None:
    """Start gRPC ModelService. Use ``bind`` for a full address, or ``port`` for ``[::]:port``."""
    addr = bind or os.environ.get("GRPC_BIND") or f"[::]:{port}"
    servicer = create_model_servicer(kind, base_model_id=base_model_id)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    model_service_pb2_grpc.add_ModelServiceServicer_to_server(servicer, server)
    server.add_insecure_port(addr)
    server.start()
    print(f"[gRPC] backend={kind!s} bind={addr}")
    server.wait_for_termination()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="ModelService gRPC server (single entry for all backends).",
    )
    ap.add_argument(
        "--kind",
        default=BackendKind.STUB.value,
        choices=[k.value for k in BackendKind],
        help="Backend implementation",
    )
    ap.add_argument(
        "--bind",
        default=None,
        help="Full listen address, e.g. [::]:50051 or 0.0.0.0:50051 (overrides --port)",
    )
    ap.add_argument(
        "--port",
        type=int,
        default=50051,
        help="Used as [::]:PORT when --bind is omitted (default 50051)",
    )
    ap.add_argument(
        "--base-model-id",
        default=None,
        help="Required for causal_lm_peft; or set BASE_MODEL_ID",
    )
    ns = ap.parse_args()
    serve(
        kind=ns.kind,
        bind=ns.bind,
        port=ns.port,
        base_model_id=ns.base_model_id,
    )


if __name__ == "__main__":
    main()
