from model_logic.protos import model_service_pb2


class StubBackend:
    """No real model; matches model_endpoint grpc_server stub behavior."""

    def load_adapter(
        self, request: model_service_pb2.LoadAdapterRequest
    ) -> model_service_pb2.LoadAdapterResponse:
        print(
            f"[gRPC] LoadAdapter: adapter_id={request.adapter_id} "
            f"path={request.adapter_path}"
        )
        return model_service_pb2.LoadAdapterResponse(
            adapter_id=request.adapter_id,
            status="loaded",
        )

    def unload_adapter(
        self, request: model_service_pb2.UnloadAdapterRequest
    ) -> model_service_pb2.UnloadAdapterResponse:
        print(f"[gRPC] UnloadAdapter: adapter_id={request.adapter_id}")
        return model_service_pb2.UnloadAdapterResponse(
            adapter_id=request.adapter_id,
            status="unloaded",
        )

    def prefill(
        self, request: model_service_pb2.PrefillRequest
    ) -> model_service_pb2.PrefillResponse:
        n = len(request.request_ids)
        if not (
            n == len(request.prompts)
            == len(request.max_tokens)
            == len(request.adapter_ids)
        ):
            return model_service_pb2.PrefillResponse(
                batch_id=request.batch_id,
                status="error",
                message=(
                    f"length mismatch: request_ids={n} "
                    f"prompts={len(request.prompts)} "
                    f"max_tokens={len(request.max_tokens)} "
                    f"adapter_ids={len(request.adapter_ids)}"
                ),
            )
        print(
            f"[gRPC] Prefill: batch_id={request.batch_id} n={n} "
            f"adapters={list(request.adapter_ids)}"
        )
        return model_service_pb2.PrefillResponse(
            batch_id=request.batch_id,
            status="accepted",
            message="stub",
        )

    def decode(
        self, request: model_service_pb2.DecodeRequest
    ) -> model_service_pb2.DecodeResponse:
        print(f"[gRPC] Decode: batch_id={request.batch_id}")
        return model_service_pb2.DecodeResponse(
            batch_id=request.batch_id,
            request_ids=[],
            generated_texts=[],
            is_finished=[],
        )
