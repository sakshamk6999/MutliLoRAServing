from typing import Protocol

from model_logic.protos import model_service_pb2


class ModelBackend(Protocol):
    """Generation/adapters implementation behind the gRPC surface."""

    def load_adapter(
        self, request: model_service_pb2.LoadAdapterRequest
    ) -> model_service_pb2.LoadAdapterResponse:
        ...

    def unload_adapter(
        self, request: model_service_pb2.UnloadAdapterRequest
    ) -> model_service_pb2.UnloadAdapterResponse:
        ...

    def prefill(
        self, request: model_service_pb2.PrefillRequest
    ) -> model_service_pb2.PrefillResponse:
        ...

    def decode(
        self, request: model_service_pb2.DecodeRequest
    ) -> model_service_pb2.DecodeResponse:
        ...
