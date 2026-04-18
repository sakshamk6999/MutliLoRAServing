from model_logic.protos import model_service_pb2, model_service_pb2_grpc

from .protocol import ModelBackend


class DelegatingModelServicer(model_service_pb2_grpc.ModelServiceServicer):
    """Routes RPCs to a `ModelBackend` instance."""

    def __init__(self, backend: ModelBackend) -> None:
        self._backend = backend

    def LoadAdapter(self, request, context):
        return self._backend.load_adapter(request)

    def UnloadAdapter(self, request, context):
        return self._backend.unload_adapter(request)

    def Prefill(self, request, context):
        return self._backend.prefill(request)

    def Decode(self, request, context):
        return self._backend.decode(request)
