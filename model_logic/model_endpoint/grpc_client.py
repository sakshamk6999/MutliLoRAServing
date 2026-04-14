import grpc
from model_logic.protos import model_service_pb2, model_service_pb2_grpc


class ModelServiceClient:
    def __init__(self, target: str = "localhost:50051"):
        self.channel = grpc.insecure_channel(target)
        self.stub = model_service_pb2_grpc.ModelServiceStub(self.channel)

    def load_adapter(self, adapter_id: str, adapter_path: str, rank: int = 8, alpha: float = 16.0):
        req = model_service_pb2.LoadAdapterRequest(
            adapter_id=adapter_id,
            adapter_path=adapter_path,
            rank=rank,
            alpha=alpha,
        )
        return self.stub.LoadAdapter(req)

    def unload_adapter(self, adapter_id: str):
        req = model_service_pb2.UnloadAdapterRequest(adapter_id=adapter_id)
        return self.stub.UnloadAdapter(req)

    def prefill(self, batch_id: str, request_ids: list, prompts: list, max_tokens: list):
        req = model_service_pb2.PrefillRequest(
            batch_id=batch_id,
            request_ids=request_ids,
            prompts=prompts,
            max_tokens=max_tokens,
        )
        return self.stub.Prefill(req)

    def decode(self, batch_id: str):
        req = model_service_pb2.DecodeRequest(batch_id=batch_id)
        return self.stub.Decode(req)

    def close(self):
        self.channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
