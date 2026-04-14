from concurrent import futures
import grpc
from model_logic.protos import model_service_pb2, model_service_pb2_grpc


class ModelServiceServicer(model_service_pb2_grpc.ModelServiceServicer):
    def LoadAdapter(self, request, context):
        print(f"[gRPC] LoadAdapter: adapter_id={request.adapter_id} path={request.adapter_path}")
        return model_service_pb2.LoadAdapterResponse(
            adapter_id=request.adapter_id,
            status="loaded",
        )

    def UnloadAdapter(self, request, context):
        print(f"[gRPC] UnloadAdapter: adapter_id={request.adapter_id}")
        return model_service_pb2.UnloadAdapterResponse(
            adapter_id=request.adapter_id,
            status="unloaded",
        )

    def Prefill(self, request, context):
        print(f"[gRPC] Prefill: batch_id={request.batch_id} n={len(request.prompts)}")
        return model_service_pb2.PrefillResponse(
            batch_id=request.batch_id,
            status="accepted",
            message="stub",
        )

    def Decode(self, request, context):
        print(f"[gRPC] Decode: batch_id={request.batch_id}")
        return model_service_pb2.DecodeResponse(
            batch_id=request.batch_id,
            request_ids=[],
            generated_texts=[],
            is_finished=[],
        )


def serve(port: int = 50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    model_service_pb2_grpc.add_ModelServiceServicer_to_server(ModelServiceServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"[gRPC server] listening on :{port}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
