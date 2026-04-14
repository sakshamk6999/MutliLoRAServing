from contextlib import asynccontextmanager
import uuid
import zmq
from fastapi import FastAPI
from data_model.data_types import ClientRequest

ZMQ_PUSH_ADDR = "tcp://*:5555"


@asynccontextmanager
async def lifespan(app: FastAPI):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.bind(ZMQ_PUSH_ADDR)
    app.state.zmq_socket = sock
    print(f"[Interface] ZMQ PUSH bound to {ZMQ_PUSH_ADDR}")
    yield
    sock.close()
    ctx.term()


app = FastAPI(lifespan=lifespan)


@app.post("/generate")
async def generate(request: ClientRequest):
    rid = str(uuid.uuid4())
    app.state.zmq_socket.send_json({
        "request_id": rid,
        "prompt": request.prompt,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    })
    print(f"[Interface] queued {rid}")
    return {"request_id": rid, "status": "queued"}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("application_endpoint.server:app", host="0.0.0.0", port=8000, reload=False)
