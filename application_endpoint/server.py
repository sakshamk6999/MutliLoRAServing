import threading
import uuid
from contextlib import asynccontextmanager

import zmq
from fastapi import FastAPI, HTTPException
from data_model.data_types import ClientRequest, ModelResponse

ZMQ_PUSH_ADDR   = "tcp://*:5555"        # push requests to MLP classifier
ZMQ_RESULT_ADDR = "tcp://localhost:5557" # pull results from router


def _result_collector(ctx: zmq.Context, store: dict):
    """Background thread: pulls ModelResponse objects from the router result socket."""
    sock = ctx.socket(zmq.PULL)
    sock.connect(ZMQ_RESULT_ADDR)
    sock.setsockopt(zmq.RCVTIMEO, 500)  # 500 ms poll so thread exits cleanly on shutdown
    print(f"[Interface] result collector connected to {ZMQ_RESULT_ADDR}")
    while True:
        try:
            raw = sock.recv_json()
            resp = ModelResponse(**raw)
            store[resp.request_id] = resp
        except zmq.error.Again:
            continue  # timeout — loop and check again
        except zmq.error.ZMQError:
            break     # socket closed on shutdown


@asynccontextmanager
async def lifespan(app: FastAPI):
    ctx = zmq.Context()

    push_sock = ctx.socket(zmq.PUSH)
    push_sock.bind(ZMQ_PUSH_ADDR)
    app.state.zmq_socket = push_sock

    result_store: dict[str, ModelResponse] = {}
    app.state.result_store = result_store

    collector = threading.Thread(
        target=_result_collector,
        args=(ctx, result_store),
        daemon=True,
        name="result-collector",
    )
    collector.start()

    print(f"[Interface] ZMQ PUSH bound to {ZMQ_PUSH_ADDR}")
    yield

    push_sock.close()
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


@app.get("/result/{request_id}")
async def get_result(request_id: str):
    """Poll for the result of a previously queued request."""
    resp = app.state.result_store.get(request_id)
    if resp is None:
        raise HTTPException(status_code=202, detail="pending")
    # Remove from store once delivered
    del app.state.result_store[request_id]
    return resp.model_dump()


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("application_endpoint.server:app", host="0.0.0.0", port=8000, reload=False)
