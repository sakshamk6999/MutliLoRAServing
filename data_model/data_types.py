from pydantic import BaseModel, ConfigDict
from typing import List


class ClientRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    prompt: str
    max_tokens: int = 256
    temperature: float = 1.0


class TaggedRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    request_id: str
    prompt: str
    max_tokens: int
    temperature: float
    task_type: str  # e.g. "diagnosis"


class BatchRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    batch_id: str
    requests: List[TaggedRequest]


class ModelResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    request_id: str
    generated_text: str
    finish_reason: str  # "stop" | "length"
