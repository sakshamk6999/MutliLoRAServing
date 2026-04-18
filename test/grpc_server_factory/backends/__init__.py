# Import `backends.peft` only when using CAUSAL_LM_PEFT (torch/peft deps).
from .stub import StubBackend

__all__ = ["StubBackend"]
