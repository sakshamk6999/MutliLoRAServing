import os
import json
import torch
from safetensors.torch import load_file as safetensors_load_file

_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


class LoRAAdapterManager:
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, num_layers: int, dtype=torch.float16):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.dtype = dtype

        # adapter_id -> {layer_idx -> {module -> {"A": Tensor, "B": Tensor, "scaling": float}}}
        self._adapters: dict[str, dict] = {}
        self._id_to_int: dict[str, int] = {}
        self._int_to_id: list[str] = []

    # ------------------------------------------------------------------ #

    def load_adapter(self, adapter_id: str, adapter_path: str):
        if adapter_id in self._adapters:
            return  # already loaded

        config_path = os.path.join(adapter_path, "adapter_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        rank = int(config.get("r", config.get("rank", 16)))
        alpha = float(config.get("lora_alpha", 32.0))
        scaling = alpha / rank

        weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
        if not os.path.exists(weights_path):
            weights_path = os.path.join(adapter_path, "adapter_model.bin")
            raw = torch.load(weights_path, map_location="cpu")
        else:
            raw = safetensors_load_file(weights_path, device="cpu")

        adapter_weights: dict[int, dict] = {i: {} for i in range(self.num_layers)}

        for key, tensor in raw.items():
            # Key format: base_model.model.model.layers.{i}.self_attn.{module}.lora_{A|B}.weight
            parts = key.split(".")
            try:
                layer_idx = int(parts[4])
                module = parts[6]      # e.g. "q_proj"
                lora_type = parts[7]   # "lora_A" or "lora_B"
            except (IndexError, ValueError):
                continue

            if module not in _TARGET_MODULES:
                continue
            if lora_type not in ("lora_A", "lora_B"):
                continue

            t = tensor.to(dtype=self.dtype, device="cuda")
            layer_dict = adapter_weights[layer_idx]
            if module not in layer_dict:
                layer_dict[module] = {"A": None, "B": None, "scaling": scaling}
            key_ab = lora_type[-1]  # "A" or "B"
            layer_dict[module][key_ab] = t

        self._adapters[adapter_id] = adapter_weights
        # Register integer ID
        if adapter_id not in self._id_to_int:
            idx = len(self._int_to_id)
            self._id_to_int[adapter_id] = idx
            self._int_to_id.append(adapter_id)

    def unload_adapter(self, adapter_id: str):
        if adapter_id in self._adapters:
            del self._adapters[adapter_id]
            torch.cuda.empty_cache()

    def get_lora_weights(self, adapter_id: str, layer_idx: int,
                         module: str) -> tuple[torch.Tensor, torch.Tensor, float]:
        layer_dict = self._adapters[adapter_id][layer_idx]
        entry = layer_dict.get(module)
        if entry is None or entry["A"] is None or entry["B"] is None:
            return None, None, 0.0
        return entry["A"], entry["B"], entry["scaling"]

    def is_loaded(self, adapter_id: str) -> bool:
        return adapter_id in self._adapters

    def adapter_id_to_int(self, adapter_id: str) -> int:
        """Return integer index for adapter_id; -1 means no adapter."""
        if adapter_id is None or adapter_id == "":
            return -1
        if adapter_id not in self._id_to_int:
            # Auto-register (useful if adapter was loaded externally)
            idx = len(self._int_to_id)
            self._id_to_int[adapter_id] = idx
            self._int_to_id.append(adapter_id)
        return self._id_to_int[adapter_id]

    def int_to_adapter_id(self, idx: int) -> str:
        return self._int_to_id[idx]
