import torch


class Qwen3PreAndPostWeight:
    def __init__(self, tp_rank, world_size, dtype, network_config: dict, mode=[]):
        self.dtype = dtype
        self.embed_tokens_weight = None
        self.final_norm_weight = None
        self.lm_head_weight = None
        self._loaded = set()

    def load_weight(self, key: str, tensor: torch.Tensor):
        t = tensor.to(dtype=self.dtype, device="cuda")
        if key == "model.embed_tokens.weight":
            self.embed_tokens_weight = t
            self._loaded.add(key)
        elif key == "model.norm.weight":
            self.final_norm_weight = t.float()  # keep norm in fp32
            self._loaded.add(key)
        elif key == "lm_head.weight":
            self.lm_head_weight = t
            self._loaded.add(key)

    def verify_load(self):
        assert self.embed_tokens_weight is not None, "embed_tokens_weight not loaded"
        assert self.final_norm_weight is not None, "final_norm_weight not loaded"
        # lm_head may be tied to embed_tokens in some Qwen variants
        if self.lm_head_weight is None:
            self.lm_head_weight = self.embed_tokens_weight


class Qwen3TransformerLayerWeight:
    def __init__(self, layer_idx, tp_rank, world_size, dtype, network_config: dict, mode=[]):
        self.layer_idx = layer_idx
        self.dtype = dtype
        self._loaded = set()

        self.attn_norm_weight = None
        self.ffn_norm_weight = None

        self.q_proj_weight = None
        self.q_proj_bias = None
        self.k_proj_weight = None
        self.k_proj_bias = None
        self.v_proj_weight = None
        self.v_proj_bias = None
        self.o_proj_weight = None
        self.o_proj_bias = None

        self.gate_proj_weight = None
        self.up_proj_weight = None
        self.down_proj_weight = None

        # Qwen3 uses q_norm and k_norm (per-head RMSNorm before RoPE)
        self.q_norm_weight = None
        self.k_norm_weight = None

    def load_weight(self, key: str, tensor: torch.Tensor):
        prefix = f"model.layers.{self.layer_idx}."
        if not key.startswith(prefix):
            return
        suffix = key[len(prefix):]
        t = tensor.to(dtype=self.dtype, device="cuda")

        mapping = {
            "input_layernorm.weight":            ("attn_norm_weight", t.float()),
            "post_attention_layernorm.weight":   ("ffn_norm_weight", t.float()),
            "self_attn.q_proj.weight":           ("q_proj_weight", t),
            "self_attn.q_proj.bias":             ("q_proj_bias", t),
            "self_attn.k_proj.weight":           ("k_proj_weight", t),
            "self_attn.k_proj.bias":             ("k_proj_bias", t),
            "self_attn.v_proj.weight":           ("v_proj_weight", t),
            "self_attn.v_proj.bias":             ("v_proj_bias", t),
            "self_attn.o_proj.weight":           ("o_proj_weight", t),
            "self_attn.o_proj.bias":             ("o_proj_bias", t),
            "self_attn.q_norm.weight":           ("q_norm_weight", t.float()),
            "self_attn.k_norm.weight":           ("k_norm_weight", t.float()),
            "mlp.gate_proj.weight":              ("gate_proj_weight", t),
            "mlp.up_proj.weight":                ("up_proj_weight", t),
            "mlp.down_proj.weight":              ("down_proj_weight", t),
        }

        if suffix in mapping:
            attr, val = mapping[suffix]
            setattr(self, attr, val)
            self._loaded.add(suffix)

    def verify_load(self):
        required = [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ]
        for key in required:
            assert key in self._loaded, f"Layer {self.layer_idx}: {key} not loaded"
