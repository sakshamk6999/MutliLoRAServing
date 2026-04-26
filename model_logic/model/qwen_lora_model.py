import sys
import os
# Ensure model_logic/ is on the path
_model_logic_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _model_logic_dir not in sys.path:
    sys.path.insert(0, _model_logic_dir)

import torch
from base_model.base_model import BaseModel
from base_model.infer_struct import InferStateInfo  # noqa: F401 (re-exported for use as infer_state_class)
from model.qwen3.weights import Qwen3PreAndPostWeight, Qwen3TransformerLayerWeight
from model.qwen3.layer_infer import Qwen3PreLayerInfer, Qwen3PostLayerInfer
from model.lora.lora_layer_infer import LoRATransformerLayerInfer
from model.lora.adapter_manager import LoRAAdapterManager


class QwenLoRAModel(BaseModel):
    pre_and_post_weight_class = Qwen3PreAndPostWeight
    transformer_weight_class = Qwen3TransformerLayerWeight
    pre_layer_infer_class = Qwen3PreLayerInfer
    post_layer_infer_class = Qwen3PostLayerInfer
    transformer_layer_infer_class = LoRATransformerLayerInfer
    infer_state_class = InferStateInfo

    def __init__(self, weight_dir: str, max_total_token_num: int,
                 mem_adapter_size: int = 0,
                 adapter_dirs: dict[str, str] | None = None,
                 load_way: str = "HF", mode=[], dummy: bool = False,
                 use_triton: bool = False):
        self.tp_rank_ = 0
        self.world_size_ = 1
        self._pending_adapter_dirs = adapter_dirs or {}
        self._use_triton = use_triton
        super().__init__(weight_dir, max_total_token_num, mem_adapter_size,
                         load_way=load_way, mode=mode, dummy=dummy)

    def _init_infer_layer(self):
        """Override to construct LoRATransformerLayerInfer (adapter_manager injected later)."""
        self.pre_infer = self.pre_layer_infer_class(
            tp_rank=self.tp_rank_, world_size=self.world_size_,
            network_config=self.config, mode=self.mode)
        self.post_infer = self.post_layer_infer_class(
            tp_rank=self.tp_rank_, world_size=self.world_size_,
            network_config=self.config, mode=self.mode)
        self.layers_infer = [
            LoRATransformerLayerInfer(
                i, tp_rank=self.tp_rank_, world_size=self.world_size_,
                network_config=self.config, mode=self.mode,
                adapter_manager=None, use_triton=self._use_triton)
            for i in range(self.config["n_layer"])
        ]

    def _init_custom(self):
        self.adapter_manager = LoRAAdapterManager(
            hidden_size=self.config["hidden_size"],
            num_heads=self.config["num_attention_heads"],
            num_kv_heads=self.config.get("num_key_value_heads",
                                         self.config["num_attention_heads"]),
            head_dim=self.head_dim_,
            num_layers=self.layers_num,
            dtype=torch.float16,
        )
        for layer in self.layers_infer:
            layer.adapter_manager = self.adapter_manager

        for adapter_id, path in self._pending_adapter_dirs.items():
            self.adapter_manager.load_adapter(adapter_id, path)

    # ------------------------------------------------------------------ #
    #  Public adapter API                                                  #
    # ------------------------------------------------------------------ #

    def load_adapter(self, adapter_id: str, adapter_path: str):
        self.adapter_manager.load_adapter(adapter_id, adapter_path)

    def unload_adapter(self, adapter_id: str):
        self.adapter_manager.unload_adapter(adapter_id)

    # ------------------------------------------------------------------ #
    #  Forward override — accepts adapter_ids                             #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def forward(self, batch_size, total_token_num, max_len_in_batch,
                input_ids: torch.Tensor,
                b_loc: torch.Tensor, b_start_loc: torch.Tensor, b_seq_len: torch.Tensor,
                adapter_ids: list[str] | None = None,
                is_prefill: bool = True):
        if adapter_ids is not None:
            self._current_adapter_ids_int = torch.tensor(
                [self.adapter_manager.adapter_id_to_int(aid) for aid in adapter_ids],
                dtype=torch.long, device="cuda")
        else:
            self._current_adapter_ids_int = None

        return super().forward(batch_size, total_token_num, max_len_in_batch,
                               input_ids, b_loc, b_start_loc, b_seq_len,
                               is_prefill=is_prefill)

    def _prefill(self, batch_size, total_token_num, max_len_in_batch,
                 input_ids, b_loc, b_start_loc, b_seq_len):
        infer_state = self.infer_state_class()
        infer_state.is_prefill = True
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        infer_state.b_loc = b_loc
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len
        infer_state.adapter_ids_int = self._current_adapter_ids_int

        infer_state.mem_manager = self.mem_manager
        infer_state.prefill_mem_index = self.mem_manager.alloc(total_token_num)
        infer_state.prefill_key_buffer = torch.empty(
            (total_token_num, self.tp_k_head_num_, self.head_dim_),
            dtype=torch.float16, device="cuda")
        infer_state.prefill_value_buffer = torch.empty(
            (total_token_num, self.tp_v_head_num_, self.head_dim_),
            dtype=torch.float16, device="cuda")

        from utils import init_bloc
        init_bloc(b_loc, b_seq_len, max_len_in_batch, infer_state.prefill_mem_index)

        infer_state.init_some_extra_state(
            self, batch_size, total_token_num, max_len_in_batch,
            input_ids, b_loc, b_start_loc, b_seq_len, True)
        return self._context_forward(input_ids, infer_state)

    def _decode(self, batch_size, total_token_num, max_len_in_batch,
                input_ids, b_loc, b_start_loc, b_seq_len):
        infer_state = self.infer_state_class()
        infer_state.is_prefill = False
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        infer_state.b_loc = b_loc
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len
        infer_state.adapter_ids_int = self._current_adapter_ids_int

        infer_state.mem_manager = self.mem_manager

        alloc_mem = self.mem_manager.alloc_contiguous(batch_size)
        if alloc_mem is not None:
            infer_state.decode_is_contiguous = True
            infer_state.decode_mem_index = alloc_mem[0]
            infer_state.decode_mem_start = alloc_mem[1]
            infer_state.decode_mem_end = alloc_mem[2]
            b_loc[:, max_len_in_batch - 1] = infer_state.decode_mem_index
        else:
            infer_state.decode_is_contiguous = False
            alloc_mem = self.mem_manager.alloc(batch_size)
            infer_state.decode_mem_index = alloc_mem
            infer_state.decode_key_buffer = torch.empty(
                (batch_size, self.tp_k_head_num_, self.head_dim_),
                dtype=torch.float16, device="cuda")
            infer_state.decode_value_buffer = torch.empty(
                (batch_size, self.tp_v_head_num_, self.head_dim_),
                dtype=torch.float16, device="cuda")
            b_loc[:, max_len_in_batch - 1] = infer_state.decode_mem_index

        infer_state.init_some_extra_state(
            self, batch_size, total_token_num, max_len_in_batch,
            input_ids, b_loc, b_start_loc, b_seq_len, False)
        return self._token_forward(input_ids, infer_state)
