import torch
import torch.nn.functional as F

from model.qwen3.layer_infer import Qwen3TransformerLayerInfer, _rms_norm


def _build_token_mask(req_mask: torch.Tensor, infer_state, is_prefill: bool) -> torch.Tensor:
    """Expand a per-request boolean mask to a per-token boolean mask.

    req_mask: [batch_size] bool
    Returns: [total_token_num] bool  (prefill) or [batch_size] bool (decode, same as req_mask)
    """
    if not is_prefill:
        return req_mask  # one token per request in decode

    total = infer_state.total_token_num
    token_mask = torch.zeros(total, dtype=torch.bool, device=req_mask.device)
    for i in range(infer_state.batch_size):
        if req_mask[i]:
            start = infer_state.b_start_loc[i].item()
            length = infer_state.b_seq_len[i].item()
            token_mask[start: start + length] = True
    return token_mask


class LoRATransformerLayerInfer(Qwen3TransformerLayerInfer):
    def __init__(self, layer_idx, tp_rank=0, world_size=1,
                 network_config=None, mode=[], adapter_manager=None,
                 use_triton: bool = False):
        super().__init__(layer_idx, tp_rank, world_size, network_config, mode)
        self.adapter_manager = adapter_manager
        self.use_triton = use_triton

    # ------------------------------------------------------------------ #

    def context_forward(self, hidden_states: torch.Tensor, infer_state, layer_weight) -> torch.Tensor:
        return self._forward_with_lora(hidden_states, infer_state, layer_weight, is_prefill=True)

    def token_forward(self, hidden_states: torch.Tensor, infer_state, layer_weight) -> torch.Tensor:
        return self._forward_with_lora(hidden_states, infer_state, layer_weight, is_prefill=False)

    # ------------------------------------------------------------------ #

    def _forward_with_lora(self, hidden_states: torch.Tensor, infer_state,
                           layer_weight, is_prefill: bool) -> torch.Tensor:
        residual = hidden_states
        normed = _rms_norm(hidden_states, layer_weight.attn_norm_weight, self.rms_norm_eps)

        q = F.linear(normed, layer_weight.q_proj_weight, layer_weight.q_proj_bias)
        k = F.linear(normed, layer_weight.k_proj_weight, layer_weight.k_proj_bias)
        v = F.linear(normed, layer_weight.v_proj_weight, layer_weight.v_proj_bias)

        if self.adapter_manager is not None and infer_state.adapter_ids_int is not None:
            q, k, v = self._apply_qkv_lora(normed, q, k, v, infer_state, is_prefill)

        attn_out = self._attention_from_qkv(q, k, v, infer_state, layer_weight, is_prefill)

        if self.adapter_manager is not None and infer_state.adapter_ids_int is not None:
            attn_out = self._apply_o_lora(attn_out, infer_state, layer_weight, is_prefill)
        else:
            attn_out = F.linear(attn_out, layer_weight.o_proj_weight,
                                layer_weight.o_proj_bias if layer_weight.o_proj_bias is not None else None)

        hidden_states = residual + attn_out
        residual = hidden_states
        normed2 = _rms_norm(hidden_states, layer_weight.ffn_norm_weight, self.rms_norm_eps)
        ffn_out = self._ffn(normed2, layer_weight)
        return residual + ffn_out

    # ------------------------------------------------------------------ #
    #  LoRA delta dispatch                                                 #
    # ------------------------------------------------------------------ #

    def _apply_qkv_lora(self, normed, q, k, v, infer_state, is_prefill):
        if self.use_triton and is_prefill:
            return self._apply_qkv_lora_triton(normed, q, k, v, infer_state)
        return self._apply_qkv_lora_pytorch(normed, q, k, v, infer_state, is_prefill)

    def _apply_o_lora(self, attn_hidden, infer_state, layer_weight, is_prefill):
        if self.use_triton and is_prefill:
            return self._apply_o_lora_triton(attn_hidden, infer_state, layer_weight)
        return self._apply_o_lora_pytorch(attn_hidden, infer_state, layer_weight, is_prefill)

    # ------------------------------------------------------------------ #
    #  PyTorch LoRA path                                                   #
    # ------------------------------------------------------------------ #

    def _apply_qkv_lora_pytorch(self, normed, q, k, v, infer_state, is_prefill):
        adapter_ids_int = infer_state.adapter_ids_int
        unique_ids = adapter_ids_int.unique()

        for aid_int in unique_ids:
            aid_int_val = aid_int.item()
            if aid_int_val == -1:
                continue

            adapter_id = self.adapter_manager.int_to_adapter_id(aid_int_val)
            req_mask = (adapter_ids_int == aid_int)
            token_mask = _build_token_mask(req_mask, infer_state, is_prefill)
            x_sub = normed[token_mask]

            for proj, target in [("q_proj", q), ("k_proj", k), ("v_proj", v)]:
                A, B, scaling = self.adapter_manager.get_lora_weights(
                    adapter_id, self.layer_idx, proj)
                if A is None:
                    continue
                delta = (x_sub @ A.T) @ B.T * scaling
                target[token_mask] = target[token_mask] + delta

        return q, k, v

    def _apply_o_lora_pytorch(self, attn_hidden, infer_state, layer_weight, is_prefill):
        out = F.linear(attn_hidden, layer_weight.o_proj_weight,
                       layer_weight.o_proj_bias if layer_weight.o_proj_bias is not None else None)

        adapter_ids_int = infer_state.adapter_ids_int
        unique_ids = adapter_ids_int.unique()

        for aid_int in unique_ids:
            aid_int_val = aid_int.item()
            if aid_int_val == -1:
                continue

            adapter_id = self.adapter_manager.int_to_adapter_id(aid_int_val)
            req_mask = (adapter_ids_int == aid_int)
            token_mask = _build_token_mask(req_mask, infer_state, is_prefill)

            A, B, scaling = self.adapter_manager.get_lora_weights(
                adapter_id, self.layer_idx, "o_proj")
            if A is None:
                continue
            x_sub = attn_hidden[token_mask]
            delta = (x_sub @ A.T) @ B.T * scaling
            out[token_mask] = out[token_mask] + delta

        return out

    # ------------------------------------------------------------------ #
    #  Triton LoRA path (prefill only)                                     #
    # ------------------------------------------------------------------ #

    def _build_triton_state(self, infer_state):
        """Build shared index tensors for triton kernel calls.

        Returns (unique_int_ids, local_id_map, max_rank, b_indicies, b_lora_ranks,
                 b_lora_start, b_loc) or None if no active adapters.

        Memory layout (per projection):
          - b_lora_ranks[a] = actual_rank * 4   (kernel divides by 4 internally)
          - b_lora_start[a] = a * max_rank       (page start for adapter a)
          - b_loc = identity [0..total_pages)
          - Slot num_adapters is the null adapter (scale=0, zero weights)
        """
        adapter_ids_int = infer_state.adapter_ids_int
        active = [x for x in adapter_ids_int.tolist() if x >= 0]
        if not active:
            return None

        unique_int_ids = sorted(set(active))
        num_adapters = len(unique_int_ids)
        local_id_map = {gid: lid for lid, gid in enumerate(unique_int_ids)}

        max_rank = 0
        ranks = {}
        for gid in unique_int_ids:
            aid = self.adapter_manager.int_to_adapter_id(gid)
            A, _, _ = self.adapter_manager.get_lora_weights(aid, self.layer_idx, "q_proj")
            r = A.shape[0] if A is not None else 0
            ranks[gid] = r
            max_rank = max(max_rank, r)

        if max_rank == 0:
            return None

        device = adapter_ids_int.device
        null_slot = num_adapters

        b_indicies = torch.full((infer_state.batch_size,), null_slot,
                                dtype=torch.int32, device=device)
        for i, gid in enumerate(adapter_ids_int.tolist()):
            if gid >= 0:
                b_indicies[i] = local_id_map[gid]

        # b_lora_ranks: rank*4 for real adapters; max_rank*4 for null (scale=0 guards it)
        b_lora_ranks = torch.zeros(num_adapters + 1, dtype=torch.int32, device=device)
        for lid, gid in enumerate(unique_int_ids):
            b_lora_ranks[lid] = ranks[gid] * 4
        b_lora_ranks[null_slot] = max_rank * 4

        b_lora_start = torch.arange(num_adapters + 1, dtype=torch.int32, device=device) * max_rank
        total_pages = (num_adapters + 1) * max_rank
        b_loc = torch.arange(total_pages, dtype=torch.int32, device=device)

        return unique_int_ids, local_id_map, max_rank, ranks, b_indicies, b_lora_ranks, b_lora_start, b_loc

    def _pack_lora_weights(self, unique_int_ids, local_id_map, ranks, max_rank,
                           proj_name, feat_out, total_pages, device, dtype):
        """Pack A and B matrices for one projection into flat page buffers.

        W_A: [total_pages, hidden_size]  — each row is one rank dimension of A
        W_B: [total_pages, feat_out]     — each row encodes feat_out//rank output dims of B
        scale_vec: [num_adapters+1]      — null slot has scale=0
        """
        hidden_size = self.adapter_manager.hidden_size
        W_A = torch.zeros(total_pages, hidden_size, dtype=dtype, device=device)
        W_B = torch.zeros(total_pages, feat_out, dtype=dtype, device=device)
        scale_vec = torch.zeros(len(unique_int_ids) + 1, dtype=dtype, device=device)

        for lid, gid in enumerate(unique_int_ids):
            aid = self.adapter_manager.int_to_adapter_id(gid)
            A, B, sc = self.adapter_manager.get_lora_weights(aid, self.layer_idx, proj_name)
            if A is None:
                continue
            rank = ranks[gid]
            page_start = lid * max_rank

            # A: [rank, hidden_size] — one row per rank dim
            W_A[page_start:page_start + rank] = A

            # B: [feat_out, rank] — pack into rank pages each of size feat_out
            # W_B[page_start+p, n_local*rank+d] = B[p*(feat_out//rank)+n_local, d]
            # which equals B[p*s:(p+1)*s, :].reshape(-1) in row-major order
            assert feat_out % rank == 0, (
                f"feat_out ({feat_out}) must be divisible by rank ({rank}) for triton path"
            )
            s = feat_out // rank
            for p in range(rank):
                W_B[page_start + p] = B[p * s:(p + 1) * s].reshape(-1)

            scale_vec[lid] = sc
        # null slot scale stays 0

        return W_A, W_B, scale_vec

    def _apply_qkv_lora_triton(self, normed, q, k, v, infer_state):
        try:
            from .triton_kernels.lora_prefill import (
                lora_get_qkvo_fwd_shrink, lora_get_qkvo_fwd_expand)
        except ImportError:
            return self._apply_qkv_lora_pytorch(normed, q, k, v, infer_state, is_prefill=True)

        state = self._build_triton_state(infer_state)
        if state is None:
            return q, k, v

        unique_int_ids, local_id_map, max_rank, ranks, b_indicies, b_lora_ranks, b_lora_start, b_loc = state
        total_tokens = normed.shape[0]
        hidden_size = normed.shape[1]
        total_pages = b_loc.shape[0]
        device = normed.device
        dtype = normed.dtype

        for proj_name, proj_out in [("q_proj", q), ("k_proj", k), ("v_proj", v)]:
            feat_out = proj_out.shape[1]
            W_A, W_B, scale_vec = self._pack_lora_weights(
                unique_int_ids, local_id_map, ranks, max_rank,
                proj_name, feat_out, total_pages, device, dtype)

            intermediate = torch.zeros(total_tokens, max_rank, dtype=dtype, device=device)

            lora_get_qkvo_fwd_shrink(
                normed, W_A, intermediate,
                b_loc, b_lora_start, b_lora_ranks,
                infer_state.b_start_loc, infer_state.b_seq_len, b_indicies,
                hidden_size, 0, max_rank, infer_state.max_len_in_batch,
            )
            lora_get_qkvo_fwd_expand(
                intermediate, W_B, proj_out, scale_vec,
                b_loc, b_lora_start, b_lora_ranks,
                infer_state.b_start_loc, infer_state.b_seq_len, b_indicies,
                feat_out, 0, max_rank, infer_state.max_len_in_batch,
            )

        return q, k, v

    def _apply_o_lora_triton(self, attn_hidden, infer_state, layer_weight):
        out = F.linear(attn_hidden, layer_weight.o_proj_weight,
                       layer_weight.o_proj_bias if layer_weight.o_proj_bias is not None else None)

        try:
            from .triton_kernels.lora_prefill import (
                lora_get_qkvo_fwd_shrink, lora_get_qkvo_fwd_expand)
        except ImportError:
            return self._apply_o_lora_pytorch(attn_hidden, infer_state, layer_weight, is_prefill=True)

        state = self._build_triton_state(infer_state)
        if state is None:
            return out

        unique_int_ids, local_id_map, max_rank, ranks, b_indicies, b_lora_ranks, b_lora_start, b_loc = state
        total_tokens = attn_hidden.shape[0]
        hidden_size = attn_hidden.shape[1]
        feat_out = out.shape[1]
        total_pages = b_loc.shape[0]
        device = attn_hidden.device
        dtype = attn_hidden.dtype

        W_A, W_B, scale_vec = self._pack_lora_weights(
            unique_int_ids, local_id_map, ranks, max_rank,
            "o_proj", feat_out, total_pages, device, dtype)

        intermediate = torch.zeros(total_tokens, max_rank, dtype=dtype, device=device)

        lora_get_qkvo_fwd_shrink(
            attn_hidden, W_A, intermediate,
            b_loc, b_lora_start, b_lora_ranks,
            infer_state.b_start_loc, infer_state.b_seq_len, b_indicies,
            hidden_size, 0, max_rank, infer_state.max_len_in_batch,
        )
        lora_get_qkvo_fwd_expand(
            intermediate, W_B, out, scale_vec,
            b_loc, b_lora_start, b_lora_ranks,
            infer_state.b_start_loc, infer_state.b_seq_len, b_indicies,
            feat_out, 0, max_rank, infer_state.max_len_in_batch,
        )

        return out

    # ------------------------------------------------------------------ #
    #  Override _attention to accept pre-computed q, k, v                #
    # ------------------------------------------------------------------ #

    def _attention_from_qkv(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                             infer_state, layer_weight, is_prefill: bool) -> torch.Tensor:
        """Like parent _attention but q/k/v are already computed (flat, not yet reshaped)."""
        total_tokens = q.shape[0]
        batch_size = infer_state.batch_size

        q = q.view(total_tokens, self.num_heads, self.head_dim)
        k = k.view(total_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(total_tokens, self.num_kv_heads, self.head_dim)

        from model.qwen3.layer_infer import _build_rope_cache, _apply_rope, _rms_norm as rn
        if layer_weight.q_norm_weight is not None:
            q = rn(q, layer_weight.q_norm_weight, self.rms_norm_eps)
        if layer_weight.k_norm_weight is not None:
            k = rn(k, layer_weight.k_norm_weight, self.rms_norm_eps)

        positions = self._get_positions(infer_state, is_prefill)
        cos, sin = _build_rope_cache(positions, self.head_dim, self.rope_theta)
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        mem = infer_state.mem_manager
        if is_prefill:
            mem.key_buffer[self.layer_idx][infer_state.prefill_mem_index] = k
            mem.value_buffer[self.layer_idx][infer_state.prefill_mem_index] = v
        else:
            mem.key_buffer[self.layer_idx][infer_state.decode_mem_index] = k
            mem.value_buffer[self.layer_idx][infer_state.decode_mem_index] = v

        attn_outputs = []
        for i in range(batch_size):
            seq_len = infer_state.b_seq_len[i].item()
            if is_prefill:
                start = infer_state.b_start_loc[i].item()
                qi = q[start: start + seq_len]
            else:
                qi = q[i: i + 1]

            cache_slots = infer_state.b_loc[i, :seq_len]
            ki = mem.key_buffer[self.layer_idx][cache_slots]
            vi = mem.value_buffer[self.layer_idx][cache_slots]

            if self.num_kv_heads != self.num_heads:
                ratio = self.num_heads // self.num_kv_heads
                ki = ki.repeat_interleave(ratio, dim=1)
                vi = vi.repeat_interleave(ratio, dim=1)

            qi = qi.transpose(0, 1).unsqueeze(0)
            ki = ki.transpose(0, 1).unsqueeze(0)
            vi = vi.transpose(0, 1).unsqueeze(0)

            out = torch.nn.functional.scaled_dot_product_attention(qi, ki, vi, is_causal=is_prefill)
            q_len = qi.shape[2]
            out = out.squeeze(0).transpose(0, 1).reshape(q_len, self.num_heads * self.head_dim)
            attn_outputs.append(out)

        return torch.cat(attn_outputs, dim=0)
