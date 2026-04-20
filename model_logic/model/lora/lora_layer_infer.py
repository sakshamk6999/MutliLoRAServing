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
                 network_config=None, mode=[], adapter_manager=None):
        super().__init__(layer_idx, tp_rank, world_size, network_config, mode)
        self.adapter_manager = adapter_manager  # injected after construction

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

        # Base projections for the full batch
        q = F.linear(normed, layer_weight.q_proj_weight, layer_weight.q_proj_bias)
        k = F.linear(normed, layer_weight.k_proj_weight, layer_weight.k_proj_bias)
        v = F.linear(normed, layer_weight.v_proj_weight, layer_weight.v_proj_bias)

        # Apply LoRA deltas for q, k, v
        if self.adapter_manager is not None and infer_state.adapter_ids_int is not None:
            q, k, v = self._apply_qkv_lora(normed, q, k, v, infer_state, is_prefill)

        # Attention (inherits from Qwen3TransformerLayerInfer but we pass q/k/v directly)
        attn_out = self._attention_from_qkv(q, k, v, infer_state, layer_weight, is_prefill)

        # Apply LoRA delta for o_proj
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
    #  LoRA delta application                                             #
    # ------------------------------------------------------------------ #

    def _apply_qkv_lora(self, normed, q, k, v, infer_state, is_prefill):
        adapter_ids_int = infer_state.adapter_ids_int  # [batch_size]
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

    def _apply_o_lora(self, attn_hidden: torch.Tensor, infer_state, layer_weight, is_prefill):
        """Apply o_proj linear + LoRA delta."""
        # Base o_proj
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
