import torch
import torch.nn.functional as F
import math


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_fp32 = x.float()
    norm = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)
    return (norm * weight).to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [N, num_heads, head_dim], cos/sin: [N, 1, head_dim]
    return x * cos + _rotate_half(x) * sin


def _build_rope_cache(positions: torch.Tensor, head_dim: int,
                      theta: float = 1_000_000.0) -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    freq = 1.0 / (theta ** (torch.arange(0, half, device=positions.device).float() / half))
    emb = positions.float().unsqueeze(-1) * freq.unsqueeze(0)  # [N, half]
    emb = torch.cat([emb, emb], dim=-1)  # [N, head_dim]
    cos = emb.cos().unsqueeze(1)  # [N, 1, head_dim]
    sin = emb.sin().unsqueeze(1)
    return cos, sin


class Qwen3PreLayerInfer:
    def __init__(self, tp_rank=0, world_size=1, network_config=None, mode=[]):
        pass

    def context_forward(self, input_ids: torch.Tensor, infer_state, weight) -> torch.Tensor:
        return F.embedding(input_ids, weight.embed_tokens_weight)

    def token_forward(self, input_ids: torch.Tensor, infer_state, weight) -> torch.Tensor:
        return F.embedding(input_ids, weight.embed_tokens_weight)


class Qwen3PostLayerInfer:
    def __init__(self, tp_rank=0, world_size=1, network_config=None, mode=[]):
        pass

    def token_forward(self, hidden_states: torch.Tensor, infer_state,
                      weight, return_logics: bool = False) -> torch.Tensor:
        if infer_state.is_prefill:
            # Packed prefill: hidden_states is [total_token_num, H].
            # Select the last token of each request.
            last_idx = infer_state.b_start_loc + infer_state.b_seq_len - 1
            last_hidden = hidden_states[last_idx]  # [batch_size, H]
        else:
            # Decode: hidden_states is already [batch_size, H] — one row per request.
            last_hidden = hidden_states
        normed = _rms_norm(last_hidden, weight.final_norm_weight)
        logits = F.linear(normed, weight.lm_head_weight)
        return logits


class Qwen3TransformerLayerInfer:
    def __init__(self, layer_idx, tp_rank=0, world_size=1, network_config=None, mode=[]):
        self.layer_idx = layer_idx
        cfg = network_config or {}
        self.num_heads = cfg.get("num_attention_heads", 16)
        self.num_kv_heads = cfg.get("num_key_value_heads", 8)
        self.head_dim = cfg.get("hidden_size", 2048) // self.num_heads
        self.rope_theta = float(cfg.get("rope_theta", 1_000_000.0))
        self.rms_norm_eps = float(cfg.get("rms_norm_eps", 1e-6))

    # ------------------------------------------------------------------ #
    #  Public entry points                                                 #
    # ------------------------------------------------------------------ #

    def context_forward(self, hidden_states: torch.Tensor, infer_state, layer_weight) -> torch.Tensor:
        return self._layer_forward(hidden_states, infer_state, layer_weight, is_prefill=True)

    def token_forward(self, hidden_states: torch.Tensor, infer_state, layer_weight) -> torch.Tensor:
        return self._layer_forward(hidden_states, infer_state, layer_weight, is_prefill=False)

    # ------------------------------------------------------------------ #
    #  Core layer logic                                                    #
    # ------------------------------------------------------------------ #

    def _layer_forward(self, hidden_states: torch.Tensor, infer_state, layer_weight,
                       is_prefill: bool) -> torch.Tensor:
        residual = hidden_states
        normed = _rms_norm(hidden_states, layer_weight.attn_norm_weight, self.rms_norm_eps)

        q = F.linear(normed, layer_weight.q_proj_weight, layer_weight.q_proj_bias)
        k = F.linear(normed, layer_weight.k_proj_weight, layer_weight.k_proj_bias)
        v = F.linear(normed, layer_weight.v_proj_weight, layer_weight.v_proj_bias)

        attn_out = self._attention(normed, q, k, v, infer_state, layer_weight, is_prefill)
        hidden_states = residual + attn_out

        residual = hidden_states
        normed2 = _rms_norm(hidden_states, layer_weight.ffn_norm_weight, self.rms_norm_eps)
        ffn_out = self._ffn(normed2, layer_weight)
        return residual + ffn_out

    # ------------------------------------------------------------------ #
    #  Attention                                                           #
    # ------------------------------------------------------------------ #

    def _attention(self, normed: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
                   v: torch.Tensor, infer_state, layer_weight, is_prefill: bool) -> torch.Tensor:
        total_tokens = q.shape[0]
        batch_size = infer_state.batch_size

        # Reshape to [N, num_heads, head_dim]
        q = q.view(total_tokens, self.num_heads, self.head_dim)
        k = k.view(total_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(total_tokens, self.num_kv_heads, self.head_dim)

        # Per-head RMSNorm (Qwen3 feature)
        if layer_weight.q_norm_weight is not None:
            q = _rms_norm(q, layer_weight.q_norm_weight, self.rms_norm_eps)
        if layer_weight.k_norm_weight is not None:
            k = _rms_norm(k, layer_weight.k_norm_weight, self.rms_norm_eps)

        # Build per-token positions for RoPE
        positions = self._get_positions(infer_state, is_prefill)  # [total_tokens]
        cos, sin = _build_rope_cache(positions, self.head_dim, self.rope_theta)
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # Write new K, V into KV cache
        mem = infer_state.mem_manager
        if is_prefill:
            mem.key_buffer[self.layer_idx][infer_state.prefill_mem_index] = k
            mem.value_buffer[self.layer_idx][infer_state.prefill_mem_index] = v
        else:
            mem.key_buffer[self.layer_idx][infer_state.decode_mem_index] = k
            mem.value_buffer[self.layer_idx][infer_state.decode_mem_index] = v

        # Run attention per request (handles variable-length sequences)
        attn_outputs = []
        for i in range(batch_size):
            seq_len = infer_state.b_seq_len[i].item()
            # Gather this request's Q
            if is_prefill:
                start = infer_state.b_start_loc[i].item()
                qi = q[start: start + seq_len]  # [seq_len, num_heads, head_dim]
            else:
                qi = q[i: i + 1]  # [1, num_heads, head_dim]

            # Gather all K, V for this request from cache
            cache_slots = infer_state.b_loc[i, :seq_len]  # [seq_len]
            ki = mem.key_buffer[self.layer_idx][cache_slots]    # [seq_len, num_kv_heads, head_dim]
            vi = mem.value_buffer[self.layer_idx][cache_slots]

            # GQA expansion: repeat K/V to match num_heads
            if self.num_kv_heads != self.num_heads:
                ratio = self.num_heads // self.num_kv_heads
                ki = ki.repeat_interleave(ratio, dim=1)
                vi = vi.repeat_interleave(ratio, dim=1)

            # SDPA expects [batch, heads, seq, head_dim]
            qi = qi.transpose(0, 1).unsqueeze(0)  # [1, num_heads, q_len, head_dim]
            ki = ki.transpose(0, 1).unsqueeze(0)
            vi = vi.transpose(0, 1).unsqueeze(0)

            out = F.scaled_dot_product_attention(qi, ki, vi, is_causal=is_prefill)
            # out: [1, num_heads, q_len, head_dim]
            q_len = qi.shape[2]
            out = out.squeeze(0).transpose(0, 1).reshape(q_len, self.num_heads * self.head_dim)
            attn_outputs.append(out)

        attn_out = torch.cat(attn_outputs, dim=0)  # [total_tokens, hidden_size]
        return F.linear(attn_out, layer_weight.o_proj_weight,
                        layer_weight.o_proj_bias if layer_weight.o_proj_bias is not None else None)

    def _get_positions(self, infer_state, is_prefill: bool) -> torch.Tensor:
        """Build absolute position indices for each token in the batch."""
        if is_prefill:
            parts = []
            for i in range(infer_state.batch_size):
                seq_len = infer_state.b_seq_len[i].item()
                parts.append(torch.arange(seq_len, device="cuda", dtype=torch.long))
            return torch.cat(parts)
        else:
            # In decode, each request contributes one new token at position seq_len-1
            return (infer_state.b_seq_len - 1).long()

    # ------------------------------------------------------------------ #
    #  FFN (SwiGLU)                                                        #
    # ------------------------------------------------------------------ #

    def _ffn(self, x: torch.Tensor, layer_weight) -> torch.Tensor:
        gate = F.linear(x, layer_weight.gate_proj_weight)
        up = F.linear(x, layer_weight.up_proj_weight)
        return F.linear(F.silu(gate) * up, layer_weight.down_proj_weight)
