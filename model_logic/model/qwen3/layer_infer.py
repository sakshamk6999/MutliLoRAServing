import torch
import torch.nn.functional as F
import triton

from model.qwen3.triton_kernels.rmsnorm import rmsnorm_forward
from model.qwen3.triton_kernels.rotary_emb import rotary_emb_fwd
from model.qwen3.triton_kernels.context_flashattention_nopad import context_attention_fwd
from model.qwen3.triton_kernels.token_attention_nopad_att1 import token_att_fwd
from model.qwen3.triton_kernels.token_attention_nopad_softmax import token_softmax_fwd
from model.qwen3.triton_kernels.token_attention_nopad_reduceV import token_att_fwd2
from model.qwen3.triton_kernels.token_attention_softmax_and_reducev import token_softmax_reducev_fwd


# ── Helpers ────────────────────────────────────────────────────────────────────

def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """PyTorch fallback for per-head norms on 3-D inputs [T, H, D]."""
    x_fp32 = x.float()
    norm = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)
    return (norm * weight).to(x.dtype)


def _build_rope_cos_sin(
    positions: torch.Tensor, head_dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cos/sin in the format expected by rotary_emb_fwd: [N, head_dim//2].

    The triton kernel stores x0 = x[:head_dim//2] and x1 = x[head_dim//2:], then
    computes out0 = x0*cos - x1*sin, out1 = x0*sin + x1*cos.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, half, device=positions.device, dtype=torch.float32) / half)
    )
    angles = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)  # [N, half]
    return angles.cos().to(torch.float16), angles.sin().to(torch.float16)


# ── Pre / Post layers ──────────────────────────────────────────────────────────

class Qwen3PreLayerInfer:
    def __init__(self, tp_rank=0, world_size=1, network_config=None, mode=[]):
        pass

    def context_forward(self, input_ids: torch.Tensor, infer_state, weight) -> torch.Tensor:
        return F.embedding(input_ids, weight.embed_tokens_weight)

    def token_forward(self, input_ids: torch.Tensor, infer_state, weight) -> torch.Tensor:
        return F.embedding(input_ids, weight.embed_tokens_weight)


class Qwen3PostLayerInfer:
    def __init__(self, tp_rank=0, world_size=1, network_config=None, mode=[]):
        cfg = network_config or {}
        self.rms_norm_eps = float(cfg.get("rms_norm_eps", 1e-6))

    def token_forward(self, hidden_states: torch.Tensor, infer_state,
                      weight, return_logics: bool = False) -> torch.Tensor:
        if infer_state.is_prefill:
            last_idx = infer_state.b_start_loc + infer_state.b_seq_len - 1
            last_hidden = hidden_states[last_idx]
        else:
            last_hidden = hidden_states
        normed = rmsnorm_forward(last_hidden, weight.final_norm_weight, self.rms_norm_eps)
        return F.linear(normed, weight.lm_head_weight)


# ── Transformer layer ──────────────────────────────────────────────────────────

class Qwen3TransformerLayerInfer:
    def __init__(self, layer_idx, tp_rank=0, world_size=1, network_config=None, mode=[]):
        self.layer_idx = layer_idx
        cfg = network_config or {}
        self.num_heads    = cfg.get("num_attention_heads", 16)
        self.num_kv_heads = cfg.get("num_key_value_heads", 8)
        self.head_dim     = cfg.get("hidden_size", 2048) // self.num_heads
        self.embed_dim    = cfg.get("hidden_size", 2048)
        self.rope_theta   = float(cfg.get("rope_theta", 1_000_000.0))
        self.rms_norm_eps = float(cfg.get("rms_norm_eps", 1e-6))
        self._gqa_ratio   = self.num_heads // self.num_kv_heads

    # ── Public entry points ────────────────────────────────────────────────

    def context_forward(self, hidden_states: torch.Tensor, infer_state, layer_weight) -> torch.Tensor:
        return self._layer_forward(hidden_states, infer_state, layer_weight, is_prefill=True)

    def token_forward(self, hidden_states: torch.Tensor, infer_state, layer_weight) -> torch.Tensor:
        return self._layer_forward(hidden_states, infer_state, layer_weight, is_prefill=False)

    # ── Core layer forward ─────────────────────────────────────────────────

    def _layer_forward(self, hidden_states: torch.Tensor, infer_state,
                       layer_weight, is_prefill: bool) -> torch.Tensor:
        residual = hidden_states

        # RMSNorm — triton fused kernel
        normed = rmsnorm_forward(hidden_states, layer_weight.attn_norm_weight, self.rms_norm_eps)

        # QKV projections
        q = F.linear(normed, layer_weight.q_proj_weight, layer_weight.q_proj_bias)
        k = F.linear(normed, layer_weight.k_proj_weight, layer_weight.k_proj_bias)
        v = F.linear(normed, layer_weight.v_proj_weight, layer_weight.v_proj_bias)

        total = q.shape[0]
        q = q.view(total, self.num_heads,    self.head_dim).contiguous()
        k = k.view(total, self.num_kv_heads, self.head_dim).contiguous()
        v = v.view(total, self.num_kv_heads, self.head_dim).contiguous()

        # Per-head QK norm (Qwen3 specific) — keep Python path; operates on [T,H,D]
        if layer_weight.q_norm_weight is not None:
            q = _rms_norm(q, layer_weight.q_norm_weight, self.rms_norm_eps)
        if layer_weight.k_norm_weight is not None:
            k = _rms_norm(k, layer_weight.k_norm_weight, self.rms_norm_eps)

        # RoPE — triton fused in-place kernel; cos/sin: [T, head_dim//2]
        positions = self._get_positions(infer_state, is_prefill)
        cos, sin = _build_rope_cos_sin(positions, self.head_dim, self.rope_theta)
        rotary_emb_fwd(q, cos, sin)
        rotary_emb_fwd(k, cos, sin)

        # Write K/V to paged KV cache
        mem = infer_state.mem_manager
        if is_prefill:
            mem.key_buffer[self.layer_idx][infer_state.prefill_mem_index] = k
            mem.value_buffer[self.layer_idx][infer_state.prefill_mem_index] = v
        else:
            mem.key_buffer[self.layer_idx][infer_state.decode_mem_index] = k
            mem.value_buffer[self.layer_idx][infer_state.decode_mem_index] = v

        # Attention
        if is_prefill:
            attn_out = self._context_attention(q, k, v, infer_state)
        else:
            attn_out = self._token_attention(q, infer_state)

        attn_out = attn_out.view(total, self.embed_dim)
        attn_out = F.linear(attn_out, layer_weight.o_proj_weight,
                            layer_weight.o_proj_bias if layer_weight.o_proj_bias is not None else None)

        hidden_states = residual + attn_out
        residual = hidden_states

        # FFN RMSNorm — triton fused
        normed2 = rmsnorm_forward(hidden_states, layer_weight.ffn_norm_weight, self.rms_norm_eps)
        ffn_out = self._ffn(normed2, layer_weight)
        return residual + ffn_out

    # ── Context (prefill) attention — triton flash-attention ──────────────

    def _context_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           infer_state) -> torch.Tensor:
        """Fused causal flash-attention for variable-length prefill batch.

        context_attention_fwd assumes all Q/K/V heads are the same count.
        For GQA (num_kv_heads < num_heads) we expand K and V before the call;
        the expanded tensors are temporary and not written to the KV cache.
        """
        if self._gqa_ratio > 1:
            k_full = k.repeat_interleave(self._gqa_ratio, dim=1)
            v_full = v.repeat_interleave(self._gqa_ratio, dim=1)
        else:
            k_full, v_full = k, v

        o = torch.empty_like(q)
        context_attention_fwd(
            q, k_full, v_full, o,
            infer_state.b_start_loc,
            infer_state.b_seq_len,
            infer_state.max_len_in_batch,
        )
        return o

    # ── Token (decode) attention ───────────────────────────────────────────

    def _token_attention(self, q: torch.Tensor, infer_state) -> torch.Tensor:
        """Decode-step attention.

        Triton paged kernels are used when GQA ratio == 1 (no head expansion
        needed).  For GQA models we fall back to a per-request SDPA loop —
        expanding the full paged cache to num_heads would be wasteful, while
        per-request expansion is cheap (only the needed slots are loaded).
        """
        if self._gqa_ratio > 1:
            return self._token_attention_sdpa(q, infer_state)
        return self._token_attention_triton(q, infer_state)

    def _token_attention_sdpa(self, q: torch.Tensor, infer_state) -> torch.Tensor:
        """Per-request SDPA decode attention — used for GQA models."""
        batch_size = infer_state.batch_size
        mem = infer_state.mem_manager
        attn_outputs = []
        for i in range(batch_size):
            seq_len = infer_state.b_seq_len[i].item()
            qi = q[i: i + 1].view(1, self.num_heads, self.head_dim)
            cache_slots = infer_state.b_loc[i, :seq_len]
            ki = mem.key_buffer[self.layer_idx][cache_slots]    # [seq_len, kv_h, d]
            vi = mem.value_buffer[self.layer_idx][cache_slots]
            if self._gqa_ratio > 1:
                ki = ki.repeat_interleave(self._gqa_ratio, dim=1)
                vi = vi.repeat_interleave(self._gqa_ratio, dim=1)
            qi = qi.transpose(0, 1).unsqueeze(0)   # [1, num_heads, 1, head_dim]
            ki = ki.transpose(0, 1).unsqueeze(0)   # [1, num_heads, seq_len, head_dim]
            vi = vi.transpose(0, 1).unsqueeze(0)
            out = F.scaled_dot_product_attention(qi, ki, vi, is_causal=False)
            out = out.squeeze(0).transpose(0, 1).reshape(1, self.embed_dim)
            attn_outputs.append(out)
        return torch.cat(attn_outputs, dim=0)

    def _token_attention_triton(self, q: torch.Tensor, infer_state) -> torch.Tensor:
        """Triton paged decode attention (MHA / GQA ratio == 1).

        The att_m logit buffer is indexed by cumulative KV positions, NOT by
        the batch-token positions stored in infer_state.b_start_loc.  We
        compute kv_start_loc on the fly from b_seq_len so the decode forward
        call does not need to know about this internal layout.
        """
        batch_size = infer_state.batch_size
        mem = infer_state.mem_manager

        # Cumulative KV positions: [0, b_seq_len[0], b_seq_len[0]+b_seq_len[1], ...]
        kv_start_loc = torch.zeros(batch_size, dtype=torch.int32, device=q.device)
        kv_start_loc[1:] = infer_state.b_seq_len[:-1].cumsum(0).int()
        total_kv = int(infer_state.b_seq_len.sum().item())

        k_cache = mem.key_buffer[self.layer_idx]   # [total_slots, num_kv_heads, head_dim]
        v_cache = mem.value_buffer[self.layer_idx]

        calcu_shape = (batch_size, self.num_heads, self.head_dim)
        att_m = torch.empty((self.num_heads, total_kv), dtype=q.dtype, device="cuda")

        token_att_fwd(
            q.view(calcu_shape),
            k_cache, att_m,
            infer_state.b_loc,
            kv_start_loc,
            infer_state.b_seq_len,
            infer_state.max_len_in_batch,
        )

        o = torch.empty_like(q)
        other_kv_index = getattr(infer_state, "other_kv_index", 0)
        if triton.__version__ >= "2.1.0":
            token_softmax_reducev_fwd(
                att_m, v_cache, o.view(calcu_shape),
                infer_state.b_loc,
                kv_start_loc,
                infer_state.b_seq_len,
                infer_state.max_len_in_batch,
                other_kv_index,
            )
        else:
            prob = torch.empty_like(att_m)
            token_softmax_fwd(
                att_m, kv_start_loc, infer_state.b_seq_len,
                prob, infer_state.max_len_in_batch,
            )
            token_att_fwd2(
                prob, v_cache, o.view(calcu_shape),
                infer_state.b_loc,
                kv_start_loc,
                infer_state.b_seq_len,
                infer_state.max_len_in_batch,
            )
        return o

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_positions(self, infer_state, is_prefill: bool) -> torch.Tensor:
        if is_prefill:
            parts = []
            for i in range(infer_state.batch_size):
                parts.append(torch.arange(infer_state.b_seq_len[i].item(),
                                          device="cuda", dtype=torch.long))
            return torch.cat(parts)
        else:
            return (infer_state.b_seq_len - 1).long()

    def _ffn(self, x: torch.Tensor, layer_weight) -> torch.Tensor:
        gate = F.linear(x, layer_weight.gate_proj_weight)
        F.silu(gate, inplace=True)
        up = F.linear(x, layer_weight.up_proj_weight)
        return F.linear(gate * up, layer_weight.down_proj_weight)
