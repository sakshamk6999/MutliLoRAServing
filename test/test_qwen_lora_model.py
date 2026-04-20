"""
Unit tests for QwenLoRAModel.

All tests use a tiny synthetic model (2 layers, hidden=128) and a fake LoRA
adapter so they run without downloading any real weights.

Requires CUDA — skipped automatically on CPU-only machines.
"""

import json
import os
import sys
import tempfile

import pytest
import torch

# ── Path setup ─────────────────────────────────────────────────────────────────
_model_logic = os.path.join(os.path.dirname(__file__), "..", "model_logic")
_model_logic = os.path.abspath(_model_logic)
if _model_logic not in sys.path:
    sys.path.insert(0, _model_logic)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

# ── Constants for the tiny synthetic model ────────────────────────────────────
HIDDEN = 128
N_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = HIDDEN // N_HEADS          # 32
N_LAYERS = 2
INTERMEDIATE = 256
VOCAB = 1000
LORA_RANK = 4
LORA_ALPHA = 8.0


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def weight_dir(tmp_path_factory):
    """Write a tiny Qwen3-like config + safetensors weights to a temp dir."""
    from safetensors.torch import save_file

    d = tmp_path_factory.mktemp("qwen3_tiny")

    config = {
        "model_type": "qwen3",
        "hidden_size": HIDDEN,
        "num_hidden_layers": N_LAYERS,
        "num_attention_heads": N_HEADS,
        "num_key_value_heads": N_KV_HEADS,
        "intermediate_size": INTERMEDIATE,
        "vocab_size": VOCAB,
        "rope_theta": 1_000_000.0,
        "rms_norm_eps": 1e-6,
        # aliases expected by repair_config
        "n_layer": N_LAYERS,
        "n_embed": HIDDEN,
    }
    (d / "config.json").write_text(json.dumps(config))

    def rand(*shape):
        return torch.randn(*shape, dtype=torch.float16) * 0.01

    weights = {
        "model.embed_tokens.weight": rand(VOCAB, HIDDEN),
        "model.norm.weight":         torch.ones(HIDDEN),
        "lm_head.weight":            rand(VOCAB, HIDDEN),
    }
    for i in range(N_LAYERS):
        p = f"model.layers.{i}"
        kv_dim = N_KV_HEADS * HEAD_DIM
        weights.update({
            f"{p}.input_layernorm.weight":            torch.ones(HIDDEN),
            f"{p}.post_attention_layernorm.weight":   torch.ones(HIDDEN),
            f"{p}.self_attn.q_proj.weight":           rand(HIDDEN, HIDDEN),
            f"{p}.self_attn.k_proj.weight":           rand(kv_dim, HIDDEN),
            f"{p}.self_attn.v_proj.weight":           rand(kv_dim, HIDDEN),
            f"{p}.self_attn.o_proj.weight":           rand(HIDDEN, HIDDEN),
            f"{p}.self_attn.q_norm.weight":           torch.ones(HEAD_DIM),
            f"{p}.self_attn.k_norm.weight":           torch.ones(HEAD_DIM),
            f"{p}.mlp.gate_proj.weight":              rand(INTERMEDIATE, HIDDEN),
            f"{p}.mlp.up_proj.weight":                rand(INTERMEDIATE, HIDDEN),
            f"{p}.mlp.down_proj.weight":              rand(HIDDEN, INTERMEDIATE),
        })

    save_file(weights, str(d / "model.safetensors"))
    return str(d)


@pytest.fixture(scope="module")
def adapter_dir(tmp_path_factory):
    """Write a tiny PEFT-format LoRA adapter to a temp dir."""
    from safetensors.torch import save_file

    d = tmp_path_factory.mktemp("lora_tiny")

    adapter_cfg = {
        "r": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    (d / "adapter_config.json").write_text(json.dumps(adapter_cfg))

    kv_dim = N_KV_HEADS * HEAD_DIM
    module_dims = {
        "q_proj": (HIDDEN, HIDDEN),
        "k_proj": (kv_dim, HIDDEN),
        "v_proj": (kv_dim, HIDDEN),
        "o_proj": (HIDDEN, HIDDEN),
    }
    scaling = LORA_ALPHA / LORA_RANK

    adapter_weights = {}
    for i in range(N_LAYERS):
        for module, (out_dim, in_dim) in module_dims.items():
            pfx = f"base_model.model.model.layers.{i}.self_attn.{module}"
            # lora_A: [rank, in_dim],  lora_B: [out_dim, rank]
            adapter_weights[f"{pfx}.lora_A.weight"] = (
                torch.randn(LORA_RANK, in_dim, dtype=torch.float16) * 0.01
            )
            adapter_weights[f"{pfx}.lora_B.weight"] = torch.zeros(
                out_dim, LORA_RANK, dtype=torch.float16
            )  # zeros → delta is 0, output equals base model

    save_file(adapter_weights, str(d / "adapter_model.safetensors"))
    return str(d)


@pytest.fixture(scope="module")
def model(weight_dir):
    """Instantiate QwenLoRAModel once for the entire test module."""
    from model.qwen_lora_model import QwenLoRAModel

    return QwenLoRAModel(
        weight_dir=weight_dir,
        max_total_token_num=512,
        mem_adapter_size=0,
        dummy=False,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_prefill_batch(token_seqs: list[list[int]], device="cuda"):
    """Build tensors for a prefill forward call from a list of token sequences."""
    batch_size = len(token_seqs)
    seq_lens = [len(s) for s in token_seqs]
    total = sum(seq_lens)

    flat = torch.tensor([t for s in token_seqs for t in s], dtype=torch.long, device=device)
    b_seq_len = torch.tensor(seq_lens, dtype=torch.long, device=device)
    b_start_loc = torch.zeros(batch_size, dtype=torch.long, device=device)
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], dim=0)
    max_len = int(b_seq_len.max().item())

    b_loc = torch.zeros(batch_size, max_len + 10, dtype=torch.long, device=device)

    return flat, b_loc, b_start_loc, b_seq_len, max_len, total


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestWeightLoading:
    def test_model_instantiates(self, model):
        assert model is not None

    def test_embed_tokens_on_cuda(self, model):
        assert model.pre_post_weight.embed_tokens_weight.device.type == "cuda"

    def test_transformer_weights_on_cuda(self, model):
        for w in model.trans_layers_weight:
            assert w.q_proj_weight.device.type == "cuda"

    def test_correct_layer_count(self, model):
        assert len(model.trans_layers_weight) == N_LAYERS
        assert model.layers_num == N_LAYERS

    def test_vocab_size(self, model):
        assert model.vocab_size == VOCAB


class TestAdapterManager:
    def test_load_adapter(self, model, adapter_dir):
        model.load_adapter("test-adapter", adapter_dir)
        assert model.adapter_manager.is_loaded("test-adapter")

    def test_adapter_weights_shape(self, model, adapter_dir):
        model.load_adapter("test-adapter", adapter_dir)  # idempotent
        A, B, scaling = model.adapter_manager.get_lora_weights("test-adapter", 0, "q_proj")
        assert A.shape == (LORA_RANK, HIDDEN)
        assert B.shape == (HIDDEN, LORA_RANK)
        assert abs(scaling - LORA_ALPHA / LORA_RANK) < 1e-5

    def test_adapter_id_to_int(self, model, adapter_dir):
        model.load_adapter("test-adapter", adapter_dir)
        idx = model.adapter_manager.adapter_id_to_int("test-adapter")
        assert isinstance(idx, int) and idx >= 0

    def test_no_adapter_returns_minus_one(self, model):
        assert model.adapter_manager.adapter_id_to_int(None) == -1
        assert model.adapter_manager.adapter_id_to_int("") == -1

    def test_unload_adapter(self, model, adapter_dir):
        model.load_adapter("temp-adapter", adapter_dir)
        assert model.adapter_manager.is_loaded("temp-adapter")
        model.unload_adapter("temp-adapter")
        assert not model.adapter_manager.is_loaded("temp-adapter")


class TestPrefill:
    def test_single_request_logits_shape(self, model):
        tokens = [[1, 2, 3, 4, 5]]
        flat, b_loc, b_start_loc, b_seq_len, max_len, total = _make_prefill_batch(tokens)
        logits = model.forward(
            batch_size=1, total_token_num=total, max_len_in_batch=max_len,
            input_ids=flat, b_loc=b_loc, b_start_loc=b_start_loc, b_seq_len=b_seq_len,
            adapter_ids=[None], is_prefill=True,
        )
        assert logits.shape == (1, VOCAB)

    def test_batch_logits_shape(self, model):
        tokens = [[1, 2, 3], [4, 5, 6, 7]]
        flat, b_loc, b_start_loc, b_seq_len, max_len, total = _make_prefill_batch(tokens)
        logits = model.forward(
            batch_size=2, total_token_num=total, max_len_in_batch=max_len,
            input_ids=flat, b_loc=b_loc, b_start_loc=b_start_loc, b_seq_len=b_seq_len,
            adapter_ids=[None, None], is_prefill=True,
        )
        assert logits.shape == (2, VOCAB)

    def test_logits_are_finite(self, model):
        tokens = [[10, 20, 30]]
        flat, b_loc, b_start_loc, b_seq_len, max_len, total = _make_prefill_batch(tokens)
        logits = model.forward(
            batch_size=1, total_token_num=total, max_len_in_batch=max_len,
            input_ids=flat, b_loc=b_loc, b_start_loc=b_start_loc, b_seq_len=b_seq_len,
            adapter_ids=[None], is_prefill=True,
        )
        assert torch.isfinite(logits).all()

    def test_kv_cache_slots_allocated(self, model):
        tokens = [[1, 2, 3, 4]]
        flat, b_loc, b_start_loc, b_seq_len, max_len, total = _make_prefill_batch(tokens)
        free_before = model.mem_manager.can_use_mem_size
        model.forward(
            batch_size=1, total_token_num=total, max_len_in_batch=max_len,
            input_ids=flat, b_loc=b_loc, b_start_loc=b_start_loc, b_seq_len=b_seq_len,
            adapter_ids=[None], is_prefill=True,
        )
        assert model.mem_manager.can_use_mem_size == free_before - 4
        # cleanup
        model.mem_manager.free(b_loc[0, :4])


class TestLoRAPrefill:
    def test_with_adapter_same_shape(self, model, adapter_dir):
        """LoRA-enabled forward must return the same shape as base."""
        model.load_adapter("test-adapter", adapter_dir)
        tokens = [[1, 2, 3, 4, 5]]
        flat, b_loc, b_start_loc, b_seq_len, max_len, total = _make_prefill_batch(tokens)
        logits = model.forward(
            batch_size=1, total_token_num=total, max_len_in_batch=max_len,
            input_ids=flat, b_loc=b_loc, b_start_loc=b_start_loc, b_seq_len=b_seq_len,
            adapter_ids=["test-adapter"], is_prefill=True,
        )
        assert logits.shape == (1, VOCAB)
        assert torch.isfinite(logits).all()
        model.mem_manager.free(b_loc[0, :5])

    def test_zero_lora_b_matches_base(self, model, adapter_dir):
        """With lora_B initialised to zero the LoRA delta is zero → identical to base."""
        model.load_adapter("test-adapter", adapter_dir)
        tokens = [[7, 8, 9]]

        flat, b_loc, b_start_loc, b_seq_len, max_len, total = _make_prefill_batch(tokens)
        logits_base = model.forward(
            batch_size=1, total_token_num=total, max_len_in_batch=max_len,
            input_ids=flat, b_loc=b_loc, b_start_loc=b_start_loc, b_seq_len=b_seq_len,
            adapter_ids=[None], is_prefill=True,
        )
        model.mem_manager.free(b_loc[0, :3])

        flat, b_loc, b_start_loc, b_seq_len, max_len, total = _make_prefill_batch(tokens)
        logits_lora = model.forward(
            batch_size=1, total_token_num=total, max_len_in_batch=max_len,
            input_ids=flat, b_loc=b_loc, b_start_loc=b_start_loc, b_seq_len=b_seq_len,
            adapter_ids=["test-adapter"], is_prefill=True,
        )
        model.mem_manager.free(b_loc[0, :3])

        assert torch.allclose(logits_base.float(), logits_lora.float(), atol=1e-3)

    def test_nonzero_lora_changes_output(self, model, adapter_dir):
        """A nonzero lora_B must change the output vs. the base model."""
        from safetensors.torch import save_file
        import tempfile, pathlib

        # Build a second adapter with nonzero lora_B
        d = pathlib.Path(tempfile.mkdtemp())
        kv_dim = N_KV_HEADS * HEAD_DIM
        module_dims = {
            "q_proj": (HIDDEN, HIDDEN), "k_proj": (kv_dim, HIDDEN),
            "v_proj": (kv_dim, HIDDEN), "o_proj": (HIDDEN, HIDDEN),
        }
        cfg = {"r": LORA_RANK, "lora_alpha": LORA_ALPHA,
               "target_modules": list(module_dims), "bias": "none", "task_type": "CAUSAL_LM"}
        (d / "adapter_config.json").write_text(json.dumps(cfg))

        weights = {}
        for i in range(N_LAYERS):
            for mod, (out_dim, in_dim) in module_dims.items():
                pfx = f"base_model.model.model.layers.{i}.self_attn.{mod}"
                weights[f"{pfx}.lora_A.weight"] = torch.randn(LORA_RANK, in_dim, dtype=torch.float16)
                weights[f"{pfx}.lora_B.weight"] = torch.randn(out_dim, LORA_RANK, dtype=torch.float16)
        save_file(weights, str(d / "adapter_model.safetensors"))

        model.load_adapter("nonzero-adapter", str(d))
        tokens = [[1, 2, 3]]

        flat, b_loc, b_start_loc, b_seq_len, max_len, total = _make_prefill_batch(tokens)
        logits_base = model.forward(
            batch_size=1, total_token_num=total, max_len_in_batch=max_len,
            input_ids=flat, b_loc=b_loc, b_start_loc=b_start_loc, b_seq_len=b_seq_len,
            adapter_ids=[None], is_prefill=True,
        )
        model.mem_manager.free(b_loc[0, :3])

        flat, b_loc, b_start_loc, b_seq_len, max_len, total = _make_prefill_batch(tokens)
        logits_lora = model.forward(
            batch_size=1, total_token_num=total, max_len_in_batch=max_len,
            input_ids=flat, b_loc=b_loc, b_start_loc=b_start_loc, b_seq_len=b_seq_len,
            adapter_ids=["nonzero-adapter"], is_prefill=True,
        )
        model.mem_manager.free(b_loc[0, :3])

        assert not torch.allclose(logits_base.float(), logits_lora.float(), atol=1e-4), \
            "nonzero LoRA should change the output"
        model.unload_adapter("nonzero-adapter")


class TestMixedAdapterBatch:
    def test_two_adapters_in_same_batch(self, model, adapter_dir):
        """Two requests with different adapters must each return [vocab] logits."""
        model.load_adapter("test-adapter", adapter_dir)
        tokens = [[1, 2, 3], [4, 5, 6, 7]]
        flat, b_loc, b_start_loc, b_seq_len, max_len, total = _make_prefill_batch(tokens)
        logits = model.forward(
            batch_size=2, total_token_num=total, max_len_in_batch=max_len,
            input_ids=flat, b_loc=b_loc, b_start_loc=b_start_loc, b_seq_len=b_seq_len,
            adapter_ids=["test-adapter", None], is_prefill=True,
        )
        assert logits.shape == (2, VOCAB)
        assert torch.isfinite(logits).all()
        for i, seq in enumerate(tokens):
            model.mem_manager.free(b_loc[i, :len(seq)])

    def test_mixed_adapter_outputs_differ(self, model, adapter_dir):
        """Same tokens, different adapters → different logits."""
        from safetensors.torch import save_file
        import tempfile, pathlib

        d = pathlib.Path(tempfile.mkdtemp())
        kv_dim = N_KV_HEADS * HEAD_DIM
        module_dims = {
            "q_proj": (HIDDEN, HIDDEN), "k_proj": (kv_dim, HIDDEN),
            "v_proj": (kv_dim, HIDDEN), "o_proj": (HIDDEN, HIDDEN),
        }
        cfg = {"r": LORA_RANK, "lora_alpha": LORA_ALPHA,
               "target_modules": list(module_dims), "bias": "none", "task_type": "CAUSAL_LM"}
        (d / "adapter_config.json").write_text(json.dumps(cfg))
        weights = {}
        for i in range(N_LAYERS):
            for mod, (out_dim, in_dim) in module_dims.items():
                pfx = f"base_model.model.model.layers.{i}.self_attn.{mod}"
                weights[f"{pfx}.lora_A.weight"] = torch.randn(LORA_RANK, in_dim, dtype=torch.float16)
                weights[f"{pfx}.lora_B.weight"] = torch.randn(out_dim, LORA_RANK, dtype=torch.float16)
        save_file(weights, str(d / "adapter_model.safetensors"))
        model.load_adapter("adapter-B", str(d))

        tokens = [[1, 2, 3], [1, 2, 3]]  # identical prompts
        flat, b_loc, b_start_loc, b_seq_len, max_len, total = _make_prefill_batch(tokens)
        logits = model.forward(
            batch_size=2, total_token_num=total, max_len_in_batch=max_len,
            input_ids=flat, b_loc=b_loc, b_start_loc=b_start_loc, b_seq_len=b_seq_len,
            adapter_ids=["test-adapter", "adapter-B"], is_prefill=True,
        )
        assert not torch.allclose(logits[0].float(), logits[1].float(), atol=1e-4), \
            "different adapters on same input should produce different logits"
        for i in range(2):
            model.mem_manager.free(b_loc[i, :3])
        model.unload_adapter("adapter-B")


class TestDecode:
    def test_decode_step_shape(self, model):
        """After prefill, one decode step should return [batch, vocab] logits."""
        tokens = [[1, 2, 3, 4]]
        flat, b_loc, b_start_loc, b_seq_len, max_len, total = _make_prefill_batch(tokens)
        model.forward(
            batch_size=1, total_token_num=total, max_len_in_batch=max_len,
            input_ids=flat, b_loc=b_loc, b_start_loc=b_start_loc, b_seq_len=b_seq_len,
            adapter_ids=[None], is_prefill=True,
        )

        # Decode step: extend b_loc and b_seq_len
        b_seq_len += 1
        new_max = int(b_seq_len.max().item())
        next_token = torch.tensor([42], dtype=torch.long, device="cuda")
        b_start_loc_dec = torch.zeros(1, dtype=torch.long, device="cuda")

        logits = model.forward(
            batch_size=1, total_token_num=1, max_len_in_batch=new_max,
            input_ids=next_token, b_loc=b_loc, b_start_loc=b_start_loc_dec,
            b_seq_len=b_seq_len, adapter_ids=[None], is_prefill=False,
        )
        assert logits.shape == (1, VOCAB)
        assert torch.isfinite(logits).all()
        # cleanup: free all 5 slots (4 prefill + 1 decode)
        model.mem_manager.free(b_loc[0, :5])

    def test_multiple_decode_steps(self, model):
        """Run 3 decode steps and verify each returns valid logits."""
        tokens = [[5, 6, 7]]
        flat, b_loc, b_start_loc, b_seq_len, max_len, total = _make_prefill_batch(tokens)
        logits = model.forward(
            batch_size=1, total_token_num=total, max_len_in_batch=max_len,
            input_ids=flat, b_loc=b_loc, b_start_loc=b_start_loc, b_seq_len=b_seq_len,
            adapter_ids=[None], is_prefill=True,
        )
        next_tok = logits.argmax(dim=-1)

        for step in range(3):
            b_seq_len += 1
            new_max = int(b_seq_len.max().item())
            b_start_loc_dec = torch.zeros(1, dtype=torch.long, device="cuda")
            logits = model.forward(
                batch_size=1, total_token_num=1, max_len_in_batch=new_max,
                input_ids=next_tok, b_loc=b_loc, b_start_loc=b_start_loc_dec,
                b_seq_len=b_seq_len, adapter_ids=[None], is_prefill=False,
            )
            assert logits.shape == (1, VOCAB), f"step {step}: wrong logits shape"
            assert torch.isfinite(logits).all(), f"step {step}: non-finite logits"
            next_tok = logits.argmax(dim=-1)

        model.mem_manager.free(b_loc[0, :6])  # 3 prefill + 3 decode
