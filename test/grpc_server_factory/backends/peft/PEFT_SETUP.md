# PEFT setup (gRPC `CausalLMPEFTBackend`)

This backend **loads pre-trained Hugging Face PEFT adapters from disk**. It does **not** create new LoRA adapters in code and does **not** train.

## Intended flow

1. **One base causal LM** is loaded at startup (`AutoModelForCausalLM.from_pretrained` for your base model id/path).

2. **Adapters are registered** with gRPC **`LoadAdapter`**: `adapter_id` + `adapter_path` pointing to a real PEFT export (checkpoint directory with adapter weights and config).  
   - First adapter: `PeftModel.from_pretrained(base_model, adapter_path, adapter_name=adapter_id)`.  
   - Further adapters: `model.load_adapter(adapter_path, adapter_name=adapter_id)`.

3. **Inference** uses **`set_adapter(adapter_id)`** before forwards for that row. Rank, alpha, and target modules come from **each checkpoint** (e.g. `adapter_config.json`), not from parsing strings or from ad-hoc `LoraConfig(r=…)` without weights.

4. **`PrefillRequest`** (see `model_logic/protos/model_service.proto`) carries **`adapter_ids`** parallel to `request_ids` / `prompts` / `max_tokens`. Each index picks which **already-loaded** `adapter_id` to use. The router typically sends values aligned with classifier **`task_type`**; those strings must match the **`adapter_id`** values used in **`LoadAdapter`**.

## What this is *not*

- **Not** `get_peft_model(base, LoraConfig(r=rank_from_filename))` without loading a checkpoint — that would add an empty/untrained LoRA shell and is **not** used here.

- **Not** a rank sweep or synthetic adapter naming convention; you assume **correct** checkpoints and matching ids.

## Operational checklist

- Base model id/path matches what adapters were trained against.

- Each `LoadAdapter(adapter_id, adapter_path)` uses a valid PEFT directory.

- Every `adapter_id` appearing in **`Prefill.adapter_ids`** has been loaded (or prefill returns an error).

## Hardware note

The backend uses a **single** device string (`cuda` / `cpu`) and `.to(device)` — no built-in multi-GPU `device_map` / `max_memory` sharding in this module. Extend here if you need multi-GPU loading later.

## Related files

- Implementation: `causal_lm_backend.py`
- Metrics: `metrics.py`
- Proto: `model_logic/protos/model_service.proto`
