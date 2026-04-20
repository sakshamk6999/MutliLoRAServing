import os
import json
import torch
from safetensors.torch import load_file as safetensors_load_file


def get_config_json(weight_dir: str) -> dict:
    config_path = os.path.join(weight_dir, "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def hf_load_config(weight_dir: str, mode: str = "model"):
    """Resolve local or HF hub path and return (config_dict, resolved_weight_dir)."""
    if os.path.isdir(weight_dir):
        config = get_config_json(weight_dir)
        return config, weight_dir
    # Try HF cache / snapshot download
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(repo_id=weight_dir)
    config = get_config_json(local_dir)
    return config, local_dir


def repair_config(config: dict, same_names: list):
    """Copy the first present key's value to all other aliases in same_names."""
    val = None
    for name in same_names:
        if name in config:
            val = config[name]
            break
    if val is not None:
        for name in same_names:
            config[name] = val


def init_bloc(b_loc: torch.Tensor, b_seq_len: torch.Tensor,
              max_len_in_batch: int, prefill_mem_index: torch.Tensor):
    """Fill b_loc[i, :seq_len_i] with the allocated KV cache slot indices for request i.

    prefill_mem_index is a flat tensor of all allocated slots packed in request order.
    """
    start = 0
    for i in range(b_seq_len.shape[0]):
        seq_len = b_seq_len[i].item()
        b_loc[i, :seq_len] = prefill_mem_index[start: start + seq_len]
        start += seq_len


def load_hf_weights(dtype: str, weight_dir: str, pre_post_layer,
                    transformer_layer_list: list, dummy: bool = False):
    """Load HuggingFace safetensors weights and dispatch to layer weight objects."""
    if dummy:
        pre_post_layer.verify_load()
        for layer in transformer_layer_list:
            layer.verify_load()
        return

    index_path = os.path.join(weight_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        # Group by shard file
        shard_to_keys: dict[str, list[str]] = {}
        for key, shard in weight_map.items():
            shard_to_keys.setdefault(shard, []).append(key)
        # Load each shard once
        for shard, keys in shard_to_keys.items():
            shard_path = os.path.join(weight_dir, shard)
            tensors = safetensors_load_file(shard_path)
            for key in keys:
                tensor = tensors[key]
                _dispatch_weight(key, tensor, pre_post_layer, transformer_layer_list)
    else:
        # Single shard
        shard_path = os.path.join(weight_dir, "model.safetensors")
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"No safetensors weights found in {weight_dir}")
        tensors = safetensors_load_file(shard_path)
        for key, tensor in tensors.items():
            _dispatch_weight(key, tensor, pre_post_layer, transformer_layer_list)

    pre_post_layer.verify_load()
    for layer in transformer_layer_list:
        layer.verify_load()


def _dispatch_weight(key: str, tensor: torch.Tensor, pre_post_layer,
                     transformer_layer_list: list):
    """Route a weight tensor to the appropriate layer weight object."""
    # Transformer layer weights: model.layers.{idx}.*
    if key.startswith("model.layers."):
        parts = key.split(".")
        layer_idx = int(parts[2])
        if layer_idx < len(transformer_layer_list):
            transformer_layer_list[layer_idx].load_weight(key, tensor)
    else:
        pre_post_layer.load_weight(key, tensor)
