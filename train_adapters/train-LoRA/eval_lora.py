"""
Evaluation script for a fine-tuned LoRA adapter.

Loads a base model + saved LoRA adapter, runs greedy generation on a dataset
split, and reports ROUGE-1/2/L and exact-match scores.

Usage:
    python eval_lora.py \
        --base_model        Qwen/Qwen2.5-7B-Instruct \
        --adapter_path      ./adapters/diagnosis \
        --dataset_path      ./processed_evalllm_datasets/diagnosis \
        --instruction_column instruction \
        --label_column       output \
        --split             test \
        --max_new_tokens    256 \
        --batch_size        4
"""

import argparse
import json
import os

import torch
from datasets import load_from_disk
from dotenv import load_dotenv
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
_hf_token = os.getenv("HUGGING_FACE_LOGIN_TOKEN")
if _hf_token:
    login(token=_hf_token)


# ── ROUGE (no extra dependency — computed from scratch) ───────────────────────

def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _rouge_n(pred_tokens, ref_tokens, n):
    pred_ngrams = _ngrams(pred_tokens, n)
    ref_ngrams  = _ngrams(ref_tokens,  n)
    if not ref_ngrams:
        return 0.0
    ref_set  = {}
    for ng in ref_ngrams:
        ref_set[ng] = ref_set.get(ng, 0) + 1
    matches = 0
    for ng in pred_ngrams:
        if ref_set.get(ng, 0) > 0:
            matches += 1
            ref_set[ng] -= 1
    recall    = matches / len(ref_ngrams)
    precision = matches / len(pred_ngrams) if pred_ngrams else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)   # F1


def _lcs_len(a, b):
    """Length of the longest common subsequence."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Use two-row DP to save memory
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def _rouge_l(pred_tokens, ref_tokens):
    if not ref_tokens or not pred_tokens:
        return 0.0
    lcs      = _lcs_len(pred_tokens, ref_tokens)
    recall    = lcs / len(ref_tokens)
    precision = lcs / len(pred_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_rouge(prediction: str, reference: str):
    pred_tok = prediction.lower().split()
    ref_tok  = reference.lower().split()
    return {
        "rouge1": _rouge_n(pred_tok, ref_tok, 1),
        "rouge2": _rouge_n(pred_tok, ref_tok, 2),
        "rougeL": _rouge_l(pred_tok, ref_tok),
    }


# ── Generation ────────────────────────────────────────────────────────────────

def generate_batch(model, tokenizer, instructions, max_new_tokens, device):
    """Tokenise a batch of instructions and run greedy generation."""
    encodings = tokenizer(
        instructions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **encodings,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Strip the prompt tokens — keep only generated tokens
    prompt_lens = encodings["input_ids"].shape[1]
    generated   = output_ids[:, prompt_lens:]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a LoRA adapter")
    parser.add_argument("--base_model",          type=str, required=True,
                        help="HuggingFace base model name or local path")
    parser.add_argument("--adapter_path",        type=str, required=True,
                        help="Path to saved LoRA adapter (output of train_lora.py)")
    parser.add_argument("--dataset_path",        type=str, required=True,
                        help="Path to processed dataset directory (load_from_disk)")
    parser.add_argument("--instruction_column",  type=str, required=True)
    parser.add_argument("--label_column",        type=str, required=True)
    parser.add_argument("--split",               type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--max_new_tokens",      type=int, default=256)
    parser.add_argument("--batch_size",          type=int, default=4)
    parser.add_argument("--max_examples",        type=int, default=0,
                        help="Cap number of examples (0 = use all)")
    parser.add_argument("--output_file",         type=str, default=None,
                        help="Optional JSON file to save per-example results")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Model ────────────────────────────────────────────────────────────────
    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # required for batch generation

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()
    model.to(device)
    print(f"Adapter loaded from: {args.adapter_path}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    ds    = load_from_disk(args.dataset_path)
    split = ds[args.split]
    if args.max_examples > 0:
        split = split.select(range(min(args.max_examples, len(split))))
    print(f"Evaluating on '{args.split}' split: {len(split)} examples")

    instructions = split[args.instruction_column]
    references   = split[args.label_column]

    # ── Inference ─────────────────────────────────────────────────────────────
    predictions = []
    for start in range(0, len(instructions), args.batch_size):
        batch_instr = instructions[start : start + args.batch_size]
        batch_preds = generate_batch(model, tokenizer, batch_instr, args.max_new_tokens, device)
        predictions.extend(batch_preds)
        done = min(start + args.batch_size, len(instructions))
        print(f"  {done}/{len(instructions)}", end="\r")
    print()

    # ── Metrics ───────────────────────────────────────────────────────────────
    rouge1_scores, rouge2_scores, rougeL_scores, exact_matches = [], [], [], []
    records = []

    for pred, ref, instr in zip(predictions, references, instructions):
        scores = compute_rouge(pred, ref)
        em     = int(pred.strip() == ref.strip())

        rouge1_scores.append(scores["rouge1"])
        rouge2_scores.append(scores["rouge2"])
        rougeL_scores.append(scores["rougeL"])
        exact_matches.append(em)

        records.append({
            "instruction": instr,
            "reference":   ref,
            "prediction":  pred,
            "rouge1":      round(scores["rouge1"], 4),
            "rouge2":      round(scores["rouge2"], 4),
            "rougeL":      round(scores["rougeL"], 4),
            "exact_match": em,
        })

    n = len(records)
    avg_r1 = sum(rouge1_scores) / n
    avg_r2 = sum(rouge2_scores) / n
    avg_rl = sum(rougeL_scores) / n
    avg_em = sum(exact_matches) / n

    print(f"\n{'Metric':<15} {'Score':>8}")
    print("-" * 25)
    print(f"{'ROUGE-1':<15} {avg_r1:>8.4f}")
    print(f"{'ROUGE-2':<15} {avg_r2:>8.4f}")
    print(f"{'ROUGE-L':<15} {avg_rl:>8.4f}")
    print(f"{'Exact Match':<15} {avg_em:>8.4f}")
    print(f"\nTotal examples evaluated: {n}")

    # ── Sample predictions ────────────────────────────────────────────────────
    print("\n── Sample predictions (first 3) ─────────────────────────────────────")
    for rec in records[:3]:
        print(f"\nInstruction : {rec['instruction'][:120]}")
        print(f"Reference   : {rec['reference'][:120]}")
        print(f"Prediction  : {rec['prediction'][:120]}")
        print(f"ROUGE-L={rec['rougeL']:.4f}  EM={rec['exact_match']}")

    # ── Optional output ───────────────────────────────────────────────────────
    if args.output_file:
        summary = {
            "adapter_path": args.adapter_path,
            "split":        args.split,
            "num_examples": n,
            "metrics": {
                "rouge1":      round(avg_r1, 4),
                "rouge2":      round(avg_r2, 4),
                "rougeL":      round(avg_rl, 4),
                "exact_match": round(avg_em, 4),
            },
            "predictions": records,
        }
        with open(args.output_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
