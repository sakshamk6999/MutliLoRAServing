"""
Evaluation script for the task classifier.

Loads a saved checkpoint, runs inference on a dataset split, and reports
accuracy, per-class precision / recall / F1, and a confusion matrix.

Usage:
    python eval_classifier.py \
        --checkpoint_path ./classifier_checkpoint/best_model.pt \
        --data_dir        ./classifier_data \
        --split           test \
        --batch_size      64
"""

import argparse
import json
import os
from collections import defaultdict

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from classifier_model import ClassifierModel


# ── Dataset ──────────────────────────────────────────────────────────────────

class ClassifierDataset(Dataset):
    def __init__(self, hf_split, tokenizer, max_length):
        self.texts  = hf_split["text"]
        self.labels = hf_split["label"]
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Metrics ───────────────────────────────────────────────────────────────────

def classification_report(all_labels, all_preds, idx_to_task):
    num_classes = len(idx_to_task)
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for true, pred in zip(all_labels, all_preds):
        if true == pred:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1

    print(f"\n{'Task':<35} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 80)

    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    for idx in range(num_classes):
        task   = idx_to_task[idx]
        prec   = tp[idx] / (tp[idx] + fp[idx]) if (tp[idx] + fp[idx]) > 0 else 0.0
        rec    = tp[idx] / (tp[idx] + fn[idx]) if (tp[idx] + fn[idx]) > 0 else 0.0
        f1     = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        support = tp[idx] + fn[idx]
        print(f"{task:<35} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {support:>10}")
        macro_p  += prec
        macro_r  += rec
        macro_f1 += f1

    n = num_classes
    print("-" * 80)
    print(f"{'macro avg':<35} {macro_p/n:>10.4f} {macro_r/n:>10.4f} {macro_f1/n:>10.4f} {len(all_labels):>10}")


def confusion_matrix(all_labels, all_preds, idx_to_task):
    n = len(idx_to_task)
    matrix = [[0] * n for _ in range(n)]
    for true, pred in zip(all_labels, all_preds):
        matrix[true][pred] += 1

    tasks = [idx_to_task[i] for i in range(n)]
    col_w = max(len(t) for t in tasks) + 2

    header = f"{'':>{col_w}}" + "".join(f"{t[:12]:>14}" for t in tasks)
    print(f"\nConfusion Matrix (rows=true, cols=predicted):\n{header}")
    print("-" * len(header))
    for i, row in enumerate(matrix):
        row_str = f"{tasks[i]:>{col_w}}" + "".join(f"{v:>14}" for v in row)
        print(row_str)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate task classifier")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to best_model.pt saved by train_classifier.py")
    parser.add_argument("--data_dir", type=str, default="./classifier_data",
                        help="Path to preprocessed DatasetDict")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_file", type=str, default=None,
                        help="Optional JSON file to save per-example predictions")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load checkpoint ──────────────────────────────────────────────────────
    checkpoint     = torch.load(args.checkpoint_path, map_location=device)
    label_map      = checkpoint["label_map"]          # {task_name: int_idx}
    training_args  = checkpoint["args"]
    encoder_model  = training_args["encoder_model"]
    max_length     = training_args.get("max_length", 128)
    num_classes    = len(label_map)

    idx_to_task = {v: k for k, v in label_map.items()}  # {int_idx: task_name}
    print(f"Classes ({num_classes}): {label_map}")
    print(f"Checkpoint epoch={checkpoint['epoch']}  val_acc={checkpoint['val_acc']:.4f}")

    # ── Model & tokenizer ────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(encoder_model)
    model = ClassifierModel(
        encoder_model=encoder_model,
        emb_dim=training_args.get("emb_dim", 768),
        classification_labels=num_classes,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    # ── Dataset ──────────────────────────────────────────────────────────────
    ds      = load_from_disk(args.data_dir)
    split   = ds[args.split]
    dataset = ClassifierDataset(split, tokenizer, max_length)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=args.num_workers)
    print(f"\nEvaluating on '{args.split}' split: {len(dataset)} examples")

    # ── Inference ────────────────────────────────────────────────────────────
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].tolist()

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds  = logits.argmax(dim=-1).tolist()

            all_preds.extend(preds)
            all_labels.extend(labels)

    # ── Results ──────────────────────────────────────────────────────────────
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"\nOverall accuracy: {accuracy:.4f}  ({sum(p==l for p,l in zip(all_preds,all_labels))}/{len(all_labels)})")

    classification_report(all_labels, all_preds, idx_to_task)
    confusion_matrix(all_labels, all_preds, idx_to_task)

    # ── Optional output ───────────────────────────────────────────────────────
    if args.output_file:
        texts  = split["text"]
        records = [
            {
                "text":       texts[i],
                "true_label": idx_to_task[all_labels[i]],
                "pred_label": idx_to_task[all_preds[i]],
                "correct":    all_labels[i] == all_preds[i],
            }
            for i in range(len(all_labels))
        ]
        with open(args.output_file, "w") as f:
            json.dump({"accuracy": accuracy, "predictions": records}, f, indent=2)
        print(f"\nPredictions saved to {args.output_file}")


if __name__ == "__main__":
    main()
