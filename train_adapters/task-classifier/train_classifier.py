"""
Training script for the task classifier.

Loads the preprocessed DatasetDict (produced by preprocess_classifier_data.py),
fine-tunes ClassifierModel, and saves the best checkpoint.

Usage:
    python train_classifier.py \
        --data_dir        ./classifier_data \
        --encoder_model   bert-base-uncased \
        --emb_dim         768 \
        --save_path       ./classifier_checkpoint \
        --learning_rate   2e-5 \
        --epochs          10 \
        --batch_size      32 \
        --max_length      128
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_from_disk

from classifier_model import ClassifierModel


# ── Dataset wrapper ──────────────────────────────────────────────────────────

class ClassifierDataset(Dataset):
    def __init__(self, hf_split, tokenizer, max_length):
        self.texts = hf_split["text"]
        self.labels = hf_split["label"]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Training helpers ─────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


# ── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train task classifier")
    parser.add_argument("--data_dir", type=str, default="./classifier_data",
                        help="Path to preprocessed DatasetDict (from preprocess_classifier_data.py)")
    parser.add_argument("--encoder_model", type=str, default="bert-base-uncased")
    parser.add_argument("--emb_dim", type=int, default=768,
                        help="Hidden size of the encoder (must match encoder_model)")
    parser.add_argument("--save_path", type=str, default="./classifier_checkpoint",
                        help="Directory to save the best model checkpoint")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (epochs without val improvement)")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset and label map
    ds = load_from_disk(args.data_dir)
    label_map_path = os.path.join(args.data_dir, "label_map.json")
    with open(label_map_path) as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    print(f"Classes ({num_classes}): {label_map}")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model)

    train_dataset = ClassifierDataset(ds["train"], tokenizer, args.max_length)
    val_dataset   = ClassifierDataset(ds["val"],   tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = ClassifierModel(
        encoder_model=args.encoder_model,
        emb_dim=args.emb_dim,
        classification_labels=num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_path, exist_ok=True)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "label_map": label_map,
                "args": vars(args),
            }
            torch.save(checkpoint, os.path.join(args.save_path, "best_model.pt"))
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  -> No improvement ({epochs_without_improvement}/{args.patience})")
            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.")
                break

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    print(f"Checkpoint saved to: {args.save_path}/best_model.pt")


if __name__ == "__main__":
    main()
