import argparse
import sys
import os

import torch
import zmq
from transformers import AutoTokenizer

# _CLASSIFIER_DIR = os.path.join(os.path.dirname(__file__), "train_adapters", "task-classifier")
# sys.path.insert(0, os.path.abspath(_CLASSIFIER_DIR))
from .classifier_model import ClassifierModel
from data_model.data_types import TaggedRequest

PULL_ADDR = "tcp://localhost:5555"
PUSH_ADDR = "tcp://*:5556"
MAX_LENGTH = 128

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    encoder_model = checkpoint["args"]["encoder_model"]
    num_classes   = len(checkpoint["label_map"])
    label_map     = checkpoint["label_map"]  # {str_index: task_name}

    model = ClassifierModel(
        encoder_model=encoder_model,
        classification_labels=num_classes,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(encoder_model)

    return model, tokenizer, label_map


def tokenize(tokenizer, prompt: str) -> dict:
    encoding = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )
    # Only pass keys the model expects
    return {
        "input_ids":      encoding["input_ids"].to(device),
        "attention_mask": encoding["attention_mask"].to(device),
    }


def classify_task(model, tokenizer, label_map: dict, prompt: str) -> str:
    inputs = tokenize(tokenizer, prompt)
    with torch.no_grad():
        logits = model(**inputs)  # (1, num_classes)
    pred_idx = logits.argmax(dim=-1).item()
    return label_map[str(pred_idx)]


def run(checkpoint_path: str):
    model, tokenizer, label_map = load_model(checkpoint_path)
    print(f"[MLP] model loaded from {checkpoint_path}")

    ctx = zmq.Context()

    pull = ctx.socket(zmq.PULL)
    pull.connect(PULL_ADDR)

    push = ctx.socket(zmq.PUSH)
    push.bind(PUSH_ADDR)

    print(f"[MLP] ready — pulling from {PULL_ADDR}, pushing to {PUSH_ADDR}")

    while True:
        raw = pull.recv_json()
        task_type = classify_task(model, tokenizer, label_map, raw["prompt"])
        tagged = TaggedRequest(task_type=task_type, **raw)
        push.send_json(tagged.model_dump())
        print(f"[MLP] {tagged.request_id} → {task_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLP task classifier service")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="train_adapters/task-classifier/classifier_checkpoint/best_model.pt",
    )
    args = parser.parse_args()
    run(args.checkpoint_path)
