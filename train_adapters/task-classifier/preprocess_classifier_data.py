"""
Preprocessing script for task classifier.

Loads each task dataset from processed_evalllm_datasets/, samples up to
--samples_per_task examples per split, attaches an integer label and the
task name, then saves the combined dataset to --output_dir.

Usage:
    python preprocess_classifier_data.py \
        --datasets_root ../train-LoRA/processed_evalllm_datasets \
        --output_dir    ./classifier_data \
        --samples_per_task 2000 \
        --seed 42
"""

import argparse
import json
import os
import random

from datasets import DatasetDict, concatenate_datasets, load_from_disk


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare classifier dataset from task-specific datasets")
    parser.add_argument(
        "--datasets_root",
        type=str,
        default="../train-LoRA/processed_evalllm_datasets",
        help="Root directory containing one sub-folder per task",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./classifier_data",
        help="Where to save the combined classifier dataset",
    )
    parser.add_argument(
        "--samples_per_task",
        type=int,
        default=2000,
        help="Max examples to sample per task per split (0 = use all)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--instruction_column",
        type=str,
        default="instruction",
        help="Column to use as classifier input text",
    )
    return parser.parse_args()


def sample_split(dataset_split, n, seed):
    if n > 0 and len(dataset_split) > n:
        return dataset_split.shuffle(seed=seed).select(range(n))
    return dataset_split


def main():
    args = parse_args()
    random.seed(args.seed)

    task_dirs = sorted(
        d for d in os.listdir(args.datasets_root)
        if os.path.isdir(os.path.join(args.datasets_root, d))
    )

    if not task_dirs:
        raise ValueError(f"No task sub-directories found in {args.datasets_root}")

    label_map = {task: idx for idx, task in enumerate(task_dirs)}
    print(f"Found {len(task_dirs)} tasks: {task_dirs}")
    print(f"Label map: {label_map}")

    split_buckets: dict[str, list] = {"train": [], "val": [], "test": []}

    for task_name, label_id in label_map.items():
        task_path = os.path.join(args.datasets_root, task_name)
        ds = load_from_disk(task_path)

        for split_name in split_buckets:
            if split_name not in ds:
                print(f"  [warn] split '{split_name}' missing in {task_name}, skipping")
                continue

            split = ds[split_name]
            split = sample_split(split, args.samples_per_task, args.seed)

            # Keep only the instruction column, then add label columns
            split = split.select_columns([args.instruction_column])
            split = split.map(
                lambda example: {
                    "text": example[args.instruction_column],
                    "label": label_id,
                    "task_name": task_name,
                },
                remove_columns=[args.instruction_column],
            )

            split_buckets[split_name].append(split)
            print(f"  {task_name}/{split_name}: {len(split)} examples  (label={label_id})")

    # Concatenate all tasks per split and shuffle
    combined_splits = {}
    for split_name, parts in split_buckets.items():
        if not parts:
            continue
        combined = concatenate_datasets(parts).shuffle(seed=args.seed)
        combined_splits[split_name] = combined
        print(f"\n{split_name} total: {len(combined)} examples")

    combined_ds = DatasetDict(combined_splits)

    os.makedirs(args.output_dir, exist_ok=True)
    combined_ds.save_to_disk(args.output_dir)

    # Save label map alongside the dataset for later use
    label_map_path = os.path.join(args.output_dir, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\nDataset saved to: {args.output_dir}")
    print(f"Label map saved to: {label_map_path}")


if __name__ == "__main__":
    main()
