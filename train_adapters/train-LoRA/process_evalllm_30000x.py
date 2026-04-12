from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from datasets import load_dataset
import argparse

def main(save_dir:str):
    ds = load_dataset("8F-ai/EvalLLM-30000X")

    unique_tasks = ds.unique("task_family")['train']

    for task in unique_tasks:
        print(f"task is {task}")
        filtered_dataset = ds.filter(lambda x: x['task_family'] == task)
        split_dataset = filtered_dataset['train'].train_test_split(test_size=0.2)
        print("train_size:", len(split_dataset['train']), "test_size:", len(split_dataset['test']))

        split_dataset.save_to_disk(f"{save_dir}/{task}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process dataset")
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()
    main(args.save_dir)