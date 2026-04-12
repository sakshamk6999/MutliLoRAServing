from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from datasets import load_dataset

ds = load_dataset("8F-ai/EvalLLM-30000X")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
tokenized_data = tokenizer([d['output'] for d in ds['train']], padding='longest', return_tensors='pt')

print("tokenized_shape", tokenized_data['input_ids'].shape)