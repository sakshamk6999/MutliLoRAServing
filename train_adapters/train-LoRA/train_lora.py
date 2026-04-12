import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from dotenv import load_dotenv
from huggingface_hub import login
import os

load_dotenv()

login(token=os.getenv("HUGGING_FACE_LOGIN_TOKEN"))

def parse_args():
    parser = argparse.ArgumentParser(description="Supervised fine-tuning with LoRA adapters")
    parser.add_argument("--dataset_name", type=str, required=True, help="HuggingFace dataset name")
    parser.add_argument("--base_model", type=str, required=True, help="HuggingFace base model name or path")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--adapter_save_path", type=str, required=True, help="Path to save the trained adapter")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=512, help="Max token sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device training batch size")
    parser.add_argument("--instruction_column", type=str, required=True, help="Dataset column containing the instruction/prompt")
    parser.add_argument("--label_column", type=str, required=True, help="Dataset column containing the target response")
    return parser.parse_args()


def tokenize(examples, tokenizer, instruction_column, label_column, max_length):
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for instruction, label in zip(examples[instruction_column], examples[label_column]):
        instruction_ids = tokenizer.encode(instruction, add_special_tokens=True)
        label_ids = tokenizer.encode(label, add_special_tokens=False)
        # Append EOS after the label
        label_ids = label_ids + [tokenizer.eos_token_id]

        input_ids = instruction_ids + label_ids

        # Truncate to max_length
        input_ids = input_ids[:max_length]

        # Mask instruction tokens in labels with -100 so loss is only on label tokens
        instruction_len = min(len(instruction_ids), max_length)
        labels = [-100] * instruction_len + input_ids[instruction_len:]

        attention_mask = [1] * len(input_ids)

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
    }


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True)
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset(args.dataset_name)

    tokenize_fn = lambda examples: tokenize(
        examples, tokenizer, args.instruction_column, args.label_column, args.max_length
    )

    train_dataset = dataset["train"].map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names)
    eval_dataset = dataset["val"].map(tokenize_fn, batched=True, remove_columns=dataset["val"].column_names)

    print("train dataset length", len(train_dataset))
    print("eval dataset length", len(eval_dataset))

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8)

    training_args = TrainingArguments(
        output_dir=args.adapter_save_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    model.save_pretrained(args.adapter_save_path)
    tokenizer.save_pretrained(args.adapter_save_path)
    print(f"Adapter saved to {args.adapter_save_path}")


if __name__ == "__main__":
    main()
