#!/bin/bash
set -euo pipefail

source ./train_adapter.cfg

IFS=: read -ra DATASET_LIST <<< "$DATA_SETS"

echo "=============================="
echo "LoRA Adapter Training Pipeline"
echo "=============================="
echo "Base model   : $BASE_MODEL"
echo "Datasets     : ${DATASET_LIST[*]}"
echo "LoRA rank    : $LORA_RANK  |  alpha: $LORA_ALPHA"
echo "LR           : $LEARNING_RATE  |  epochs: $EPOCHS"
echo "=============================="

for dataset in "${DATASET_LIST[@]}"; do
    # Derive a filesystem-safe name from the dataset (replace / and - with _)
    dataset_slug=$(echo "$dataset" | tr '/:-' '_')
    adapter_path="${ADAPTER_SAVE_ROOT}/${dataset_slug}"

    echo ""
    echo ">>> Training adapter for dataset: $dataset"
    echo "    Saving to: $adapter_path"

    python train_lora.py \
        --dataset_name        "$dataset" \
        --base_model          "$BASE_MODEL" \
        --lora_rank           "$LORA_RANK" \
        --lora_alpha          "$LORA_ALPHA" \
        --adapter_save_path   "$adapter_path" \
        --learning_rate       "$LEARNING_RATE" \
        --epochs              "$EPOCHS" \
        --batch_size          "$BATCH_SIZE" \
        --max_length          "$MAX_LENGTH" \
        --instruction_column  "$INSTRUCTION_COLUMN" \
        --label_column        "$LABEL_COLUMN"

    echo "<<< Done: $dataset"
done

echo ""
echo "All adapters trained successfully."
