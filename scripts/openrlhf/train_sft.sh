#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# Prefer config-driven run; fall back to explicit CLI if your OpenRLHF build lacks --config.
if python3 -m openrlhf.cli.train_sft --help 2>&1 | grep -q -- '--config'; then
  python3 -m openrlhf.cli.train_sft --config configs/openrlhf/sft.yaml
else
  python3 -m openrlhf.cli.train_sft \
    --pretrain "${PRETRAIN:-Qwen/Qwen3-4B-Thinking-2507}" \
    --dataset data/sft.jsonl \
    --input_key messages \
    --apply_chat_template \
    --max_epochs "${MAX_EPOCHS:-1}" \
    --learning_rate "${LR:-2e-5}" \
    --batch_size "${BS:-1}" \
    --micro_train_batch_size "${MBS:-1}" \
    --max_len "${MAX_LEN:-8192}" \
    --output_dir "${OUTPUT_DIR:-checkpoints/aide_search_sft}"
fi
