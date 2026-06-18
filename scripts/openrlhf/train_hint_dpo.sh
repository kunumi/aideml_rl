#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

if python3 -m openrlhf.cli.train_dpo --help 2>&1 | grep -q -- '--config'; then
  python3 -m openrlhf.cli.train_dpo --config configs/openrlhf/hint_dpo.yaml
else
  python3 -m openrlhf.cli.train_dpo \
    --pretrain "${PRETRAIN:-checkpoints/aide_hint_controller_sft}" \
    --dataset data/heuristic_runs/hint_controller/preferences.jsonl \
    --prompt_key prompt \
    --chosen_key chosen \
    --rejected_key rejected \
    --apply_chat_template \
    --max_epochs "${MAX_EPOCHS:-1}" \
    --learning_rate "${LR:-5e-7}" \
    --batch_size "${BS:-1}" \
    --micro_train_batch_size "${MBS:-1}" \
    --max_len "${MAX_LEN:-8192}" \
    --beta "${BETA:-0.1}" \
    --output_dir "${OUTPUT_DIR:-checkpoints/aide_hint_controller_dpo}"
fi
