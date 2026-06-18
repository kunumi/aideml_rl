#!/usr/bin/env bash
set -euo pipefail

python3 -m openrlhf.cli.train_ppo_ray \
  --config configs/openrlhf/ppo_multiturn.yaml

