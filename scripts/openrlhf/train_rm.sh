#!/usr/bin/env bash
set -euo pipefail

python3 -m openrlhf.cli.train_rm \
  --config configs/openrlhf/rm.yaml

