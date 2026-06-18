#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

if python3 -m openrlhf.cli.train_ppo_ray --help 2>&1 | grep -q -- '--config'; then
  python3 -m openrlhf.cli.train_ppo_ray --config configs/openrlhf/grpo_singleturn.yaml
else
  echo "OpenRLHF CLI flags differ by version; set --config support or edit this script." >&2
  exit 1
fi
