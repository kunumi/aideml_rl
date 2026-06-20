#!/usr/bin/env python3
"""Flatten an OpenRLHF YAML config into argparse-style CLI tokens.

OpenRLHF's train_sft/train_dpo CLIs take flat namespaced flags
(e.g. ``--train.batch_size 64``) and have no native ``--config`` support, so we
convert the YAML here. One token is printed per line, ready for bash ``mapfile``.

Usage:
    python _yaml_to_args.py CONFIG.yaml [key.path=value ...]

Trailing ``key=value`` arguments override (or add) flattened YAML entries.
"""

from __future__ import annotations

import sys

import yaml

# Boolean leaves that are argparse ``store_true`` flags: emit a bare flag when
# True, nothing when False. Any other boolean (e.g. ``logger.wandb.key``) is a
# valued argument and is emitted as ``--key true/false``.
STORE_TRUE_FLAGS = {
    "data.apply_chat_template",
    "data.multiturn",
    "data.disable_fast_tokenizer",
    "ds.packing_samples",
    "ds.adam_offload",
    "ds.overlap_comm",
    "ds.use_liger_kernel",
    "ds.load_in_4bit",
    "ds.use_universal_ckpt",
    "ds.deepcompile",
    "ckpt.save_hf",
    "ckpt.disable_ds",
    "ckpt.load_enable",
    "model.gradient_checkpointing_enable",
    "model.gradient_checkpointing_reentrant",
    "model.pretrain_mode_enable",
    "model.ipo_enable",
    "ref.offload",
    "train.full_determinism_enable",
}


def _flatten(node: dict, prefix: str = "") -> dict:
    flat: dict = {}
    for key, value in node.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, dotted))
        else:
            flat[dotted] = value
    return flat


def _coerce_override(value: str):
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    return value


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: _yaml_to_args.py CONFIG.yaml [key=value ...]")

    with open(sys.argv[1]) as fh:
        cfg = yaml.safe_load(fh) or {}
    flat = _flatten(cfg)

    for override in sys.argv[2:]:
        if "=" not in override:
            continue
        key, raw = override.split("=", 1)
        flat[key] = _coerce_override(raw)

    for key, value in flat.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if key in STORE_TRUE_FLAGS:
                if value:
                    print(f"--{key}")
            else:
                print(f"--{key}")
                print("true" if value else "false")
        else:
            print(f"--{key}")
            print(value)


if __name__ == "__main__":
    main()
