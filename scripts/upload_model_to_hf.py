#!/usr/bin/env python3
"""Upload a trained AIDE controller checkpoint to Hugging Face (model repo)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.hf_utils import upload_model_dir

from dotenv import load_dotenv

load_dotenv()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload AIDE model checkpoint to Hugging Face.")
    p.add_argument(
        "--model_dir",
        type=str,
        default="checkpoints/aide_hint_controller_dpo",
        help="Local checkpoint directory (config.json + model.safetensors, etc.).",
    )
    p.add_argument(
        "--repo_id",
        type=str,
        default="guilhermedrud/aide_rl_dpo",
        help="Target Hugging Face model repo id.",
    )
    p.add_argument(
        "--public",
        action="store_true",
        help="Make the model repo public (default: private).",
    )
    p.add_argument("--revision", type=str, default="main")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not (model_dir / "config.json").is_file():
        raise SystemExit(f"Missing config.json under {model_dir}")
    uri = upload_model_dir(
        model_dir,
        args.repo_id,
        private=not args.public,
        revision=args.revision,
    )
    print(f"Model uploaded: {uri}")


if __name__ == "__main__":
    main()
