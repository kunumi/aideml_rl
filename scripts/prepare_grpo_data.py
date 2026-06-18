#!/usr/bin/env python3
"""Build GRPO prompt JSONL (no assistant turn) from offline_decisions.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SYSTEM_PROMPT = (
    "You are the AIDE search policy. Given the JSON observation, choose the next tree action. "
    "Reply with exactly one JSON object: "
    '{"action":"draft|debug|improve","parent_id":null or "<node id>","rationale":"<=120 chars"}'
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in_path", type=str, default="data/offline_decisions.jsonl")
    p.add_argument("--out_path", type=str, default="data/grpo_prompts.jsonl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_path)
    if not in_path.is_file():
        raise FileNotFoundError(in_path)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "error" in d:
                continue
            lab = d.get("label")
            if not isinstance(lab, dict):
                continue
            rec = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": d["state_text"]},
                ],
                "label": lab,
            }
            fout.write(json.dumps(rec) + "\n")
            n += 1
    print(f"Wrote {n} GRPO prompt rows to {out_path}")


if __name__ == "__main__":
    main()
