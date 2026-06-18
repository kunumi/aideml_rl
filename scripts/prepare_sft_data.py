#!/usr/bin/env python3
"""Build SFT JSONL from offline heuristic decision rows."""

from __future__ import annotations

import argparse
import json
import math
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
    p.add_argument("--out_path", type=str, default="data/sft.jsonl")
    p.add_argument("--positive_only", action="store_true")
    p.add_argument("--top_quantile", type=float, default=None, help="Keep rows with reward >= this quantile (0..1).")
    p.add_argument("--weight_by_reward", action="store_true", help="RWR-style per-sample weight exp(r/T).")
    p.add_argument("--rwr_temperature", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_path)
    if not in_path.is_file():
        raise FileNotFoundError(in_path)

    rows_in: list[dict] = []
    with in_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "error" in d:
                continue
            rows_in.append(d)

    rewards = [float(r["reward"]) for r in rows_in]
    thresh = None
    if args.top_quantile is not None:
        rewards_sorted = sorted(rewards)
        idx = int(len(rewards_sorted) * float(args.top_quantile))
        idx = max(0, min(len(rewards_sorted) - 1, idx))
        thresh = rewards_sorted[idx]

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as out:
        for r in rows_in:
            rew = float(r["reward"])
            if args.positive_only and rew <= 0:
                continue
            if thresh is not None and rew < thresh:
                continue
            action = r.get("action") or {}
            assistant = json.dumps(
                {
                    "action": action.get("action"),
                    "parent_id": action.get("parent_id"),
                    "rationale": action.get("rationale", ""),
                },
                separators=(",", ":"),
            )
            rec = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": r["state_text"]},
                    {"role": "assistant", "content": assistant},
                ],
                "label": r.get("meta", {}),
            }
            if args.weight_by_reward:
                rec["weight"] = math.exp(rew / max(args.rwr_temperature, 1e-6))
            out.write(json.dumps(rec) + "\n")
            n += 1
    print(f"Wrote {n} SFT rows to {out_path}")


if __name__ == "__main__":
    main()
