#!/usr/bin/env python3
"""Sample and inspect exported hint controller rows."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--path",
        type=str,
        default="data/heuristic_runs/hint_controller/sft.jsonl",
    )
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stats_only", action="store_true")
    return p.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "error" in d:
                continue
            rows.append(d)
    return rows


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    if not path.is_file():
        raise FileNotFoundError(path)

    rows = load_rows(path)
    print(f"Loaded {len(rows)} rows from {path}")

    actions = Counter(r.get("action", "?") for r in rows)
    print("\nAction distribution:")
    for k, v in sorted(actions.items()):
        print(f"  {k}: {v}")

    sources = Counter(r.get("metadata", {}).get("hint_source", "?") for r in rows)
    print("\nHint source distribution:")
    for k, v in sorted(sources.items()):
        print(f"  {k}: {v}")

    if args.stats_only:
        return

    random.seed(args.seed)
    sample = random.sample(rows, min(args.n, len(rows)))
    for i, row in enumerate(sample, 1):
        print(f"\n{'=' * 60}\nSample {i}")
        print(f"task={row.get('task_id')} node={row.get('node_id')} action={row.get('action')}")
        print(f"delta={row.get('delta_metric')} confidence={row.get('confidence')}")
        print(f"future={row.get('future_node_id')} source={row.get('metadata', {}).get('hint_source')}")
        print("\n--- INPUT (excerpt) ---")
        print((row.get("input") or "")[:1200])
        print("\n--- TARGET ---")
        print(row.get("target"))


if __name__ == "__main__":
    main()
