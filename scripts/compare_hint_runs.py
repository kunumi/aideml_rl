#!/usr/bin/env python3
"""Compare AIDE run logs across controller / baseline settings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aide.journal import Journal
from aide.utils import serialize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--logs_dirs",
        type=str,
        nargs="+",
        required=True,
        help="One or more log roots to compare.",
    )
    p.add_argument("--max_nodes", type=int, default=None)
    return p.parse_args()


def _metric_history(journal: Journal) -> list[float | None]:
    out: list[float | None] = []
    best: float | None = None
    maximize = True
    for n in journal.nodes:
        if n.metric is not None and n.metric.maximize is not None:
            maximize = bool(n.metric.maximize)
            break
    for n in journal.nodes:
        if n.is_buggy or n.metric is None or n.metric.value is None:
            out.append(best)
            continue
        v = float(n.metric.value)
        if best is None:
            best = v
        elif maximize:
            best = max(best, v)
        else:
            best = min(best, v)
        out.append(best)
    return out


def _first_valid_step(journal: Journal) -> int | None:
    for i, n in enumerate(journal.nodes):
        if not n.is_buggy and n.metric is not None and n.metric.value is not None:
            return i + 1
    return None


def summarize_journal(path: Path, max_nodes: int | None) -> dict:
    journal = serialize.load_json(path, Journal)
    nodes = journal.nodes[:max_nodes] if max_nodes else journal.nodes
    partial = Journal(nodes=nodes)
    hist = _metric_history(partial)
    return {
        "journal": str(path),
        "n_nodes": len(nodes),
        "n_buggy": sum(1 for n in nodes if n.is_buggy),
        "best_metric": hist[-1] if hist else None,
        "first_valid_step": _first_valid_step(partial),
        "n_with_hints": sum(1 for n in nodes if n.hint),
    }


def main() -> None:
    args = parse_args()
    all_rows: list[dict] = []
    for root in args.logs_dirs:
        root_path = Path(root)
        for jp in sorted(root_path.rglob("journal.json")):
            all_rows.append(summarize_journal(jp, args.max_nodes))

    print(json.dumps(all_rows, indent=2))

    by_run = {}
    for row in all_rows:
        run = Path(row["journal"]).parent.name
        by_run.setdefault(run, []).append(row)

    print("\n=== Summary by run folder ===")
    for run, rows in sorted(by_run.items()):
        bests = [r["best_metric"] for r in rows if r["best_metric"] is not None]
        avg_best = sum(bests) / len(bests) if bests else None
        print(
            f"{run}: journals={len(rows)} avg_best={avg_best} "
            f"avg_nodes={sum(r['n_nodes'] for r in rows)/len(rows):.1f}"
        )


if __name__ == "__main__":
    main()
