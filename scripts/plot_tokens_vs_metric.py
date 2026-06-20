#!/usr/bin/env python3
"""
Plot tokens vs metric from eval_results.csv and per-run journal.json files.

Produces three plot types under data/eval/plots/:
  - scatter_tokens_vs_metric.png   (run-level total tokens vs official metric)
  - per_node_tokens_vs_metric.png  (per-node tokens vs node metric)
  - cumulative_tokens_vs_best.png  (cumulative tokens vs best-metric-so-far)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aide.utils import serialize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot tokens vs metric from eval runs.")
    p.add_argument("--results_csv", type=str, default="data/eval/eval_results.csv")
    p.add_argument("--out_dir", type=str, default="data/eval/plots")
    return p.parse_args()


def _load_journal(log_dir: str | Path):
    path = Path(log_dir) / "journal.json"
    if not path.is_file():
        return None
    return serialize.loads_json(path.read_text())


def _metric_value(node) -> float | None:
    if node.metric is None or node.metric.value is None:
        return None
    try:
        return float(node.metric.value)
    except (TypeError, ValueError):
        return None


def plot_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    valid = df.dropna(subset=["total_tokens", "official_metric"])
    if valid.empty:
        print("[plot] skip scatter: no rows with total_tokens and official_metric")
        return

    tasks = valid["task"].unique()
    n = len(tasks)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    colors = {"relbench": "tab:blue", "mlebench": "tab:orange"}

    for idx, task in enumerate(tasks):
        ax = axes[idx // ncols][idx % ncols]
        sub = valid[valid["task"] == task]
        for benchmark, grp in sub.groupby("benchmark"):
            ax.scatter(
                grp["total_tokens"],
                grp["official_metric"],
                label=benchmark,
                c=colors.get(benchmark, "gray"),
                alpha=0.8,
                s=60,
            )
        ax.set_title(task)
        ax.set_xlabel("Total tokens (run)")
        ax.set_ylabel("Official metric")
        ax.legend(fontsize=8)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Run-level tokens vs official metric", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_tokens_vs_metric.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_node(df: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for _, run in df.iterrows():
        journal = _load_journal(run["log_dir"])
        if journal is None:
            continue
        for node in journal.nodes:
            mv = _metric_value(node)
            if mv is None:
                continue
            rows.append(
                {
                    "task": run["task"],
                    "benchmark": run["benchmark"],
                    "seed": run["seed"],
                    "stage": node.stage_name,
                    "total_tokens": node.total_tokens,
                    "metric_value": mv,
                }
            )
    if not rows:
        print("[plot] skip per-node: no node metrics in journals")
        return

    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    stage_colors = {"draft": "tab:green", "improve": "tab:blue", "debug": "tab:red"}
    for stage, grp in plot_df.groupby("stage"):
        ax.scatter(
            grp["total_tokens"],
            grp["metric_value"],
            label=stage,
            c=stage_colors.get(stage, "gray"),
            alpha=0.5,
            s=25,
        )
    ax.set_xlabel("Node total tokens")
    ax.set_ylabel("Node validation metric")
    ax.set_title("Per-node tokens vs metric")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "per_node_tokens_vs_metric.png", dpi=150)
    plt.close(fig)


def plot_cumulative(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0
    for _, run in df.iterrows():
        journal = _load_journal(run["log_dir"])
        if journal is None or not journal.nodes:
            continue
        cum_tokens = []
        best_so_far = []
        running_tokens = 0
        best_node = None
        for node in journal.nodes:
            running_tokens += node.total_tokens
            if not node.is_buggy and node.metric is not None and node.metric.value is not None:
                if best_node is None or node.metric > best_node.metric:
                    best_node = node
            cum_tokens.append(running_tokens)
            best_so_far.append(
                best_node.metric.value if best_node is not None else None
            )
        if not cum_tokens:
            continue
        label = f"{run['task']} s{run['seed']}"
        ax.plot(cum_tokens, best_so_far, alpha=0.7, label=label)
        plotted += 1

    if plotted == 0:
        print("[plot] skip cumulative: no journal data")
        plt.close(fig)
        return

    ax.set_xlabel("Cumulative tokens")
    ax.set_ylabel("Best validation metric so far")
    ax.set_title("Cumulative tokens vs best-metric-so-far")
    if plotted <= 12:
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_tokens_vs_best.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = Path(args.results_csv)
    if not results_path.is_file():
        raise SystemExit(f"Results file not found: {results_path}")

    df = pd.read_csv(results_path)
    plot_scatter(df, out_dir)
    plot_per_node(df, out_dir)
    plot_cumulative(df, out_dir)
    print(f"[plot] saved plots to {out_dir}")


if __name__ == "__main__":
    main()
