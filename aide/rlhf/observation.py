"""Canonical observation text for offline RL, SFT, GRPO, and live LLMPolicy."""

from __future__ import annotations

import json
from typing import Any


def build_observation(
    journal_summaries: list[dict[str, Any]],
    task_desc: str,
    baseline_metric: float,
    maximize: bool,
    step_idx: int,
    total_steps: int,
    *,
    best_metric_so_far: float | None = None,
    num_drafts: int = 0,
    n_debuggable_leaves: int = 0,
    n_good_nodes: int = 0,
    recent_actions: list[dict[str, Any]] | None = None,
) -> str:
    """Return a single JSON string used as the user observation everywhere."""
    payload: dict[str, Any] = {
        "title": f"AIDE_search_policy_turn_{step_idx + 1}_of_{total_steps}",
        "task": task_desc,
        "baseline_metric": baseline_metric,
        "higher_is_better": maximize,
        "best_metric_so_far": best_metric_so_far,
        "tree_stats": {
            "num_drafts": num_drafts,
            "n_debuggable_leaves": n_debuggable_leaves,
            "n_good_nodes": n_good_nodes,
            "n_nodes_in_window": len(journal_summaries),
        },
        "tree": journal_summaries,
        "recent_heuristic_actions": recent_actions or [],
        "instruction": (
            "Reply with exactly one JSON object (no markdown): "
            '{"action":"draft|debug|improve","parent_id":null or "<node id>","rationale":"<=120 chars"}'
        ),
    }
    return json.dumps(payload, separators=(",", ":"))


def task_desc_to_string(task_desc: str | dict) -> str:
    if isinstance(task_desc, str):
        return task_desc
    return json.dumps(task_desc, separators=(",", ":"))
