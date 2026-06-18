"""
Offline decision dataset: replay heuristic journals and label each step with subtree-best R_t.
"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..journal import Journal, Node
from ..policy import SearchAction
from ..utils import serialize
from ..utils.config import SearchConfig
from omegaconf import OmegaConf

from .ctu_dataset import CTUTask, load_ctu_index
from .evaluator import extract_baseline
from .observation import build_observation, task_desc_to_string


def _default_search_cfg() -> SearchConfig:
    cfg_path = Path(__file__).resolve().parent.parent / "utils" / "config.yaml"
    root = OmegaConf.load(cfg_path)
    s = root.agent.search
    return SearchConfig(
        max_debug_depth=int(s.max_debug_depth),
        debug_prob=float(s.debug_prob),
        num_drafts=int(s.num_drafts),
        policy_kind=str(s.policy_kind),
        policy_model=s.policy_model,
        policy_temp=float(s.policy_temp),
        policy_max_obs_nodes=int(s.policy_max_obs_nodes),
        controller_kind=str(getattr(s, "controller_kind", "none")),
        controller_model=getattr(s, "controller_model", None),
        controller_temp=float(getattr(s, "controller_temp", 0.7)),
        hint_max_chars=int(getattr(s, "hint_max_chars", 600)),
        hint_pool_path=getattr(s, "hint_pool_path", None),
    )


def _parse_run_dirname(dirname: str) -> tuple[str, int | None]:
    """Parse log folder like '0-ctu-accidents_accidents-original__seed0' -> (task_row, seed)."""
    m = re.match(r"^\d+-(.+)__seed(\d+)$", dirname)
    if m:
        return m.group(1), int(m.group(2))
    m2 = re.match(r"^(.+)__seed(\d+)$", dirname)
    if m2:
        return m2.group(1), int(m2.group(2))
    return dirname, None


def _clone_journal_prefix(source: Journal, prefix_len: int) -> Journal:
    """First `prefix_len` nodes of `source`, preserving parent links within the prefix."""
    j = Journal()
    if prefix_len <= 0:
        return j
    subset = source.nodes[:prefix_len]
    id2n: dict[str, Node] = {}
    for old in subset:
        n = Node(
            code=old.code,
            plan=old.plan,
            id=old.id,
            ctime=old.ctime,
            parent=None,
            children=set(),
            _term_out=old._term_out,
            exec_time=old.exec_time,
            exc_type=old.exc_type,
            exc_info=old.exc_info,
            exc_stack=old.exc_stack,
            analysis=old.analysis,
            metric=old.metric,
            is_buggy=old.is_buggy,
            hint=old.hint,
        )
        id2n[old.id] = n
    for old in subset:
        if old.parent is not None and old.parent.id in id2n:
            id2n[old.id].parent = id2n[old.parent.id]
    for old in subset:
        j.append(id2n[old.id])
    return j


def _subtree_nodes(root: Node) -> list[Node]:
    out: list[Node] = []
    q: deque[Node] = deque([root])
    while q:
        n = q.popleft()
        out.append(n)
        for ch in n.children:
            q.append(ch)
    return out


def _best_metric_float_in_nodes(nodes: list[Node]) -> float | None:
    best: Node | None = None
    for n in nodes:
        if n.is_buggy or n.metric is None or n.metric.value is None:
            continue
        if best is None or n.metric > best.metric:
            best = n
    return float(best.metric.value) if best is not None else None


def _infer_action(child: Node) -> SearchAction:
    if child.parent is None:
        return SearchAction(kind="draft", parent_id=None)
    if child.stage_name == "debug":
        return SearchAction(kind="debug", parent_id=child.parent.id)
    return SearchAction(kind="improve", parent_id=child.parent.id)


def _journal_summaries_for_obs(journal: Journal, max_nodes: int) -> list[dict[str, Any]]:
    return journal.summary_for_policy(max_nodes=max_nodes)


def _count_debuggable(journal: Journal, search_cfg: SearchConfig) -> int:
    return len(
        [
            n
            for n in journal.buggy_nodes
            if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
        ]
    )


def _compute_rt(
    best_before: float | None,
    best_subtree: float | None,
    baseline: float,
    maximize: bool,
) -> float:
    denom = max(abs(float(baseline)), 1e-8)
    if best_subtree is None:
        return 0.0
    if best_before is None:
        raw = best_subtree - float(baseline)
    else:
        raw = best_subtree - best_before
    if not maximize:
        raw = -raw
    return max(min(raw / denom, 1.0), -1.0)


def replay_journal(
    journal: Journal,
    *,
    task_desc: str | dict,
    task_row_name: str,
    task_type: str,
    task_info: dict[str, Any],
    seed: int | None,
    log_dir: str,
    total_steps: int,
    search_cfg: SearchConfig | None = None,
) -> list[dict[str, Any]]:
    search_cfg = search_cfg or _default_search_cfg()
    baseline, maximize = extract_baseline(task_info, task_type)
    td_str = task_desc_to_string(task_desc)

    rows: list[dict[str, Any]] = []
    recent: list[dict[str, Any]] = []

    for t, child in enumerate(journal.nodes):
        partial = _clone_journal_prefix(journal, t)
        best_before = _best_metric_float_in_nodes(partial.nodes)

        subtree = _subtree_nodes(child)
        best_sub = _best_metric_float_in_nodes(subtree)

        rt = _compute_rt(best_before, best_sub, baseline, maximize)
        action = _infer_action(child)

        summaries = _journal_summaries_for_obs(partial, search_cfg.policy_max_obs_nodes)
        state_text = build_observation(
            summaries,
            td_str,
            baseline_metric=baseline,
            maximize=maximize,
            step_idx=t,
            total_steps=total_steps,
            best_metric_so_far=best_before,
            num_drafts=len(partial.draft_nodes),
            n_debuggable_leaves=_count_debuggable(partial, search_cfg),
            n_good_nodes=len(partial.good_nodes),
            recent_actions=list(recent),
        )

        snap = partial.summary_for_policy(max_nodes=256)
        scfg = {
            "max_debug_depth": search_cfg.max_debug_depth,
            "debug_prob": search_cfg.debug_prob,
            "num_drafts": search_cfg.num_drafts,
            "policy_kind": search_cfg.policy_kind,
            "policy_model": search_cfg.policy_model,
            "policy_temp": search_cfg.policy_temp,
            "policy_max_obs_nodes": search_cfg.policy_max_obs_nodes,
        }

        heuristic_action = {
            "action": action.kind,
            "parent_id": action.parent_id,
            "rationale": "heuristic_logged",
        }
        recent.append({"step": t, **heuristic_action})

        rows.append(
            {
                "state_text": state_text,
                "state": json.loads(state_text),
                "action": heuristic_action,
                "reward": rt,
                "meta": {
                    "task": task_row_name,
                    "seed": seed,
                    "node_id": child.id,
                    "log_dir": log_dir,
                    "step": t,
                    "is_buggy_child": bool(child.is_buggy),
                    "exec_time": child.exec_time,
                    "best_before": best_before,
                    "best_in_subtree": best_sub,
                },
                "label": {
                    "reward": rt,
                    "heuristic_action": heuristic_action,
                    "journal_snapshot": snap,
                    "search_cfg": scfg,
                    "maximize": maximize,
                },
            }
        )

    return rows


def extract_journal_file(
    journal_path: Path,
    ctu_tasks_by_name: dict[str, CTUTask],
    *,
    total_steps: int | None = None,
) -> list[dict[str, Any]]:
    journal = serialize.load_json(journal_path, Journal)
    log_dir = str(journal_path.parent.resolve())
    dirname = journal_path.parent.name
    task_guess, seed = _parse_run_dirname(dirname)

    task = ctu_tasks_by_name.get(task_guess)
    if task is None:
        for name, t in ctu_tasks_by_name.items():
            if name in dirname or dirname.endswith(name):
                task = t
                task_guess = name
                break
    if task is None:
        raise KeyError(f"Could not map log dir '{dirname}' to a CTU task row. Known: {list(ctu_tasks_by_name)[:3]}...")

    cfg_path = journal_path.parent / "config.yaml"
    task_desc: str | dict = task.description
    if cfg_path.is_file():
        cfg = OmegaConf.load(cfg_path)
        if cfg.get("goal"):
            td = {"Task goal": cfg.goal}
            if cfg.get("eval"):
                td["Task evaluation"] = cfg.eval
            task_desc = td

    n_steps = total_steps if total_steps is not None else len(journal.nodes)
    return replay_journal(
        journal,
        task_desc=task_desc,
        task_row_name=task.row_name,
        task_type=task.task_type,
        task_info=task.info,
        seed=seed,
        log_dir=log_dir,
        total_steps=max(n_steps, len(journal.nodes)),
        search_cfg=_default_search_cfg(),
    )


def extract_logs_dir(
    logs_root: str | Path,
    ctu_csv: str | Path,
    out_jsonl: str | Path,
    *,
    total_steps: int | None = None,
) -> int:
    logs_root = Path(logs_root)
    out_jsonl = Path(out_jsonl)
    tasks = load_ctu_index(ctu_csv)
    by_name = {t.row_name: t for t in tasks}

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_jsonl.open("w") as out:
        for journal_path in sorted(logs_root.rglob("journal.json")):
            try:
                rows = extract_journal_file(journal_path, by_name, total_steps=total_steps)
            except Exception as exc:
                row = {
                    "error": str(exc),
                    "journal_path": str(journal_path),
                }
                out.write(json.dumps(row) + "\n")
                continue
            for row in rows:
                out.write(json.dumps(row) + "\n")
                n += 1
    return n
