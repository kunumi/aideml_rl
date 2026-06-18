"""
Export hindsight controller training data from completed AIDE journals.

Each row supervises (action, hint, confidence) from current node state.
"""

from __future__ import annotations

import json
import os
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf

from ..backend import query
from ..journal import Journal, Node
from ..utils import serialize
from ..utils.config import SearchConfig
from .ctu_dataset import CTUTask, load_ctu_index
from .evaluator import extract_baseline
from .hint_prompt import (
    HINT_PROMPT_VERSION,
    HINT_SYSTEM_PROMPT,
    ControllerAction,
    abandon_hint_template,
    build_history_summary,
    format_controller_input,
    format_controller_target,
)
from .observation import task_desc_to_string
from .offline_extractor import _parse_run_dirname

FutureStrategy = Literal["best_child_by_subtree", "best_descendant_k", "best_leaf"]
TargetSource = Literal["plan", "analysis", "teacher"]


@dataclass
class SubtreeStats:
    best_subtree_metric: float | None = None
    best_subtree_node_id: str | None = None
    best_child_by_subtree_metric: float | None = None
    best_child_node_id: str | None = None
    delta_to_best_child: float | None = None
    delta_to_best_subtree: float | None = None
    is_valid: bool = False
    normalized_score: float | None = None


@dataclass
class ExportConfig:
    future_strategy: FutureStrategy = "best_child_by_subtree"
    horizon: int = 2
    min_delta: float = 0.0
    target_source: TargetSource = "plan"
    teacher_model: str | None = None
    max_hint_chars: int = 600
    max_code_chars: int = 8000
    max_output_chars: int = 4000
    min_preference_gap: float = 0.01
    holdout_datasets: set[str] = field(default_factory=set)


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


def _subtree_nodes(root: Node) -> list[Node]:
    out: list[Node] = []
    q: deque[Node] = deque([root])
    while q:
        n = q.popleft()
        out.append(n)
        for ch in n.children:
            q.append(ch)
    return out


def _descendants_within_k(root: Node, k: int) -> list[Node]:
    if k <= 0:
        return []
    out: list[Node] = []
    q: deque[tuple[Node, int]] = deque((ch, 1) for ch in root.children)
    while q:
        n, depth = q.popleft()
        out.append(n)
        if depth < k:
            for ch in n.children:
                q.append((ch, depth + 1))
    return out


def _is_valid_node(node: Node) -> bool:
    return not node.is_buggy and node.metric is not None and node.metric.value is not None


def _metric_value(node: Node) -> float | None:
    if not _is_valid_node(node):
        return None
    return float(node.metric.value)  # type: ignore[union-attr]


def _normalized_score(metric: float | None, maximize: bool) -> float | None:
    if metric is None:
        return None
    return metric if maximize else -metric


def _best_node_in_list(nodes: list[Node], maximize: bool) -> Node | None:
    best: Node | None = None
    for n in nodes:
        if not _is_valid_node(n):
            continue
        if best is None or n.metric > best.metric:  # type: ignore[operator]
            best = n
    return best


def compute_subtree_stats(journal: Journal, maximize: bool) -> dict[str, SubtreeStats]:
    stats: dict[str, SubtreeStats] = {}
    for node in journal.nodes:
        st = SubtreeStats()
        st.is_valid = _is_valid_node(node)
        st.normalized_score = _normalized_score(_metric_value(node), maximize)

        subtree = _subtree_nodes(node)
        best_sub = _best_node_in_list(subtree, maximize)
        if best_sub is not None:
            st.best_subtree_metric = _metric_value(best_sub)
            st.best_subtree_node_id = best_sub.id

        best_child: Node | None = None
        for ch in node.children:
            ch_sub = _subtree_nodes(ch)
            cand = _best_node_in_list(ch_sub, maximize)
            if cand is None:
                continue
            if best_child is None or cand.metric > best_child.metric:  # type: ignore[operator]
                best_child = cand
        if best_child is not None:
            st.best_child_by_subtree_metric = _metric_value(best_child)
            st.best_child_node_id = best_child.id

        cur = _metric_value(node)
        if cur is not None and st.best_child_by_subtree_metric is not None:
            raw = st.best_child_by_subtree_metric - cur
            st.delta_to_best_child = raw if maximize else -raw
        if cur is not None and st.best_subtree_metric is not None:
            raw = st.best_subtree_metric - cur
            st.delta_to_best_subtree = raw if maximize else -raw

        stats[node.id] = st
    return stats


def select_future_node(
    node: Node,
    *,
    strategy: FutureStrategy = "best_child_by_subtree",
    horizon: int = 2,
    maximize: bool = True,
) -> Node | None:
    if strategy == "best_child_by_subtree":
        candidates = list(node.children)
    elif strategy == "best_descendant_k":
        candidates = _descendants_within_k(node, horizon)
    elif strategy == "best_leaf":
        candidates = [n for n in _subtree_nodes(node) if n.is_leaf and n.id != node.id]
    else:
        raise ValueError(f"Unknown future strategy: {strategy}")

    return _best_node_in_list(candidates, maximize)


def _has_valid_descendant_within_k(node: Node, k: int) -> bool:
    return any(_is_valid_node(n) for n in _descendants_within_k(node, k))


def _has_improving_descendant(node: Node, k: int, maximize: bool) -> bool:
    cur = _metric_value(node)
    if cur is None:
        return False
    for desc in _descendants_within_k(node, k):
        dv = _metric_value(desc)
        if dv is None:
            continue
        raw = dv - cur
        if (raw > 0 if maximize else raw < 0):
            return True
    return False


def infer_hindsight_action(
    node: Node,
    future_node: Node | None,
    *,
    horizon: int,
    maximize: bool,
) -> ControllerAction:
    if future_node is not None:
        if node.is_buggy:
            return "debug"
        return "improve"

    if node.is_buggy:
        if _has_valid_descendant_within_k(node, horizon):
            return "debug"
        return "abandon"

    if _is_valid_node(node) and _has_improving_descendant(node, horizon, maximize):
        return "improve"
    return "abandon"


def _confidence_from_delta(delta: float | None, baseline: float, action: ControllerAction) -> float:
    if action == "abandon":
        return 0.0
    if delta is None:
        return 0.3
    denom = max(abs(float(baseline)), 1e-8)
    norm = max(min(float(delta) / denom, 1.0), 0.0)
    return round(norm, 3)


def _clean_hint_text(text: str | None, max_chars: int) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text.strip())
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def _hint_from_node(node: Node, source: TargetSource, max_chars: int) -> str:
    if source == "plan":
        return _clean_hint_text(node.plan, max_chars)
    if source == "analysis":
        return _clean_hint_text(node.analysis, max_chars)
    return ""


def _teacher_hint_prompt(current: Node, future: Node, maximize: bool) -> str:
    cur_m = _metric_value(current)
    fut_m = _metric_value(future)
    cur_m_s = "N/A" if cur_m is None else f"{cur_m:.6g}"
    fut_m_s = "N/A" if fut_m is None else f"{fut_m:.6g}"
    return (
        "You are generating training data for a controller that gives hints to a coding LLM.\n\n"
        "The controller will only see the current node at inference time.\n"
        "You can also see a better future node from the completed search tree.\n\n"
        "Write a short hint that would have helped the coding LLM move from the current node "
        "toward the better node.\n"
        "Do not reveal exact final code unless the change is local and diagnostic.\n"
        "Focus on the key reasoning insight.\n"
        "Return only the hint text (no JSON).\n\n"
        f"Current node code:\n```python\n{current.code[:4000]}\n```\n\n"
        f"Current execution output:\n```text\n{current.term_out[:2000]}\n```\n\n"
        f"Current analysis:\n```text\n{(current.analysis or '')[:1500]}\n```\n\n"
        f"Better future node code:\n```python\n{future.code[:4000]}\n```\n\n"
        f"Better future execution output:\n```text\n{future.term_out[:2000]}\n```\n\n"
        f"Better future analysis:\n```text\n{(future.analysis or '')[:1500]}\n```\n\n"
        f"Metric improvement: {cur_m_s} -> {fut_m_s}\n"
    )


def generate_teacher_hint(
    current: Node,
    future: Node,
    *,
    teacher_model: str,
    maximize: bool,
    max_chars: int,
) -> str:
    prompt = _teacher_hint_prompt(current, future, maximize)
    out = query(
        system_message=prompt,
        user_message=None,
        model=teacher_model,
        temperature=0.3,
    )
    assert isinstance(out, str)
    return _clean_hint_text(out, max_chars)


def build_target_hint(
    node: Node,
    future_node: Node | None,
    action: ControllerAction,
    *,
    cfg: ExportConfig,
    maximize: bool,
) -> tuple[str, str]:
    """Return (hint_text, hint_source)."""
    if action == "abandon" or future_node is None:
        if cfg.target_source == "teacher" and cfg.teacher_model:
            # No future node; fall back to template
            return abandon_hint_template(node), "abandon_template"
        return abandon_hint_template(node), "abandon_template"

    if cfg.target_source == "teacher" and cfg.teacher_model:
        hint = generate_teacher_hint(
            node,
            future_node,
            teacher_model=cfg.teacher_model,
            maximize=maximize,
            max_chars=cfg.max_hint_chars,
        )
        if hint:
            return hint, "teacher"
        # fallback
        plan_hint = _hint_from_node(future_node, "plan", cfg.max_hint_chars)
        if plan_hint:
            return plan_hint, "plan_fallback"
        return _hint_from_node(future_node, "analysis", cfg.max_hint_chars), "analysis_fallback"

    hint = _hint_from_node(future_node, cfg.target_source, cfg.max_hint_chars)
    if hint:
        return hint, cfg.target_source
    alt = "analysis" if cfg.target_source == "plan" else "plan"
    alt_hint = _hint_from_node(future_node, alt, cfg.max_hint_chars)  # type: ignore[arg-type]
    if alt_hint:
        return alt_hint, alt
    return abandon_hint_template(node), "fallback_template"


def _task_metadata(task: CTUTask) -> dict[str, Any]:
    return {
        "task_type": task.task_type,
        "target_column": task.target_column,
        "target_table": task.target_table,
        "dataset_name": task.dataset_name,
        "task_name": task.task_name,
        "row_name": task.row_name,
    }


def _passes_delta_filter(
    action: ControllerAction,
    delta: float | None,
    min_delta: float,
    node: Node,
    future_node: Node | None,
) -> bool:
    if action == "abandon":
        return True
    if future_node is None:
        return False
    if node.is_buggy and _is_valid_node(future_node):
        return True
    if delta is None:
        return False
    return delta >= min_delta


def build_sft_row(
    node: Node,
    *,
    journal: Journal,
    task: CTUTask,
    task_desc: str,
    seed: int | None,
    log_dir: str,
    journal_path: str,
    stats: dict[str, SubtreeStats],
    cfg: ExportConfig,
    maximize: bool,
    baseline: float,
) -> dict[str, Any] | None:
    st = stats[node.id]
    future = select_future_node(
        node,
        strategy=cfg.future_strategy,
        horizon=cfg.horizon,
        maximize=maximize,
    )
    action = infer_hindsight_action(node, future, horizon=cfg.horizon, maximize=maximize)

    delta = None
    if future is not None:
        cur = _metric_value(node)
        fut = _metric_value(future)
        if cur is not None and fut is not None:
            raw = fut - cur
            delta = raw if maximize else -raw

    if not _passes_delta_filter(action, delta, cfg.min_delta, node, future):
        return None

    hint, hint_source = build_target_hint(
        node, future, action, cfg=cfg, maximize=maximize
    )
    if not hint.strip():
        return None

    confidence = _confidence_from_delta(delta, baseline, action)
    user_input = format_controller_input(
        task_desc,
        node,
        history_summary=build_history_summary(node),
        dataset_metadata=_task_metadata(task),
        max_code_chars=cfg.max_code_chars,
        max_output_chars=cfg.max_output_chars,
    )
    target = format_controller_target(action, hint, confidence, max_hint_chars=cfg.max_hint_chars)

    cur_metric = _metric_value(node)
    fut_metric = _metric_value(future) if future else None

    return {
        "task_id": task.row_name,
        "run_id": Path(log_dir).name,
        "node_id": node.id,
        "future_node_id": future.id if future else None,
        "future_selection": cfg.future_strategy,
        "action": action,
        "confidence": confidence,
        "depth": sum(1 for _ in _lineage(node)) - 1,
        "current_metric": cur_metric,
        "future_metric": fut_metric,
        "delta_metric": delta,
        "valid": st.is_valid,
        "input": user_input,
        "target": target,
        "messages": [
            {"role": "system", "content": HINT_SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": target},
        ],
        "metadata": {
            "logs_dir": log_dir,
            "journal_path": journal_path,
            "higher_is_better": maximize,
            "hint_source": hint_source,
            "prompt_version": HINT_PROMPT_VERSION,
            "seed": seed,
            "best_subtree_metric": st.best_subtree_metric,
            "best_child_metric": st.best_child_by_subtree_metric,
        },
    }


def _lineage(node: Node) -> list[Node]:
    path: list[Node] = []
    cur: Node | None = node
    while cur is not None:
        path.append(cur)
        cur = cur.parent
    path.reverse()
    return path


def _child_subtree_metric(child: Node, maximize: bool) -> float | None:
    best = _best_node_in_list(_subtree_nodes(child), maximize)
    return _metric_value(best) if best else None


def build_preference_rows(
    node: Node,
    *,
    task: CTUTask,
    task_desc: str,
    seed: int | None,
    log_dir: str,
    cfg: ExportConfig,
    maximize: bool,
    baseline: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    user_input = format_controller_input(
        task_desc,
        node,
        history_summary=build_history_summary(node),
        dataset_metadata=_task_metadata(task),
        max_code_chars=cfg.max_code_chars,
        max_output_chars=cfg.max_output_chars,
    )

    children = list(node.children)
    if len(children) >= 2:
        scored: list[tuple[Node, float | None]] = []
        for ch in children:
            scored.append((ch, _child_subtree_metric(ch, maximize)))
        scored_valid = [(ch, m) for ch, m in scored if m is not None]
        if len(scored_valid) >= 2:
            if maximize:
                best_ch, best_m = max(scored_valid, key=lambda x: x[1])  # type: ignore[arg-type]
                worst_ch, worst_m = min(scored_valid, key=lambda x: x[1])  # type: ignore[arg-type]
            else:
                best_ch, best_m = min(scored_valid, key=lambda x: x[1])  # type: ignore[arg-type]
                worst_ch, worst_m = max(scored_valid, key=lambda x: x[1])  # type: ignore[arg-type]
            gap = abs(float(best_m) - float(worst_m))  # type: ignore[arg-type]
            if gap >= cfg.min_preference_gap:
                best_action = infer_hindsight_action(node, best_ch, horizon=1, maximize=maximize)
                worst_action = infer_hindsight_action(node, worst_ch, horizon=1, maximize=maximize)
                best_hint, _ = build_target_hint(
                    node, best_ch, best_action, cfg=cfg, maximize=maximize
                )
                worst_hint, _ = build_target_hint(
                    node, worst_ch, worst_action, cfg=cfg, maximize=maximize
                )
                if best_hint and worst_hint and best_hint != worst_hint:
                    rows.append(
                        {
                            "task_id": task.row_name,
                            "run_id": Path(log_dir).name,
                            "node_id": node.id,
                            "prompt": user_input,
                            "chosen": format_controller_target(
                                best_action,
                                best_hint,
                                _confidence_from_delta(gap, baseline, best_action),
                                max_hint_chars=cfg.max_hint_chars,
                            ),
                            "rejected": format_controller_target(
                                worst_action,
                                worst_hint,
                                _confidence_from_delta(0.0, baseline, worst_action),
                                max_hint_chars=cfg.max_hint_chars,
                            ),
                            "chosen_future_node_id": best_ch.id,
                            "rejected_future_node_id": worst_ch.id,
                            "chosen_metric": best_m,
                            "rejected_metric": worst_m,
                            "metadata": {
                                "preference_type": "child_subtree_gap",
                                "prompt_version": HINT_PROMPT_VERSION,
                                "seed": seed,
                            },
                        }
                    )

    # Action-level contrast for buggy nodes: debug vs abandon
    if node.is_buggy:
        future = select_future_node(
            node, strategy=cfg.future_strategy, horizon=cfg.horizon, maximize=maximize
        )
        debug_action: ControllerAction = "debug" if future else "abandon"
        abandon_action: ControllerAction = "abandon"
        if debug_action == "debug" and future is not None:
            debug_hint, _ = build_target_hint(
                node, future, "debug", cfg=cfg, maximize=maximize
            )
            abandon_hint = abandon_hint_template(node)
            rows.append(
                {
                    "task_id": task.row_name,
                    "run_id": Path(log_dir).name,
                    "node_id": node.id,
                    "prompt": user_input,
                    "chosen": format_controller_target(
                        "debug", debug_hint, 0.7, max_hint_chars=cfg.max_hint_chars
                    ),
                    "rejected": format_controller_target(
                        "abandon", abandon_hint, 0.0, max_hint_chars=cfg.max_hint_chars
                    ),
                    "chosen_future_node_id": future.id,
                    "rejected_future_node_id": None,
                    "chosen_metric": _metric_value(future),
                    "rejected_metric": None,
                    "metadata": {
                        "preference_type": "debug_vs_abandon",
                        "prompt_version": HINT_PROMPT_VERSION,
                        "seed": seed,
                    },
                }
            )
        elif debug_action == "abandon":
            abandon_hint = abandon_hint_template(node)
            # rejected: hypothetical debug with analysis/plan from best child if any
            best_child = select_future_node(
                node, strategy="best_child_by_subtree", horizon=1, maximize=maximize
            )
            if best_child is not None:
                rej_hint, _ = build_target_hint(
                    node, best_child, "debug", cfg=cfg, maximize=maximize
                )
                rows.append(
                    {
                        "task_id": task.row_name,
                        "run_id": Path(log_dir).name,
                        "node_id": node.id,
                        "prompt": user_input,
                        "chosen": format_controller_target(
                            "abandon", abandon_hint, 0.0, max_hint_chars=cfg.max_hint_chars
                        ),
                        "rejected": format_controller_target(
                            "debug", rej_hint, 0.3, max_hint_chars=cfg.max_hint_chars
                        ),
                        "chosen_future_node_id": None,
                        "rejected_future_node_id": best_child.id,
                        "chosen_metric": None,
                        "rejected_metric": _metric_value(best_child),
                        "metadata": {
                            "preference_type": "abandon_vs_debug",
                            "prompt_version": HINT_PROMPT_VERSION,
                            "seed": seed,
                        },
                    }
                )

    return rows


def export_journal_file(
    journal_path: Path,
    ctu_tasks_by_name: dict[str, CTUTask],
    *,
    cfg: ExportConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    journal = serialize.load_json(journal_path, Journal)
    log_dir = str(journal_path.parent.resolve())
    dirname = journal_path.parent.name
    task_guess, seed = _parse_run_dirname(dirname)

    task = ctu_tasks_by_name.get(task_guess)
    if task is None:
        for name, t in ctu_tasks_by_name.items():
            if name in dirname or dirname.endswith(name):
                task = t
                break
    if task is None:
        raise KeyError(
            f"Could not map log dir '{dirname}' to a CTU task row."
        )

    task_desc: str | dict = task.description
    cfg_path = journal_path.parent / "config.yaml"
    if cfg_path.is_file():
        run_cfg = OmegaConf.load(cfg_path)
        if run_cfg.get("goal"):
            td = {"Task goal": run_cfg.goal}
            if run_cfg.get("eval"):
                td["Task evaluation"] = run_cfg.eval
            task_desc = td

    td_str = task_desc_to_string(task_desc)
    baseline, maximize = extract_baseline(task.info, task.task_type)
    stats = compute_subtree_stats(journal, maximize)

    sft_rows: list[dict[str, Any]] = []
    pref_rows: list[dict[str, Any]] = []

    for node in journal.nodes:
        row = build_sft_row(
            node,
            journal=journal,
            task=task,
            task_desc=td_str,
            seed=seed,
            log_dir=log_dir,
            journal_path=str(journal_path),
            stats=stats,
            cfg=cfg,
            maximize=maximize,
            baseline=baseline,
        )
        if row is not None:
            sft_rows.append(row)
        pref_rows.extend(
            build_preference_rows(
                node,
                task=task,
                task_desc=td_str,
                seed=seed,
                log_dir=log_dir,
                cfg=cfg,
                maximize=maximize,
                baseline=baseline,
            )
        )

    return sft_rows, pref_rows


def export_logs_dir(
    logs_root: str | Path,
    out_sft: str | Path,
    out_prefs: str | Path,
    ctu_csv: str | Path,
    *,
    cfg: ExportConfig | None = None,
) -> dict[str, int]:
    cfg = cfg or ExportConfig()
    logs_root = Path(logs_root)
    out_sft = Path(out_sft)
    out_prefs = Path(out_prefs)
    out_sft.parent.mkdir(parents=True, exist_ok=True)
    out_prefs.parent.mkdir(parents=True, exist_ok=True)

    tasks = load_ctu_index(ctu_csv)
    by_name = {t.row_name: t for t in tasks}

    holdout = cfg.holdout_datasets
    train_sft = out_sft
    val_sft = out_sft.parent / "sft_val.jsonl" if holdout else None

    counts = {"sft": 0, "sft_val": 0, "preferences": 0, "errors": 0}

    def _is_holdout(task_id: str) -> bool:
        if not holdout:
            return False
        for h in holdout:
            if h in task_id or task_id.startswith(h):
                return True
        return False

    f_val_ctx = val_sft.open("w") if val_sft else open(os.devnull, "w")
    with out_sft.open("w") as f_train, out_prefs.open("w") as f_pref, f_val_ctx as f_val:

        for journal_path in sorted(logs_root.rglob("journal.json")):
            try:
                sft_rows, pref_rows = export_journal_file(
                    journal_path, by_name, cfg=cfg
                )
            except Exception as exc:
                err = {"error": str(exc), "journal_path": str(journal_path)}
                f_train.write(json.dumps(err) + "\n")
                counts["errors"] += 1
                continue

            for row in sft_rows:
                dest = f_val if _is_holdout(row["task_id"]) else f_train
                dest.write(json.dumps(row) + "\n")
                if _is_holdout(row["task_id"]):
                    counts["sft_val"] += 1
                else:
                    counts["sft"] += 1

            for row in pref_rows:
                f_pref.write(json.dumps(row) + "\n")
                counts["preferences"] += 1

    return counts
