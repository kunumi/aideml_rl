"""Verifier-only reward for offline GRPO (single-turn, no live AIDE)."""

from __future__ import annotations

import json
import re
from typing import Any

from ..journal import Journal, Node
from ..policy import SearchAction, validate_action
from ..utils.config import SearchConfig
from ..utils.metric import MetricValue


def _extract_json_object(text: str) -> dict[str, Any] | None:
    t = (text or "").strip()
    if not t:
        return None
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", t)
    if fence:
        t = fence.group(1).strip()
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def journal_from_snapshots(snapshots: list[dict[str, Any]], maximize: bool) -> Journal:
    """Rebuild a minimal Journal for validate_action (IDs and flags preserved)."""
    j = Journal()
    id2n: dict[str, Node] = {}
    for s in snapshots:
        mid = s.get("id")
        if not mid:
            continue
        mv = s.get("metric_value")
        met = (
            MetricValue(float(mv), maximize=maximize)
            if mv is not None
            else MetricValue(None, maximize=maximize)
        )
        n = Node(
            code="",
            plan=s.get("plan_excerpt") or "",
            id=str(mid),
            is_buggy=bool(s.get("is_buggy")),
            metric=met,
            parent=None,
            children=set(),
        )
        id2n[str(mid)] = n
    for s in snapshots:
        mid = s.get("id")
        if not mid:
            continue
        pid = s.get("parent_id")
        if pid and str(pid) in id2n:
            id2n[str(mid)].parent = id2n[str(pid)]
    for s in snapshots:
        mid = s.get("id")
        if mid and str(mid) in id2n:
            j.append(id2n[str(mid)])
    return j


def search_config_from_label(d: dict[str, Any]) -> SearchConfig:
    return SearchConfig(
        max_debug_depth=int(d["max_debug_depth"]),
        debug_prob=float(d["debug_prob"]),
        num_drafts=int(d["num_drafts"]),
        policy_kind=str(d.get("policy_kind", "heuristic")),
        policy_model=d.get("policy_model"),
        policy_temp=float(d.get("policy_temp", 0.7)),
        policy_max_obs_nodes=int(d.get("policy_max_obs_nodes", 32)),
    )


def reward_one(response: str, label: dict[str, Any]) -> float:
    """
    + label['reward'] if response matches logged heuristic action and is structurally valid.
    0.0 if valid JSON + valid action but differs from heuristic.
    -1.0 if malformed or structurally invalid.
    """
    parsed = _extract_json_object(response)
    if parsed is None:
        return -1.0
    action = parsed.get("action")
    parent_id = parsed.get("parent_id")
    if action not in {"draft", "debug", "improve"}:
        return -1.0

    sa = SearchAction(
        kind=action,
        parent_id=parent_id,
        rationale=str(parsed.get("rationale", "")),
    )
    maximize = bool(label.get("maximize", True))
    snap = label.get("journal_snapshot") or []
    if not isinstance(snap, list):
        return -1.0
    try:
        j = journal_from_snapshots(snap, maximize=maximize)
    except Exception:
        return -1.0
    try:
        scfg = search_config_from_label(label["search_cfg"])
    except Exception:
        return -1.0

    ok, _reason = validate_action(sa, j, scfg)
    if not ok:
        return -1.0

    h = label.get("heuristic_action") or {}
    if h.get("action") == sa.kind and h.get("parent_id") == sa.parent_id:
        return float(label.get("reward", 0.0))
    return 0.0
