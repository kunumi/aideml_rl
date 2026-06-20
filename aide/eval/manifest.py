"""Load and validate the eval task manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


Benchmark = Literal["relbench", "mlebench"]


@dataclass
class EvalTask:
    benchmark: Benchmark
    id: str
    metric: str
    higher_is_better: bool
    notes: str = ""
    relbench: dict[str, str] | None = None
    mlebench: dict[str, str] | None = None

    @property
    def relbench_dataset(self) -> str | None:
        return self.relbench.get("dataset") if self.relbench else None

    @property
    def relbench_task(self) -> str | None:
        return self.relbench.get("task") if self.relbench else None

    @property
    def mlebench_competition_id(self) -> str | None:
        return self.mlebench.get("competition_id") if self.mlebench else None


def _parse_task_row(row: dict[str, Any]) -> EvalTask:
    benchmark = row["benchmark"]
    if benchmark not in ("relbench", "mlebench"):
        raise ValueError(f"Unknown benchmark: {benchmark!r}")
    if benchmark == "relbench" and not row.get("relbench"):
        raise ValueError(f"RelBench task {row.get('id')} missing relbench block")
    if benchmark == "mlebench" and not row.get("mlebench"):
        raise ValueError(f"MLE-bench task {row.get('id')} missing mlebench block")
    return EvalTask(
        benchmark=benchmark,
        id=row["id"],
        metric=row["metric"],
        higher_is_better=bool(row.get("higher_is_better", True)),
        notes=row.get("notes", ""),
        relbench=row.get("relbench"),
        mlebench=row.get("mlebench"),
    )


def load_eval_manifest(path: str | Path) -> list[EvalTask]:
    """Load eval tasks from a JSONL manifest."""
    path = Path(path)
    tasks: list[EvalTask] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
                tasks.append(_parse_task_row(row))
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                raise ValueError(f"Invalid manifest line {line_no} in {path}: {exc}") from exc
    return tasks


def filter_tasks(
    tasks: list[EvalTask],
    *,
    benchmark: str | None = None,
    task_id: str | None = None,
) -> list[EvalTask]:
    out = tasks
    if benchmark and benchmark != "all":
        out = [t for t in out if t.benchmark == benchmark]
    if task_id:
        out = [t for t in out if t.id == task_id]
    return out
