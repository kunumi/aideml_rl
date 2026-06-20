"""Materialize native snap-stanford RelBench tasks for AIDE."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _write_table(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def _task_type_label(task_type) -> str:
    from relbench.base import TaskType

    mapping = {
        TaskType.REGRESSION: "regression",
        TaskType.BINARY_CLASSIFICATION: "binary classification",
        TaskType.MULTICLASS_CLASSIFICATION: "multiclass classification",
        TaskType.MULTILABEL_CLASSIFICATION: "multilabel classification",
    }
    return mapping.get(task_type, str(task_type.value))


def _primary_metric_names(task) -> list[str]:
    return [fn.__name__ for fn in task.metrics]


def materialize_relbench_native(
    dataset_name: str,
    task_name: str,
    workdir: str | Path,
    *,
    download: bool = True,
) -> Path:
    """
    Materialize a native RelBench predictive task into an AIDE workspace.

    Writes train/val/test parquets, relational db_tables, and task_info.json under input/.
    """
    from relbench.base import EntityTask
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    workdir = Path(workdir)
    input_dir = workdir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_dataset(dataset_name, download=download)
    rel_task = get_task(dataset_name, task_name, download=download)
    if not isinstance(rel_task, EntityTask):
        raise TypeError(
            f"Task {dataset_name}/{task_name} is not an EntityTask; "
            "only entity prediction tasks are supported."
        )

    for split in ("train", "val", "test"):
        mask = split == "test"
        split_table = rel_task.get_table(split, mask_input_cols=mask)
        _write_table(split_table.df, input_dir / f"{split}.parquet")

    db = dataset.get_db()
    db_tables = getattr(db, "table_dict", None) or getattr(db, "tables", {})
    if isinstance(db_tables, dict):
        for name, table in db_tables.items():
            table_df = getattr(table, "df", table)
            if isinstance(table_df, pd.DataFrame):
                _write_table(table_df, input_dir / "db_tables" / f"{name}.parquet")

    task_info = {
        "benchmark": "relbench",
        "dataset": dataset_name,
        "task": task_name,
        "task_type": _task_type_label(rel_task.task_type),
        "entity_col": rel_task.entity_col,
        "entity_table": rel_task.entity_table,
        "time_col": rel_task.time_col,
        "target_col": rel_task.target_col,
        "metrics": _primary_metric_names(rel_task),
        "val_timestamp": str(dataset.val_timestamp),
        "test_timestamp": str(dataset.test_timestamp),
    }
    (input_dir / "task_info.json").write_text(json.dumps(task_info, indent=2))
    return input_dir


def build_aide_inputs(task_info: dict[str, Any], *, primary_metric: str) -> dict[str, str]:
    """Build goal/eval strings for AIDE from RelBench task metadata."""
    entity_col = task_info["entity_col"]
    time_col = task_info.get("time_col")
    target_col = task_info["target_col"]
    task_type = task_info["task_type"]
    dataset = task_info["dataset"]
    task = task_info["task"]

    join_cols = [entity_col]
    if time_col:
        join_cols.append(time_col)
    join_desc = ", ".join(join_cols)

    goal = (
        f"You are solving a RelBench relational ML task on dataset '{dataset}' "
        f"(task: '{task}').\n\n"
        f"Task type: {task_type}.\n"
        f"Predict the column '{target_col}' for each row in test.parquet.\n\n"
        f"Data layout under ./input:\n"
        f"- train.parquet, val.parquet: labeled examples (entity key + target).\n"
        f"- test.parquet: held-out rows with labels masked (entity keys only).\n"
        f"- db_tables/*.parquet: full relational database tables for feature engineering.\n\n"
        f"Entity key column: {entity_col}.\n"
        f"Entity table: {task_info['entity_table']}.\n"
    )
    if time_col:
        goal += f"Time column: {time_col}.\n"

    goal += (
        "\nTrain on train.parquet, tune/validate on val.parquet, and produce predictions "
        "for every row in test.parquet.\n"
        f"Save test predictions to ./working/submission.csv with columns: {join_desc}, "
        f"and a prediction column named '{target_col}' (one row per test row, same order "
        "as test.parquet is recommended).\n"
        "Also print the validation metric on val.parquet during training."
    )

    direction = "maximize" if primary_metric in {"roc_auc", "r2", "accuracy", "f1", "average_precision"} else "minimize"
    eval_text = (
        f"Official RelBench metric: {primary_metric} ({direction} is better). "
        f"During development, print validation {primary_metric} on val.parquet."
    )
    return {"goal": goal, "eval": eval_text}


def load_task_info(input_dir: str | Path) -> dict[str, Any]:
    path = Path(input_dir) / "task_info.json"
    return json.loads(path.read_text())
