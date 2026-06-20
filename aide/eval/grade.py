"""Official grading for RelBench and MLE-bench eval submissions."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _find_prediction_column(submission: pd.DataFrame, target_col: str) -> str:
    if target_col in submission.columns:
        return target_col
    for candidate in ("prediction", "pred", "target", "label"):
        if candidate in submission.columns:
            return candidate
    numeric_cols = [
        c
        for c in submission.columns
        if c not in submission.columns[:3] and pd.api.types.is_numeric_dtype(submission[c])
    ]
    if len(numeric_cols) == 1:
        return numeric_cols[0]
    raise ValueError(
        f"Could not find prediction column in submission (expected '{target_col}' "
        f"or 'prediction'). Columns: {list(submission.columns)}"
    )


def _align_relbench_predictions(
    submission: pd.DataFrame,
    test_table_df: pd.DataFrame,
    entity_col: str,
    time_col: str | None,
    target_col: str,
) -> np.ndarray:
    join_cols = [entity_col]
    if time_col and time_col in test_table_df.columns and time_col in submission.columns:
        join_cols.append(time_col)

    pred_col = _find_prediction_column(submission, target_col)
    if list(submission.columns[: len(join_cols)]) == join_cols and len(submission) == len(
        test_table_df
    ):
        return submission[pred_col].to_numpy()

    merged = test_table_df[join_cols].merge(
        submission[join_cols + [pred_col]],
        on=join_cols,
        how="left",
        validate="one_to_one",
    )
    if merged[pred_col].isna().any():
        missing = int(merged[pred_col].isna().sum())
        raise ValueError(
            f"Submission missing predictions for {missing} test rows "
            f"(join keys: {join_cols})"
        )
    return merged[pred_col].to_numpy()


def grade_relbench_submission(
    submission_path: str | Path,
    dataset_name: str,
    task_name: str,
    *,
    primary_metric: str,
    download: bool = False,
) -> dict[str, Any]:
    """Grade a submission.csv against the official RelBench test labels."""
    from relbench.base import EntityTask
    from relbench.tasks import get_task

    submission_path = Path(submission_path)
    if not submission_path.is_file():
        return {
            "status": "error",
            "error": f"Submission not found: {submission_path}",
            "official_metric": None,
            "all_metrics": {},
        }

    rel_task = get_task(dataset_name, task_name, download=download)
    if not isinstance(rel_task, EntityTask):
        raise TypeError(f"Task {dataset_name}/{task_name} is not an EntityTask")

    submission = pd.read_csv(submission_path)
    test_masked = rel_task.get_table("test", mask_input_cols=True).df
    test_full = rel_task.get_table("test", mask_input_cols=False).df

    try:
        pred = _align_relbench_predictions(
            submission,
            test_masked,
            rel_task.entity_col,
            rel_task.time_col,
            rel_task.target_col,
        )
        all_metrics = rel_task.evaluate(pred, test_full)
        official = all_metrics.get(primary_metric)
        if official is None:
            raise KeyError(
                f"Metric {primary_metric!r} not in evaluate() output: {list(all_metrics)}"
            )
        return {
            "status": "ok",
            "official_metric": float(official),
            "primary_metric": primary_metric,
            "all_metrics": {k: float(v) for k, v in all_metrics.items()},
            "n_test_rows": len(test_full),
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "official_metric": None,
            "all_metrics": {},
        }


def grade_mlebench_submission(
    submission_path: str | Path,
    competition_id: str,
    *,
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Grade a submission.csv with the official MLE-bench grader."""
    submission_path = Path(submission_path)
    if not submission_path.is_file():
        return {
            "status": "error",
            "error": f"Submission not found: {submission_path}",
            "official_metric": None,
            "any_medal": False,
        }

    try:
        from mlebench.grade import grade_csv
        from mlebench.registry import registry as mle_registry

        data_root = Path(data_dir) if data_dir else mle_registry.get_data_dir()
        reg = mle_registry.set_data_dir(data_root)
        competition = reg.get_competition(competition_id)
        report = grade_csv(submission_path, competition)
        report_dict = report.to_dict()
        return {
            "status": "ok" if report.valid_submission else "invalid_submission",
            "official_metric": report.score,
            "any_medal": report.any_medal,
            "gold_medal": report.gold_medal,
            "silver_medal": report.silver_medal,
            "bronze_medal": report.bronze_medal,
            "above_median": report.above_median,
            "is_lower_better": report.is_lower_better,
            "report": report_dict,
        }
    except ImportError:
        return _grade_mlebench_subprocess(submission_path, competition_id, data_dir)

    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "official_metric": None,
            "any_medal": False,
        }


def _grade_mlebench_subprocess(
    submission_path: Path,
    competition_id: str,
    data_dir: str | Path | None,
) -> dict[str, Any]:
    cmd = ["mlebench", "grade-sample", str(submission_path), competition_id]
    if data_dir:
        cmd.extend(["--data-dir", str(data_dir)])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return {
            "status": "error",
            "error": "mlebench CLI not found; pip install -e /path/to/mle-bench",
            "official_metric": None,
            "any_medal": False,
        }
    if result.returncode != 0:
        return {
            "status": "error",
            "error": result.stderr.strip() or result.stdout.strip(),
            "official_metric": None,
            "any_medal": False,
        }
    # grade-sample logs JSON via logger; try to parse last JSON object from stdout
    stdout = result.stdout.strip()
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                report_dict = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    else:
        # Fallback: scan for a JSON block
        start = stdout.rfind("{")
        if start < 0:
            return {
                "status": "error",
                "error": f"Could not parse mlebench output: {stdout[:500]}",
                "official_metric": None,
                "any_medal": False,
            }
        report_dict = json.loads(stdout[start:])

    return {
        "status": "ok" if report_dict.get("valid_submission") else "invalid_submission",
        "official_metric": report_dict.get("score"),
        "any_medal": report_dict.get("any_medal", False),
        "gold_medal": report_dict.get("gold_medal", False),
        "silver_medal": report_dict.get("silver_medal", False),
        "bronze_medal": report_dict.get("bronze_medal", False),
        "above_median": report_dict.get("above_median", False),
        "is_lower_better": report_dict.get("is_lower_better", True),
        "report": report_dict,
    }
