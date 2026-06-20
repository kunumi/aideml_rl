"""Resolve prepared MLE-bench competitions for AIDE."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def _default_mlebench_data_dir() -> Path:
    try:
        from mlebench.registry import registry

        return registry.get_data_dir()
    except ImportError:
        from appdirs import user_cache_dir

        return Path(user_cache_dir()) / "mle-bench" / "data"


def get_competition_public_dir(
    competition_id: str,
    *,
    data_dir: str | Path | None = None,
) -> Path:
    """Return the prepared public data directory for a competition."""
    root = Path(data_dir) if data_dir else _default_mlebench_data_dir()
    public_dir = root / competition_id / "prepared" / "public"
    if not public_dir.is_dir():
        raise FileNotFoundError(
            f"MLE-bench competition {competition_id!r} is not prepared at {public_dir}. "
            f"Run: mlebench prepare -c {competition_id}"
        )
    return public_dir


def materialize_mlebench(
    competition_id: str,
    workdir: str | Path,
    *,
    data_dir: str | Path | None = None,
) -> Path:
    """
    Symlink/copy MLE-bench public competition data into an AIDE workspace input dir.
    """
    workdir = Path(workdir)
    input_dir = workdir / "input"
    public_dir = get_competition_public_dir(competition_id, data_dir=data_dir)

    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    for item in public_dir.iterdir():
        dest = input_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, symlinks=True)
        else:
            shutil.copy2(item, dest)

    try:
        from mlebench.data import get_leaderboard
        from mlebench.registry import registry

        competition = registry.set_data_dir(
            Path(data_dir) if data_dir else _default_mlebench_data_dir()
        ).get_competition(competition_id)
        description = competition.description
        competition_type = competition.competition_type
        sample_submission = str(competition.sample_submission)
        lower_is_better = competition.grader.is_lower_better(get_leaderboard(competition))
    except ImportError:
        description = f"MLE-bench competition: {competition_id}"
        competition_type = "unknown"
        sample_submission = ""
        lower_is_better = True

    task_info = {
        "benchmark": "mlebench",
        "competition_id": competition_id,
        "competition_type": competition_type,
        "sample_submission": sample_submission,
        "lower_is_better": lower_is_better,
    }
    (input_dir / "task_info.json").write_text(json.dumps(task_info, indent=2))
    (input_dir / "description.md").write_text(description)
    return input_dir


def build_aide_inputs(task_info: dict[str, Any], description: str) -> dict[str, str]:
    """Build goal/eval strings for AIDE from MLE-bench competition metadata."""
    competition_id = task_info["competition_id"]
    lower = task_info.get("lower_is_better", True)
    direction = "minimize" if lower else "maximize"

    goal = (
        f"You are solving the Kaggle competition '{competition_id}' (MLE-bench).\n\n"
        f"Competition description:\n{description}\n\n"
        "All competition files are under ./input. Train a model, validate on the "
        "provided train/validation split, and write test predictions to "
        "./working/submission.csv in the format shown in the sample_submission file.\n"
        "The submission.csv file is required for grading."
    )
    eval_text = (
        f"Official MLE-bench competition score ({direction} is better). "
        "Print your best validation score during training."
    )
    return {"goal": goal, "eval": eval_text}


def load_task_info(input_dir: str | Path) -> dict[str, Any]:
    path = Path(input_dir) / "task_info.json"
    return json.loads(path.read_text())
