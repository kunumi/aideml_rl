import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class CTUTask:
    row_name: str
    dataset_name: str
    task_name: str
    task_type: str
    target_column: str
    target_table: str
    description: str
    info: dict[str, Any]
    baseline_metrics: dict[str, float]


def _split_name(name: str) -> tuple[str, str]:
    if "_" in name:
        dataset, task = name.split("_", 1)
        return dataset, task
    return name, "original"


def is_kaggle_index(csv_path: str | Path) -> bool:
    return Path(csv_path).name == "kaggle_datasets_info.csv"


def _extract_baseline_metrics(info: dict[str, Any]) -> dict[str, float]:
    return {
        k: float(v)
        for k, v in info.items()
        if k.startswith("val_") and isinstance(v, (int, float))
    }


def load_ctu_index(csv_path: str | Path) -> list[CTUTask]:
    return load_task_index(csv_path)


def load_task_index(csv_path: str | Path) -> list[CTUTask]:
    df = pd.read_csv(csv_path)
    out: list[CTUTask] = []
    for _, row in df.iterrows():
        info = json.loads(row["info"])
        dataset_name, task_name = _split_name(row["name"])
        out.append(
            CTUTask(
                row_name=row["name"],
                dataset_name=dataset_name,
                task_name=task_name,
                task_type=row["task"],
                target_column=info.get("target_column", ""),
                target_table=info.get("target_table", ""),
                description=info.get("description", "") or info.get("task_description", ""),
                info=info,
                baseline_metrics=_extract_baseline_metrics(info),
            )
        )
    return out


def _write_table(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def _apply_redelex_compat_patches() -> None:
    """Work around redelex CTU dataset bugs when loading from MariaDB."""
    try:
        from redelex.datasets.ctu_datasets import Mondial
    except ImportError:
        return

    if getattr(Mondial, "_aide_compat_patched", False):
        return

    def _safe_mondial_customize_db(self, db):
        for table_name, col_name in (
            ("organization", "Established"),
            ("politics", "Independence"),
        ):
            if table_name not in db.table_dict:
                continue
            df = db.table_dict[table_name].df
            if col_name not in df.columns:
                continue
            series = df[col_name]
            if not pd.api.types.is_datetime64_any_dtype(series):
                series = pd.to_datetime(series, errors="coerce")
            df[col_name] = series.dt.year

        return db

    Mondial.customize_db = _safe_mondial_customize_db
    Mondial._aide_compat_patched = True


def _is_temporal_split_range(split_range) -> bool:
    if isinstance(split_range, pd.Series):
        return pd.api.types.is_datetime64_any_dtype(split_range)
    return False


def _coerce_impute_target(df: pd.DataFrame, rel_task) -> pd.DataFrame:
    from relbench.base import TaskType

    target = df[rel_task.target_col]
    if rel_task.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION):
        df[rel_task.target_col] = pd.to_numeric(target, errors="coerce")
    elif rel_task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        codes, _ = pd.factorize(target, sort=True, use_na_sentinel=True)
        df[rel_task.target_col] = codes.astype(int)
    return df


def _impute_split_df_fallback(rel_task, split: str, db) -> pd.DataFrame:
    """Rebuild impute-task splits when redelex chokes on nullable target dtypes."""
    split_range = rel_task.make_split_range(db, split)
    entity_table = db.table_dict[rel_task.entity_table]
    entity_df = entity_table.df

    if _is_temporal_split_range(split_range):
        time_col = entity_table.time_col
        min_timestamp = split_range.min()
        max_timestamp = split_range.max()
        df = entity_df[
            (entity_df[time_col] >= min_timestamp) & (entity_df[time_col] <= max_timestamp)
        ].reset_index(drop=True)
        df = df[[rel_task.entity_col, time_col, rel_task.target_col]].copy()
    else:
        df = entity_df.loc[split_range, [rel_task.entity_col, rel_task.target_col]].reset_index(
            drop=True
        )

    return _coerce_impute_target(df, rel_task)


def _get_split_df(rel_task, split: str) -> pd.DataFrame:
    """
    Load a task split table from relbench/redelex.

    redelex impute tasks can fail when nullable pandas dtypes (NAType) are cast
    to float inside make_table; we rebuild those splits from the entity table.
    """
    try:
        return rel_task.get_table(split).df
    except KeyError:
        return rel_task._get_table(split).df
    except (TypeError, ValueError) as exc:
        if not (
            hasattr(rel_task, "entity_table")
            and hasattr(rel_task, "entity_col")
            and hasattr(rel_task, "target_col")
        ):
            raise exc

        db = rel_task.dataset.get_db(upto_test_timestamp=False)
        return _impute_split_df_fallback(rel_task, split, db)


def materialize_workspace_from_relbench(task: CTUTask, workdir: str | Path) -> Path:
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    _apply_redelex_compat_patches()

    workdir = Path(workdir)
    input_dir = workdir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    rel_task = get_task(task.dataset_name, task.task_name, download=False)
    dataset = get_dataset(task.dataset_name, download=False)

    for split in ("train", "val", "test"):
        split_table = _get_split_df(rel_task, split)
        _write_table(split_table, input_dir / f"{split}.parquet")

    db = dataset.get_db()
    db_tables = getattr(db, "table_dict", None) or getattr(db, "tables", {})
    if isinstance(db_tables, dict):
        for name, table in db_tables.items():
            table_df = getattr(table, "df", table)
            if isinstance(table_df, pd.DataFrame):
                _write_table(table_df, input_dir / "db_tables" / f"{name}.parquet")

    return input_dir


def materialize_workspace_from_hf(
    task: CTUTask,
    workdir: str | Path,
    *,
    hf_repo: str = "guilhermedrud/ctu_datasets",
    hf_revision: str = "main",
    hf_token: str | None = None,
    hf_data_prefix: str = "data",
) -> Path:
    from data.hf_utils import HF_KAGGLE_PREFIX, download_ctu_task_data, download_kaggle_task_data

    workdir = Path(workdir)
    input_dir = workdir / "input"
    if hf_data_prefix.strip("/") == HF_KAGGLE_PREFIX:
        return download_kaggle_task_data(
            task.row_name,
            input_dir,
            repo_id=hf_repo,
            revision=hf_revision,
            token=hf_token,
        )
    return download_ctu_task_data(
        task.row_name,
        input_dir,
        repo_id=hf_repo,
        revision=hf_revision,
        token=hf_token,
    )


def materialize_workspace_from_local(
    task: CTUTask,
    workdir: str | Path,
    *,
    local_root: str | Path,
) -> Path:
    """Copy pre-materialized parquets from a local tree (``<root>/<slug>/``)."""
    import shutil

    workdir = Path(workdir)
    input_dir = workdir / "input"
    source = Path(local_root) / task.row_name
    if not (source / "train.parquet").is_file():
        raise FileNotFoundError(f"No train.parquet under {source}")
    input_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        src = source / f"{split}.parquet"
        if src.is_file():
            shutil.copy2(src, input_dir / f"{split}.parquet")
    return input_dir


def materialize_workspace(
    task: CTUTask,
    workdir: str | Path,
    *,
    source: str = "hf",
    hf_repo: str = "guilhermedrud/ctu_datasets",
    hf_revision: str = "main",
    hf_token: str | None = None,
    hf_data_prefix: str = "data",
    local_root: str | Path | None = None,
) -> Path:
    if source == "hf":
        return materialize_workspace_from_hf(
            task,
            workdir,
            hf_repo=hf_repo,
            hf_revision=hf_revision,
            hf_token=hf_token,
            hf_data_prefix=hf_data_prefix,
        )
    if source == "local":
        if local_root is None:
            raise ValueError("local_root is required when source='local'")
        return materialize_workspace_from_local(task, workdir, local_root=local_root)
    if source in {"relbench", "mariadb"}:
        return materialize_workspace_from_relbench(task, workdir)
    raise ValueError(f"Unknown data source: {source!r} (use 'hf', 'local', or 'relbench')")


def build_aide_inputs(task: CTUTask, *, flat_tabular: bool = False) -> dict[str, str]:
    if flat_tabular or not task.target_table:
        goal = (
            f"Build a machine learning model for {task.task_type} "
            f"to predict `{task.target_column}` from the tabular features in train.parquet."
        )
    else:
        goal = (
            f"Build a machine learning model for {task.task_type} "
            f"to predict `{task.target_column}` from relational tables."
        )
    if task.description:
        goal += f" {task.description}"
    if task.task_type in {"binary_classification", "multiclass_classification"}:
        eval_metric = "Macro-F1 score on validation set"
    else:
        eval_metric = "MAE on validation set"
    return {"goal": goal, "eval": eval_metric}
