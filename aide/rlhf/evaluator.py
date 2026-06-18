from typing import Any

from aide.journal import Journal


def _best_metric_value(journal: Journal) -> float | None:
    best = journal.get_best_node(only_good=True)
    if best is None or best.metric is None:
        return None
    return best.metric.value


def extract_baseline(task_info_json: dict[str, Any], task_type: str) -> tuple[float, bool]:
    if task_type in {"binary_classification", "multiclass_classification"}:
        return float(task_info_json.get("val_macro_f1", 0.0)), True
    if task_type == "regression":
        return float(task_info_json.get("val_mae", 0.0)), False
    return float(task_info_json.get("val_metric", 0.0)), True


def compute_step_reward(
    journal: Journal,
    baseline_metric: float,
    maximize: bool,
    prev_best: float | None,
    invalid_action: bool,
    dense: bool = True,
) -> float:
    reward = 0.0
    if dense:
        best_now = _best_metric_value(journal)
        if best_now is not None and prev_best is not None:
            delta = best_now - prev_best
            if not maximize:
                delta = -delta
            denom = max(abs(baseline_metric), 1e-8)
            reward += max(min(delta / denom, 1.0), -1.0)
    if invalid_action:
        reward -= 0.05
    return float(reward)


def compute_terminal_reward(
    journal: Journal,
    baseline_metric: float,
    maximize: bool,
) -> float:
    best_final = _best_metric_value(journal)
    if best_final is None:
        return -1.0

    margin = best_final - baseline_metric
    if not maximize:
        margin = -margin

    denom = max(abs(baseline_metric), 1e-8)
    reward = margin / denom

    if reward >= 0.10:
        reward += 1.0
    if all(n.is_buggy for n in journal.nodes):
        reward -= 1.0
    return float(reward)

