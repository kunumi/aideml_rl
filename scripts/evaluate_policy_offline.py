"""Offline CTU evaluation: heuristic vs LLM search policy (`--policy_kind llm` + `--policy_model` matching your Azure/vLLM deployment)."""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aide import Experiment
from aide.rlhf.ctu_dataset import build_aide_inputs, load_ctu_index, materialize_workspace
from aide.rlhf.evaluator import extract_baseline

from dotenv import load_dotenv
load_dotenv()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="data/ctu_datasets_info.csv")
    parser.add_argument("--out", type=str, default="data/offline_eval.csv")
    parser.add_argument("--max_tasks", type=int, default=10)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--policy_kind", type=str, default="heuristic")
    parser.add_argument("--policy_model", type=str, default=None)
    parser.add_argument("--workdir", type=str, default="workspaces/offline_eval")
    parser.add_argument("--data_source", type=str, default="hf", choices=["hf", "relbench"])
    parser.add_argument("--hf_repo", type=str, default="guilhermedrud/ctu_datasets")
    parser.add_argument("--hf_revision", type=str, default="main")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_ctu_index(args.csv_path)[: args.max_tasks]
    rows: list[dict] = []
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        task_dir = workdir / task.row_name
        baseline, maximize = extract_baseline(task.info, task.task_type)
        best_metric = None
        run_status = "ok"
        error_msg = None

        print(f"Task data: {task.info} \nBaseline: {baseline}, Maximize: {maximize}\n task_dir: {task_dir}, task_name: {task.row_name}")

        try:
            materialize_workspace(
                task,
                task_dir,
                source=args.data_source,
                hf_repo=args.hf_repo,
                hf_revision=args.hf_revision,
            )
            aide_inputs = build_aide_inputs(task)

            exp = Experiment(
                data_dir=str(task_dir / "input"),
                goal=aide_inputs["goal"],
                eval=aide_inputs["eval"],
            )
            exp.cfg.agent.search.policy_kind = args.policy_kind
            if args.policy_model:
                exp.cfg.agent.search.policy_model = args.policy_model
            exp.agent.policy = exp._build_policy()

            sol = exp.run(steps=args.steps)
            best_metric = sol.valid_metric
            if best_metric is None:
                run_status = "missing_metric"
        except Exception as exc:
            run_status = "error"
            error_msg = f"{type(exc).__name__}: {exc}"

        if best_metric is None:
            win = False
        else:
            win = (best_metric >= baseline) if maximize else (best_metric <= baseline)

        rows.append(
            {
                "task": task.row_name,
                "policy_kind": args.policy_kind,
                "best_metric": best_metric,
                "baseline_metric": baseline,
                "maximize": maximize,
                "win": bool(win),
                "status": run_status,
                "error": error_msg,
            }
        )

    out_df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out}")


if __name__ == "__main__":
    main()

