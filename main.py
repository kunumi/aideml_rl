import argparse
from pathlib import Path

from aide import Experiment
from aide.rlhf.ctu_dataset import build_aide_inputs, load_ctu_index, materialize_workspace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--csv_path", type=str, default="data/ctu_datasets_info.csv")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--policy_kind", type=str, default="heuristic")
    parser.add_argument("--policy_model", type=str, default=None)
    parser.add_argument("--workdir", type=str, default="workspaces/main")
    return parser.parse_args()


def main():
    args = parse_args()
    tasks = load_ctu_index(args.csv_path)
    task = next((t for t in tasks if t.row_name == args.task_name), None)
    if task is None:
        raise ValueError(f"Task `{args.task_name}` not found in {args.csv_path}")

    workdir = Path(args.workdir) / task.row_name
    materialize_workspace(task, workdir)
    aide_inputs = build_aide_inputs(task)

    exp = Experiment(
        data_dir=str(workdir / "input"),
        goal=aide_inputs["goal"],
        eval=aide_inputs["eval"],
    )
    exp.cfg.agent.search.policy_kind = args.policy_kind
    if args.policy_model:
        exp.cfg.agent.search.policy_model = args.policy_model
    exp.agent.policy = exp._build_policy()

    sol = exp.run(steps=args.steps)
    print(f"Best validation metric: {sol.valid_metric}")


if __name__ == "__main__":
    main()