import argparse

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aide.rlhf.exporter import export_logs_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", type=str, default="data/heuristic_runs/logs")
    parser.add_argument("--out", type=str, default="data/heuristic_runs/logs/offline_eval.csv")
    parser.add_argument("--ctu_csv", type=str, default="data/ctu_datasets_info.csv")
    parser.add_argument("--total_steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = export_logs_dir(
        args.logs_dir,
        args.out,
        ctu_csv=args.ctu_csv,
        total_steps=args.total_steps,
    )
    print(f"Exported {count} offline decision rows to {args.out}")


if __name__ == "__main__":
    main()

