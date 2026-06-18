from pathlib import Path

import pandas as pd
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from tqdm import tqdm


def main() -> None:
    csv_path = Path(__file__).resolve().parent / "ctu_datasets_info.csv"
    ctu_datasets = pd.read_csv(csv_path)

    skip_datasets: list[str] = []
    for _, row in tqdm(ctu_datasets.iterrows(), total=len(ctu_datasets)):
        name_task = row["name"]
        name, task_name = name_task.split("_", 1)
        if name in skip_datasets:
            print(f"Skipping dataset: {name} - {task_name}")
            continue

        print(f"Downloading dataset and task: {name} - {task_name}")
        task = get_task(name, task_name, download=False)
        dataset = get_dataset(name, download=False)
        _ = dataset.get_db()

        _ = task.get_table("train").df
        _ = task.get_table("val").df
        _ = task.get_table("test").df


if __name__ == "__main__":
    main()