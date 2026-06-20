# Evaluation harness setup

## RelBench (native predictive tasks)

```bash
pip install -e ".[eval]"
# or: pip install relbench pyarrow matplotlib
```

RelBench caches datasets under `~/.cache/relbench` (override with `RELBENCH_CACHE_DIR`).

First run with download:

```bash
python scripts/run_eval.py --benchmark relbench --relbench_download --max_tasks 1 --steps 5
```

## MLE-bench (Lite competitions)

MLE-bench is installed separately from https://github.com/openai/mle-bench:

```bash
git clone https://github.com/openai/mle-bench
cd mle-bench
pip install -e .
```

Kaggle API credentials are required (`~/.kaggle/kaggle.json`). Prepare Lite tasks:

```bash
mlebench prepare -c nomad2018-predict-transparent-conductors
mlebench prepare -c spooky-author-identification
mlebench prepare -c detecting-insults-in-social-commentary
mlebench prepare -c leaf-classification
```

Or prepare all Lite competitions:

```bash
mlebench prepare --lite
```

Default data cache: `~/.cache/mle-bench/data` (override with `--mlebench_data_dir` on `run_eval.py`).

## Run evaluation

```bash
python scripts/run_eval.py --benchmark all --seeds 1 --steps 20
python scripts/plot_tokens_vs_metric.py
```

Results are appended to `data/eval/eval_results.csv`. Plots are written to `data/eval/plots/`.

## Task manifest

Edit `data/eval/eval_tasks.jsonl` to add or remove RelBench / MLE-bench tasks.
