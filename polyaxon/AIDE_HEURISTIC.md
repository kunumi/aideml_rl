# AIDE heuristic runs on Polyaxon (Hugging Face)

CTU parquet tables and AIDE run logs live on Hugging Face dataset repo **`guilhermedrud/ctu_datasets`**:

| Path on HF | Contents |
|------------|----------|
| `data/<task_row_name>/` | `train.parquet`, `val.parquet`, `test.parquet`, `db_tables/*.parquet` |
| `data/ctu_datasets_info.csv` | Task index |
| `runs/<task_row_name>/seed<N>/` | AIDE logs (`journal.json`, `config.yaml`, `tree_plot.html`, `best_solution.py`) |

GCS upload is on standby (`--upload_gcs`); Polyaxon jobs use **`--data_source hf --upload_hf`**.

## 1) One-time: upload CTU data to Hugging Face

From a machine with relbench access (MariaDB / local cache):

```bash
pip install huggingface_hub relbench redelex pyarrow tqdm
export HF_TOKEN=<your-write-token>

python scripts/upload_ctu_to_hf.py
# or one task:
python scripts/upload_ctu_to_hf.py --task_name ctu-legalacts_legalacts-original
```

Resume upload (skip tasks already on HF):

```bash
python scripts/upload_ctu_to_hf.py --skip_uploaded
```

## 2) Verify HF access

**Locally:**

```bash
export HF_TOKEN=<token>   # required for write probe
python scripts/check_hf_connection.py
python scripts/check_hf_connection.py --skip-write   # read-only
```

**Polyaxon:**

```bash
polyaxon run -f polyaxon/plx_hf_check_gcp.yaml --upload -p <your-project>
```

Configure `HF_TOKEN` on the Polyaxon project for write access (log uploads).

**Code upload:** Component YAMLs include a `mount:` section (Polyaxon 2.13+) so the `uploads/` folder is created automatically. On older Polyaxon versions, add `--upload` and run from the **repo root**:

```bash
polyaxon run -f polyaxon/plx_hf_check_gcp.yaml --upload -p <your-project>
```

## 3) Single heuristic run (Polyaxon)

Default seed is **`auto`**: picks the next unused `runs/<task>/seed<N>/` on HF (e.g. if seed0 and seed1 exist, uses seed2).

```bash
# From repo root (mount: auto-uploads on Polyaxon 2.13+; add --upload on older versions)
polyaxon run -f polyaxon/plx_aide_heuristic_gcp.yaml -p <your-project> \
  -P task_index=40 -P seed=auto
```

Fixed seed: `-P seed=0`

## 4) Full CTU grid

```bash
polyaxon run -f polyaxon/plx_aide_heuristic_matrix.yaml -p <your-project>
```

## 4b) Kaggle grid

```bash
polyaxon run -f polyaxon/plx_aide_heuristic_kaggle_matrix.yaml -p <your-project>
```

## 5) Local run

```bash
python scripts/batch_run_heuristic.py \
  --task_name ctu-legalacts_legalacts-original \
  --seed auto \
  --data_source hf \
  --upload_hf
```

Backfill existing logs to HF:

```bash
python scripts/upload_heuristic_logs_to_hf.py --logs_dir data/heuristic_runs/logs
```

## Prerequisites

1. Polyaxon CLI ([GUIA.md](GUIA.md))
2. **`HF_TOKEN`** on Polyaxon project (write) for `--upload_hf`
3. **`OPENAI_API_KEY`** / Azure OpenAI env vars for AIDE LLM calls
4. CTU data uploaded to `guilhermedrud/ctu_datasets/data/` before cluster runs

## GCS (standby)

See `scripts/check_gcs_connection.py` and `--upload_gcs` if you switch back to GCS later.
