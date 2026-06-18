#!/usr/bin/env bash
# Smoke-test OpenRLHF + Ray on GPU for the AIDE RLHF pipeline.
#
# Modes (env MODE):
#   gpu    - nvidia-smi + system torch only (no venv, no installs)
#   venv   - create .venv and verify torch inside it
#   check  - gpu + venv + install + imports + ray
#   sft    - minimal supervised fine-tuning
#   grpo   - minimal GRPO via Ray + train_ppo_ray
#   all    - check + sft + grpo (default)
#
# Usage:
#   MODE=gpu bash scripts/openrlhf/smoke_test.sh
#   MODE=check bash scripts/openrlhf/smoke_test.sh

set -e
cd "$(dirname "$0")/../.."

MODE="${MODE:-all}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/openrlhf_smoke}"
SMOKE_ROOT="${OUTPUT_DIR}"
VENV_DIR="${VENV_DIR:-/tmp/aide-openrlhf-venv}"
SMOKE_MODEL="${SMOKE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
NUM_GPUS="${NUM_GPUS:-1}"
RAY_IP="${RAY_IP:-127.0.0.1}"
INSTALL_OPENRLHF="${INSTALL_OPENRLHF:-1}"
HF_HOME="${HF_HOME:-${SMOKE_ROOT}/hf_cache}"
export HF_HOME

mkdir -p "${SMOKE_ROOT}"

_ok() { echo "[OK]   $*"; }
_fail() { echo "[FAIL] $*"; }
_info() { echo "[INFO] $*"; }
_warn() { echo "[WARN] $*"; }

_should_run() {
  case "${MODE}" in
    all)
      case "$1" in
        check|sft|grpo) return 0 ;;
      esac
      ;;
    "$1") return 0 ;;
  esac
  return 1
}

_stop_ray() {
  command -v ray >/dev/null 2>&1 && ray stop --force >/dev/null 2>&1 || true
}

trap _stop_ray EXIT

check_gpu_system() {
  _info "=== GPU / CUDA (system python) ==="
  command -v nvidia-smi >/dev/null 2>&1 || { _fail "nvidia-smi not found"; return 1; }
  nvidia-smi -L
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
  python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit(1)
PY
  _ok "GPU visible to system PyTorch"
}

_bootstrap_venv_pip() {
  local py="${VENV_DIR}/bin/python"
  if "${py}" -m pip --version >/dev/null 2>&1; then
    return 0
  fi
  _info "bootstrapping pip into venv (get-pip.py; apt mirrors blocked on cluster)"
  local getter="/tmp/get-pip.py"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "${getter}"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "${getter}" https://bootstrap.pypa.io/get-pip.py
  else
    _fail "need curl or wget to bootstrap pip"
    return 1
  fi
  "${py}" "${getter}"
  _ok "pip bootstrapped in venv"
}

setup_venv() {
  if [[ -n "${_VENV_READY:-}" ]]; then
    return 0
  fi
  _info "=== Python venv (${VENV_DIR}) ==="
  _info "python3: $(command -v python3) ($(python3 -V 2>&1))"

  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    rm -rf "${VENV_DIR}"
    # ensurepip needs python3.12-venv via apt, but cluster blocks archive.ubuntu.com.
    python3 -m venv --system-site-packages --without-pip "${VENV_DIR}"
  fi

  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  _bootstrap_venv_pip

  _VENV_READY=1
  _ok "venv: $(python -c 'import sys; print(sys.executable)')"
}

check_gpu_venv() {
  setup_venv
  _info "=== GPU / CUDA (venv python) ==="
  python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit(1)
PY
  _ok "GPU visible inside venv"
}

venv_python() {
  echo "${VENV_DIR}/bin/python"
}

install_openrlhf() {
  if [[ "${INSTALL_OPENRLHF}" != "1" ]]; then
    _info "Skipping OpenRLHF install (INSTALL_OPENRLHF=${INSTALL_OPENRLHF})"
    return 0
  fi
  setup_venv
  _info "=== Install OpenRLHF stack ==="
  python -m pip install -U pip wheel
  python -m pip install "vllm==0.19.1"
  python -m pip install "flash-attn" --no-build-isolation
  python -m pip install "openrlhf[vllm]" --no-build-isolation
  python -m pip install -e .
  _ok "OpenRLHF installed in venv"
}

check_imports() {
  setup_venv
  _info "=== Import check ==="
  python - <<'PY'
import openrlhf
import ray
import vllm
import torch
print("openrlhf:", getattr(openrlhf, "__version__", "unknown"))
print("ray:", ray.__version__)
print("vllm:", vllm.__version__)
print("torch cuda:", torch.cuda.is_available())
PY
  python -c "from aide.rlhf.grpo_reward_entrypoint import reward_func; print('reward_func ok')"
  _ok "Imports"
}

check_ray() {
  setup_venv
  _info "=== Ray head (ip=${RAY_IP}) ==="
  _stop_ray
  ray start \
    --head \
    --node-ip-address="${RAY_IP}" \
    --dashboard-host="${RAY_IP}" \
    --num-gpus="${NUM_GPUS}" \
    --disable-usage-stats
  ray status
  python - <<'PY'
import ray
ray.init(address="auto", ignore_reinit_error=True)
resources = ray.cluster_resources()
print("ray cluster resources:", resources)
assert resources.get("GPU", 0) >= 1, "Ray sees 0 GPUs"
PY
  _ok "Ray head running with GPU resources"
}

_run_sft() {
  setup_venv
  _info "=== SFT smoke (${SMOKE_MODEL}) ==="
  local out="${SMOKE_ROOT}/sft"
  mkdir -p "${out}"

  if python -m openrlhf.cli.train_sft --help 2>&1 | grep -q -- '--config'; then
    sed "s|output_dir:.*|output_dir: ${out}|; s|pretrain:.*|pretrain: ${SMOKE_MODEL}|" \
      configs/openrlhf/sft_smoke.yaml > "${SMOKE_ROOT}/sft_smoke_runtime.yaml"
    python -m openrlhf.cli.train_sft --config "${SMOKE_ROOT}/sft_smoke_runtime.yaml"
  else
    python -m openrlhf.cli.train_sft \
      --pretrain "${SMOKE_MODEL}" \
      --dataset data/openrlhf_smoke/sft.jsonl \
      --input_key messages \
      --apply_chat_template \
      --max_epochs 1 \
      --learning_rate 2e-5 \
      --batch_size 1 \
      --micro_train_batch_size 1 \
      --max_len 512 \
      --output_dir "${out}"
  fi
  _ok "SFT smoke finished -> ${out}"
}

_run_grpo() {
  setup_venv
  _info "=== GRPO smoke via Ray (${SMOKE_MODEL}) ==="
  local out="${SMOKE_ROOT}/grpo"
  local py
  py="$(venv_python)"
  mkdir -p "${out}"

  if ! ray status >/dev/null 2>&1; then
    check_ray
  fi

  local cfg="${SMOKE_ROOT}/grpo_smoke_runtime.yaml"
  sed "s|output_dir:.*|output_dir: ${out}|; s|pretrain:.*|pretrain: ${SMOKE_MODEL}|" \
    configs/openrlhf/grpo_singleturn_smoke.yaml > "${cfg}"

  if python -m openrlhf.cli.train_ppo_ray --help 2>&1 | grep -q -- '--config'; then
    ray job submit --address="http://${RAY_IP}:8265" \
      --working-dir "$(pwd)" \
      -- "${py}" -m openrlhf.cli.train_ppo_ray --config "${cfg}"
  else
    _warn "OpenRLHF build lacks --config; using explicit 1-GPU CLI flags"
    ray job submit --address="http://${RAY_IP}:8265" \
      --working-dir "$(pwd)" \
      -- "${py}" -m openrlhf.cli.train_ppo_ray \
      --pretrain "${SMOKE_MODEL}" \
      --dataset data/openrlhf_smoke/grpo_prompts.jsonl \
      --input_key messages \
      --apply_chat_template \
      --train.reward_func_path aide/rlhf/grpo_reward_entrypoint.py \
      --train.algorithm grpo \
      --max_epochs 1 \
      --train.batch_size 1 \
      --micro_train_batch_size 1 \
      --max_len 512 \
      --rollout.batch_size 2 \
      --rollout.n_samples_per_prompt 2 \
      --rollout.max_new_tokens 128 \
      --actor.num_nodes 1 --actor.num_gpus_per_node 1 \
      --ref.num_nodes 1 --ref.num_gpus_per_node 1 \
      --vllm.num_engines 1 --vllm.tensor_parallel_size 1 \
      --train.colocate_all \
      --vllm.gpu_memory_utilization 0.5 \
      --vllm.enforce_eager \
      --output_dir "${out}"
  fi
  _ok "GRPO smoke finished -> ${out}"
}

main() {
  _info "mode=${MODE} model=${SMOKE_MODEL} gpus=${NUM_GPUS}"
  _info "pwd=$(pwd) venv=${VENV_DIR} root=${SMOKE_ROOT}"

  if _should_run gpu; then
    check_gpu_system
  fi

  if _should_run venv; then
    check_gpu_system
    check_gpu_venv
  fi

  if _should_run check; then
    check_gpu_system
    install_openrlhf
    check_imports
    check_ray
  fi

  if _should_run sft; then
    install_openrlhf
    check_gpu_venv
    _run_sft
  fi

  if _should_run grpo; then
    install_openrlhf
    check_gpu_venv
    check_ray
    _run_grpo
  fi

  _ok "OpenRLHF smoke test passed (mode=${MODE})"
}

main "$@"
