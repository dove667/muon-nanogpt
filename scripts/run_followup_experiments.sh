#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_PATH="${DATA_PATH:-$ROOT/data/fineweb10B}"
RUN_SET="${RUN_SET:-all}"             # smoke | primary | confirm | all
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PREFIX="${RUN_PREFIX:-followup_$(date +%Y%m%d_%H%M%S)}"

BUDGET="${BUDGET:-100000000}"
SMOKE_BUDGET="${SMOKE_BUDGET:-2097152}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000000}"
EVAL_TOKENS="${EVAL_TOKENS:-524288}"
LOG_EVERY="${LOG_EVERY:-20}"
SPECTRAL_INTERVAL="${SPECTRAL_INTERVAL:-10000000}"
SPECTRAL_NUM_MATRICES="${SPECTRAL_NUM_MATRICES:-12}"
SPECTRAL_DIM_CAP="${SPECTRAL_DIM_CAP:-1024}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-540}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"

LOG_ROOT="$ROOT/experiment_logs/$RUN_PREFIX"
mkdir -p "$LOG_ROOT"
MANIFEST="$LOG_ROOT/manifest.csv"
MONITOR_LOG="$LOG_ROOT/monitor.log"

echo "run_prefix,mode,tag,orth,name,command" > "$MANIFEST"
{
  echo "[$(date -Is)] run_prefix=$RUN_PREFIX run_set=$RUN_SET data_path=$DATA_PATH budget=$BUDGET"
  nvidia-smi || true
  "$PYTHON_BIN" - <<'PY'
import torch, sys
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
} >> "$MONITOR_LOG" 2>&1

log_status() {
  local name="$1"
  local pid="$2"
  local logfile="$3"
  {
    echo "[$(date -Is)] name=$name pid=$pid"
    nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader || true
    df -h "$ROOT" "$DATA_PATH" 2>/dev/null || true
    if [[ -f "$logfile" ]]; then
      echo "--- tail $logfile ---"
      tail -n 12 "$logfile" || true
    fi
    echo
  } >> "$MONITOR_LOG" 2>&1
}

run_exp() {
  local mode="$1"
  local tag="$2"
  local orth="$3"
  shift 3

  local name="${RUN_PREFIX}_${mode}_${tag}"
  local logfile="$LOG_ROOT/${name}.log"
  local run_dir="$ROOT/runs/$name"
  if [[ -f "$run_dir/metrics.jsonl" ]] && grep -q '"status": "completed"' "$run_dir/metrics.jsonl"; then
    echo "[$(date -Is)] SKIP completed $name" | tee -a "$MONITOR_LOG"
    return 0
  fi
  if [[ -d "$run_dir" ]]; then
    echo "[$(date -Is)] REMOVE incomplete $name" | tee -a "$MONITOR_LOG"
    rm -rf "$run_dir"
  fi

  local cmd=(
    "$PYTHON_BIN" -m src.training.train
    --orth "$orth"
    --data-path "$DATA_PATH"
    --name "$name"
    --train-token-budget "$BUDGET"
    --eval-interval-tokens "$EVAL_INTERVAL"
    --eval-tokens "$EVAL_TOKENS"
    --log-every-steps "$LOG_EVERY"
  )
  if [[ "$mode" == "benchmark" ]]; then
    cmd+=(--benchmark)
  elif [[ "$mode" == "spectral" ]]; then
    cmd+=(
      --spectral
      --spectral-interval-tokens "$SPECTRAL_INTERVAL"
      --spectral-num-matrices "$SPECTRAL_NUM_MATRICES"
      --spectral-dim-cap "$SPECTRAL_DIM_CAP"
    )
  fi
  cmd+=("$@")

  printf '%s,%s,%s,%s,%s,"%s"\n' "$RUN_PREFIX" "$mode" "$tag" "$orth" "$name" "${cmd[*]}" >> "$MANIFEST"
  echo "[$(date -Is)] START $name" | tee -a "$MONITOR_LOG"
  "${cmd[@]}" > "$logfile" 2>&1 &
  local pid=$!
  local rc=0
  local last_monitor=0
  while kill -0 "$pid" 2>/dev/null; do
    local now
    now=$(date +%s)
    if (( now - last_monitor >= MONITOR_INTERVAL )); then
      log_status "$name" "$pid" "$logfile"
      last_monitor="$now"
    fi
    sleep "$POLL_INTERVAL"
  done
  wait "$pid" || rc=$?
  log_status "$name" "$pid" "$logfile"
  echo "[$(date -Is)] END $name rc=$rc" | tee -a "$MONITOR_LOG"
  if [[ "$rc" != "0" ]]; then
    exit "$rc"
  fi
}

run_smoke() {
  local old_budget="$BUDGET"
  local old_eval_interval="$EVAL_INTERVAL"
  BUDGET="$SMOKE_BUDGET"
  EVAL_INTERVAL=0
  run_exp train smoke_fast5 fast --ns-iterations 5 --seed 0
  BUDGET="$old_budget"
  EVAL_INTERVAL="$old_eval_interval"
}

run_primary() {
  "$PYTHON_BIN" scripts/plot_polynomial_maps.py

  run_exp train adamw adamw --seed 0
  run_exp train stable5 vanilla --ns-iterations 5 --seed 0
  run_exp train fast5 fast --ns-iterations 5 --seed 0
  run_exp train manual_T5_f3_s2 manual --ns-iterations 5 --fast-steps 3 --stable-steps 2 --seed 0
  run_exp train manual_T7_f4_s3 manual --ns-iterations 7 --fast-steps 4 --stable-steps 3 --seed 0
  run_exp train manual_T8_f5_s3 manual --ns-iterations 8 --fast-steps 5 --stable-steps 3 --seed 0
  run_exp train manual_T9_f9_s0 manual --ns-iterations 9 --fast-steps 9 --stable-steps 0 --seed 0
  run_exp train manual_T9_f5_s4 manual --ns-iterations 9 --fast-steps 5 --stable-steps 4 --seed 0
  run_exp train manual_T9_f4_s5 manual --ns-iterations 9 --fast-steps 4 --stable-steps 5 --seed 0
  run_exp train manual_T9_f3_s6 manual --ns-iterations 9 --fast-steps 3 --stable-steps 6 --seed 0
  run_exp train manual_T10_f5_s5 manual --ns-iterations 10 --fast-steps 5 --stable-steps 5 --seed 0

  run_exp train pe_T5_l3e-3 polar_express --pe-iterations 5 --pe-lower-bound 3e-3 --seed 0
  run_exp train pe_T5_l1e-3 polar_express --pe-iterations 5 --pe-lower-bound 1e-3 --seed 0
  run_exp train pe_T5_l3e-4 polar_express --pe-iterations 5 --pe-lower-bound 3e-4 --seed 0
  run_exp train pe_T5_l3e-5 polar_express --pe-iterations 5 --pe-lower-bound 3e-5 --seed 0
  run_exp train pe_T9_l3e-5 polar_express --pe-iterations 9 --pe-lower-bound 3e-5 --seed 0
  run_exp train pe_T10_l3e-5 polar_express --pe-iterations 10 --pe-lower-bound 3e-5 --seed 0

  run_exp spectral stable5 vanilla --ns-iterations 5 --seed 0
  run_exp spectral fast5 fast --ns-iterations 5 --seed 0
  run_exp spectral manual_T5_f3_s2 manual --ns-iterations 5 --fast-steps 3 --stable-steps 2 --seed 0
  run_exp spectral manual_T9_f4_s5 manual --ns-iterations 9 --fast-steps 4 --stable-steps 5 --seed 0
  run_exp spectral pe_T5_l1e-3 polar_express --pe-iterations 5 --pe-lower-bound 1e-3 --seed 0
  run_exp spectral pe_T5_l3e-5 polar_express --pe-iterations 5 --pe-lower-bound 3e-5 --seed 0
  run_exp spectral pe_T9_l3e-5 polar_express --pe-iterations 9 --pe-lower-bound 3e-5 --seed 0

  run_exp benchmark adamw adamw --seed 0
  run_exp benchmark stable5 vanilla --ns-iterations 5 --seed 0
  run_exp benchmark fast5 fast --ns-iterations 5 --seed 0
  run_exp benchmark manual_T5_f3_s2 manual --ns-iterations 5 --fast-steps 3 --stable-steps 2 --seed 0
  run_exp benchmark manual_T9_f4_s5 manual --ns-iterations 9 --fast-steps 4 --stable-steps 5 --seed 0
  run_exp benchmark pe_T5_l3e-5 polar_express --pe-iterations 5 --pe-lower-bound 3e-5 --seed 0
  run_exp benchmark pe_T9_l3e-5 polar_express --pe-iterations 9 --pe-lower-bound 3e-5 --seed 0
}

run_confirm() {
  run_exp train fast5_seed1 fast --ns-iterations 5 --seed 1
  run_exp train fast5_seed2 fast --ns-iterations 5 --seed 2
  run_exp train manual_T9_f4_s5_seed1 manual --ns-iterations 9 --fast-steps 4 --stable-steps 5 --seed 1
  run_exp train manual_T9_f4_s5_seed2 manual --ns-iterations 9 --fast-steps 4 --stable-steps 5 --seed 2
  run_exp train pe_T9_l3e-5_seed1 polar_express --pe-iterations 9 --pe-lower-bound 3e-5 --seed 1
  run_exp train pe_T9_l3e-5_seed2 polar_express --pe-iterations 9 --pe-lower-bound 3e-5 --seed 2

  run_exp train fast5_lr0.5 fast --ns-iterations 5 --lr-mul 0.5 --seed 0
  run_exp train fast5_lr2.0 fast --ns-iterations 5 --lr-mul 2.0 --seed 0
  run_exp train manual_T9_f4_s5_lr0.5 manual --ns-iterations 9 --fast-steps 4 --stable-steps 5 --lr-mul 0.5 --seed 0
  run_exp train manual_T9_f4_s5_lr2.0 manual --ns-iterations 9 --fast-steps 4 --stable-steps 5 --lr-mul 2.0 --seed 0
  run_exp train pe_T9_l3e-5_lr0.5 polar_express --pe-iterations 9 --pe-lower-bound 3e-5 --lr-mul 0.5 --seed 0
  run_exp train pe_T9_l3e-5_lr2.0 polar_express --pe-iterations 9 --pe-lower-bound 3e-5 --lr-mul 2.0 --seed 0
}

case "$RUN_SET" in
  smoke)
    run_smoke
    ;;
  primary)
    run_primary
    ;;
  confirm)
    run_confirm
    ;;
  all)
    run_smoke
    run_primary
    run_confirm
    ;;
  *)
    echo "Unknown RUN_SET=$RUN_SET" >&2
    exit 2
    ;;
esac

"$PYTHON_BIN" -m src.analysis.followup_report
"$PYTHON_BIN" -m src.analysis.plot_curves || true
"$PYTHON_BIN" -m src.analysis.summarize_runs || true
"$PYTHON_BIN" -m src.analysis.export_spectral_details || true

echo "[$(date -Is)] follow-up run set completed: $RUN_SET" | tee -a "$MONITOR_LOG"
