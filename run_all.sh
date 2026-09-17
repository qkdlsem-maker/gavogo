#!/usr/bin/env bash
#
# GAVOGO — authoritative execution order for the full pipeline.
#
# 63 invocations in one pass:
#   Stage 1   28   four build steps x seven datasets
#   Stage 2   28   analysis scripts, in order
#   Stage 3    7   figure-only scripts
#
# Order matters. Several Stage 2 scripts consume intermediates written by earlier
# ones; running a script in isolation against a partially built results/ tree can
# produce values that differ from the released CSVs.
#
# Usage:   bash run_all.sh            run everything
#          bash run_all.sh --stage 2  run one stage only
#
# Runtime: roughly 5-6 hours on a single machine.

set -euo pipefail

STAGE="${2:-all}"
[[ "${1:-}" == "--stage" ]] || STAGE="all"

DATASETS=(highD NGSIM MiTra ETRI EMT uniD exiD)
LOG="run_all_$(date +%Y%m%d_%H%M%S).log"
N=0

say() { printf '\n=== [%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

run() {
  N=$((N + 1))
  say "[$N/63] $*"
  python "$@" 2>&1 | tee -a "$LOG"
}

# --- preflight -------------------------------------------------------------
# 19_baselines_all.py and 13_dl_latency.py exit early without torch and leave a
# stale CSV in place, which looks like a successful run. Fail loudly instead.
python - <<'PY'
import importlib, sys
missing = [m for m in ("torch", "lightgbm", "catboost", "shap")
           if importlib.util.find_spec(m) is None]
if missing:
    sys.exit("missing optional dependencies: " + ", ".join(missing) +
             "\ninstall them before a full run:  pip install " + " ".join(missing))
PY

say "start — logging to $LOG"

# --- Stage 1: per dataset (28) --------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "1" ]]; then
  for ds in "${DATASETS[@]}"; do
    run scripts/01_build_events.py        --dataset "$ds"
    run scripts/02_build_features.py      --dataset "$ds"
    run scripts/03_build_game_features.py --dataset "$ds"
    run scripts/03b_build_game_di.py      --dataset "$ds"
  done
else
  N=28
fi

# --- Stage 2: analysis, in order (28) -------------------------------------
STAGE2=(
  04_train.py
  14_adapter_ablation.py
  16_domain_invariant.py
  34_coral.py
  17_game_di_eval.py
  18_tiv_remaining.py
  19_baselines_all.py
  05_domain_adapt_v2.py
  20_adapt_control.py
  21_margin_sweep.py
  22_discard_audit.py
  23_ood_ci.py
  24_roadtype.py
  28_horizon1.py
  29_dataset_stats.py
  30_lodo7.py
  31_simple_baselines.py
  32_purity_auc.py
  33_domain_confusion.py
  diag_purity_all.py
  diag_shortcut.py
  26_leakage_quant.py
  27_conventional_leakage.py
  11_significance.py
  12_extras.py
  13_dl_latency.py
  15_loco.py
  07_lodo.py
)

if [[ "$STAGE" == "all" || "$STAGE" == "2" ]]; then
  for s in "${STAGE2[@]}"; do run "scripts/$s"; done
else
  N=56
fi

# --- Stage 3: figures (7) --------------------------------------------------
STAGE3=(
  fig_ieee.py
  fig_gamedi_scatter.py
  fig_shap_ieee.py
  fig_reliability_standalone.py
  08_shap.py
  09_case_study.py
  10_domain_shift.py
)

if [[ "$STAGE" == "all" || "$STAGE" == "3" ]]; then
  for s in "${STAGE3[@]}"; do run "scripts/$s"; done
fi

say "done — $N invocations, log in $LOG"
