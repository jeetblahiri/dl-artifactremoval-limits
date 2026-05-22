#!/usr/bin/env bash
# Full pre-registered protocol (PREREGISTRATION.md §2).
# E1 × 5 seeds, E2 × 5 seeds, E3 × 5 seeds, E4 × 5 seeds,
# E4t × 1 seed, E5 × 1 seed, E6 × 1 seed,
# then consolidate + regenerate figures.
#
# Run under `caffeinate -i -s` so the laptop stays awake while plugged in.
# Closing the lid will still sleep the system on stock macOS configs.

set -e
set -o pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/results/logs"
mkdir -p "$LOG_DIR"

SEEDS="42 123 2024 7 31337"
echo "== Full protocol started $(date -u +%FT%TZ) =="
echo "Seeds: $SEEDS"

# ----- E0 (audit; idempotent, 1 seed) -----
if [ ! -f results/E0/seed_42/results.json ]; then
  python -m experiments.run E0 --seed 42 2>&1 | tee "$LOG_DIR/E0_seed_42.log"
fi

# ----- E1 / E2 / E3 — 5 seeds each -----
for EXP in E1 E2 E3; do
  for S in $SEEDS; do
    OUT="results/$EXP/seed_$S/results.json"
    if [ -f "$OUT" ]; then
      echo "[skip] $EXP seed=$S already exists"
      continue
    fi
    echo "[run]  $EXP seed=$S  start=$(date -u +%T)"
    python -m experiments.run "$EXP" --seed "$S" 2>&1 | tee "$LOG_DIR/${EXP}_seed_${S}.log"
  done
done

# ----- E4 (5 seeds; uses corrected KSG and per-alpha aggregator) -----
for S in $SEEDS; do
  OUT="results/E4/seed_$S/results.json"
  if [ -f "$OUT" ]; then
    echo "[skip] E4 seed=$S already exists"
    continue
  fi
  echo "[run]  E4 seed=$S  start=$(date -u +%T)"
  python -m experiments.run E4 --seed "$S" 2>&1 | tee "$LOG_DIR/E4_seed_${S}.log"
done

# ----- E4t (1 seed; D1' template-robustness sweep; inference-only) -----
if [ ! -f results/E4t/seed_42/results.json ]; then
  python -m experiments.run E4t --seed 42 2>&1 | tee "$LOG_DIR/E4t_seed_42.log"
fi

# ----- E5 (1 seed; 5 Sleep-EDF subjects; block-bootstrap inference) -----
if [ ! -f results/E5/seed_42/results.json ]; then
  python -m experiments.run E5 --seed 42 2>&1 | tee "$LOG_DIR/E5_seed_42.log"
fi

# ----- E6 (1 seed; wide IC-U-Net comparator for D_MMSE upper bound) -----
if [ ! -f results/E6/seed_42/results.json ]; then
  python -m experiments.run E6 --seed 42 2>&1 | tee "$LOG_DIR/E6_seed_42.log"
fi

# ----- Aggregate + render -----
echo "[aggregate] $(date -u +%T)"
python results/consolidate.py 2>&1 | tee "$LOG_DIR/consolidate.log"

echo "[figures]  $(date -u +%T)"
for FIG in figure1_identifiability figure2_wiener_floor figure3_dependent_mixing \
            figure4_capacity_scaling figure5_in_vivo_audit figure6_pool_audit; do
  python -m artifact_limits.viz.$FIG 2>&1 | tee -a "$LOG_DIR/figures.log"
done

echo "== Full protocol finished $(date -u +%FT%TZ) =="
