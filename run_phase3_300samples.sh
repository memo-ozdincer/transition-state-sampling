#!/bin/bash
# Phase 3 Investigation: 300-sample DFTB0 experiments
# 6 configurations: {control, forcegate, forcegate-nokicks} x {n=2.0, n=1.0}

set -euo pipefail

source /project/rrg-aspuru/memoozd/ts-tools/.venv/bin/activate
export PYTHONPATH="/project/rrg-aspuru/memoozd/ts-tools:$PYTHONPATH"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MPLBACKEND=Agg

OUTBASE="/scratch/memoozd/ts-tools-scratch/runs/phase3_300s"
H5="/project/rrg-aspuru/memoozd/data/transition1x.h5"
RUNNER="src/noisy/v2_tests/runners/run_minimization_parallel.py"
NWORKERS="${PHASE3_NWORKERS:-8}"
THREADS="${PHASE3_THREADS:-2}"
MAX_SAMPLES=300

# Common NR args
COMMON_NR="--method newton_raphson --n-steps 50000 --optimizer-mode rfo \
  --polynomial-linesearch --project-gradient-and-v \
  --trust-radius-floor 0.01 --max-atom-disp 1.3 --force-converged 1e-4 \
  --min-interatomic-dist 0.5 --log-spectrum-k 10 --relaxed-eval-threshold 0.01 \
  --accept-relaxed"

# Kick flags (for experiments WITH kicks)
KICK_FLAGS="--osc-kick --blind-kick --late-escape"

run_experiment() {
    local name=$1
    local noise=$2
    local extra_flags=$3

    local outdir="${OUTBASE}/${name}"
    mkdir -p "$outdir"

    echo "=========================================="
    echo "Starting: $name (noise=$noise)"
    echo "Output: $outdir"
    echo "Workers: $NWORKERS, Threads: $THREADS"
    echo "Time: $(date)"
    echo "=========================================="

    python "$RUNNER" \
        --h5-path "$H5" \
        --noise-seed 42 \
        --n-workers "$NWORKERS" --threads-per-worker "$THREADS" \
        --out-dir "$outdir" \
        --max-samples "$MAX_SAMPLES" \
        --start-from "midpoint_rt_noise${noise}A" \
        --scine-functional DFTB0 \
        $COMMON_NR \
        $extra_flags \
        2>&1 | tee "${outdir}/run.log"

    echo "Finished: $name at $(date)"
    echo ""
}

echo "Phase 3: 300-sample DFTB0 investigation"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

# 1. Controls (no force gate) — fast, ~2k steps each
run_experiment "control_n2.0" "2.0" "$KICK_FLAGS"
run_experiment "control_n1.0" "1.0" "$KICK_FLAGS"

# 2. Force-gate WITH kicks — 50k steps each
run_experiment "forcegate_kicks_n2.0" "2.0" "--strict-force-gate $KICK_FLAGS"
run_experiment "forcegate_kicks_n1.0" "1.0" "--strict-force-gate $KICK_FLAGS"

# 3. Force-gate WITHOUT kicks — 50k steps each
run_experiment "forcegate_nokicks_n2.0" "2.0" "--strict-force-gate"
run_experiment "forcegate_nokicks_n1.0" "1.0" "--strict-force-gate"

echo "All experiments complete at $(date)"
