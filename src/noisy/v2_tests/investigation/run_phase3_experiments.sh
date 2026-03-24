#!/bin/bash
# =============================================================================
# Phase 3 Experiments: Force–Eigenvalue Investigation
# =============================================================================
#
# Purpose: Test what happens when we prevent premature strict convergence
#          (n_neg=0 with no force check) by adding a force gate.
#
# Background (Phase 1 findings):
#   - On DFTB0, n_neg=0 fires at median F̄ ~ 0.05-0.10 eV/Å (500-1000× above threshold)
#   - On LJ, n_neg=0 fires at median F̄ ~ 5-6e-3 (50× above threshold)
#   - Force and eigenvalue convergence NEVER co-occur on either PES
#   - Adding --strict-force-gate prevents stopping at n_neg=0 until forces also converge
#
# Code change:
#   minimization.py line 2321:
#     if strict_force_gate:
#         strict_converged = n_neg == 0 and force_norm < force_converged
#     else:
#         strict_converged = n_neg == 0
#
# Expected outcome:
#   - DFTB0: Optimizer continues past n_neg=0, forces may or may not reach 1e-4
#   - LJ: Strict never fires (since F̄ and n_neg=0 never co-occur), relaxed takes over
#
# Run on a reserved node:
#   salloc --nodes=1 --ntasks=1 --cpus-per-task=48 --time=4:00:00 --mem=128G
#   ssh <node>
#   cd /project/rrg-aspuru/memoozd/ts-tools
#   bash src/noisy/v2_tests/investigation/run_phase3_experiments.sh
# =============================================================================

set -euo pipefail
source /project/rrg-aspuru/memoozd/ts-tools/.venv/bin/activate
cd /project/rrg-aspuru/memoozd/ts-tools
export PYTHONPATH="/project/rrg-aspuru/memoozd/ts-tools:$PYTHONPATH"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLBACKEND=Agg

N_WORKERS=12           # fewer workers for targeted experiments
THREADS=4
OUT_BASE="/scratch/memoozd/ts-tools-scratch/runs/phase3_investigation"
H5="/project/rrg-aspuru/memoozd/data/transition1x.h5"
NOISE_SEED=42
MAX_SAMPLES=50         # targeted subset, not full grid

# Full v13 NR flags (same as production runs)
NR_BASE="--optimizer-mode rfo --polynomial-linesearch --project-gradient-and-v \
    --trust-radius-floor 0.01 --max-atom-disp 1.3 --force-converged 1e-4 \
    --min-interatomic-dist 0.5 --log-spectrum-k 10 --relaxed-eval-threshold 0.01 \
    --accept-relaxed --osc-kick --osc-kick-patience 3 --osc-kick-cooldown 50 \
    --blind-kick --blind-kick-overlap-thresh 0.1 --blind-kick-force-thresh 0.1 \
    --blind-kick-patience 100 --adaptive-kick-scale --adaptive-kick-C 0.1 \
    --blind-kick-probe --late-escape --late-escape-after 15000 \
    --late-escape-alpha 0.1 --late-escape-cooldown 500"

COMBO=0
FAILED=0

run() {
    local tag="$1"; shift
    COMBO=$((COMBO + 1))
    echo ""
    echo "========== [$COMBO] $tag =========="
    echo "Start: $(date)"
    mkdir -p "$OUT_BASE/$tag"
    if python src/noisy/v2_tests/runners/run_minimization_parallel.py \
        --h5-path "$H5" --noise-seed $NOISE_SEED \
        --n-workers $N_WORKERS --threads-per-worker $THREADS \
        --method newton_raphson \
        --out-dir "$OUT_BASE/$tag" "$@"; then
        echo "  -> OK ($(date))"
    else
        echo "  -> FAILED exit=$? ($(date))"
        FAILED=$((FAILED + 1))
    fi
}

echo "###############################################"
echo "# PHASE 3: Force–Eigenvalue Investigation     #"
echo "# $MAX_SAMPLES samples per experiment         #"
echo "###############################################"
echo "Started: $(date)"
echo ""

# =============================================================================
# Experiment 3A: Force-gated strict convergence on DFTB0
# Key question: Can DFTB0 forces reach 1e-4 if we don't stop at n_neg=0?
# =============================================================================
echo "=== Experiment 3A: DFTB0 with strict_force_gate ==="
for NOISE in 2.0 1.0; do
    # Control: standard convergence (no force gate)
    run "3A_dftb_n${NOISE}_control" \
        --n-steps 50000 --max-samples $MAX_SAMPLES \
        --start-from "midpoint_rt_noise${NOISE}A" \
        --scine-functional DFTB0 \
        $NR_BASE

    # Treatment: force-gated strict convergence
    run "3A_dftb_n${NOISE}_forcegate" \
        --n-steps 50000 --max-samples $MAX_SAMPLES \
        --start-from "midpoint_rt_noise${NOISE}A" \
        --scine-functional DFTB0 \
        --strict-force-gate \
        $NR_BASE
done

# =============================================================================
# Experiment 3B: Force-gated strict convergence on LJ
# Key question: Does LJ relaxed convergence improve when strict can't preempt?
# =============================================================================
echo "=== Experiment 3B: LJ with strict_force_gate ==="
for NOISE in 2.0 1.0; do
    # Control: standard convergence
    run "3B_lj_n${NOISE}_control" \
        --n-steps 50000 --max-samples $MAX_SAMPLES \
        --start-from "midpoint_rt_noise${NOISE}A" \
        --calculator lj --lj-sigma-scale 0.3333 \
        $NR_BASE

    # Treatment: force-gated strict convergence
    run "3B_lj_n${NOISE}_forcegate" \
        --n-steps 50000 --max-samples $MAX_SAMPLES \
        --start-from "midpoint_rt_noise${NOISE}A" \
        --calculator lj --lj-sigma-scale 0.3333 \
        --strict-force-gate \
        $NR_BASE
done

# =============================================================================
# Experiment 3C: Tighter force thresholds on LJ
# Key question: Can LJ forces go below 5e-5 (current floor)?
# =============================================================================
echo "=== Experiment 3C: LJ with tighter force thresholds ==="
for FC in 1e-5 1e-6; do
    run "3C_lj_n2.0_fc${FC}" \
        --n-steps 50000 --max-samples $MAX_SAMPLES \
        --start-from "midpoint_rt_noise2.0A" \
        --calculator lj --lj-sigma-scale 0.3333 \
        --optimizer-mode rfo --polynomial-linesearch --project-gradient-and-v \
        --trust-radius-floor 0.01 --max-atom-disp 1.3 --force-converged $FC \
        --min-interatomic-dist 0.5 --log-spectrum-k 10 --relaxed-eval-threshold 0.01 \
        --accept-relaxed --osc-kick --osc-kick-patience 3 --osc-kick-cooldown 50 \
        --blind-kick --blind-kick-overlap-thresh 0.1 --blind-kick-force-thresh 0.1 \
        --blind-kick-patience 100 --adaptive-kick-scale --adaptive-kick-C 0.1 \
        --blind-kick-probe --late-escape --late-escape-after 15000 \
        --late-escape-alpha 0.1 --late-escape-cooldown 500
done

echo ""
echo "###############################################"
echo "# PHASE 3 COMPLETE                            #"
echo "# Total: $COMBO experiments, $FAILED failures #"
echo "# Finished: $(date)                           #"
echo "# Results in: $OUT_BASE                       #"
echo "###############################################"
