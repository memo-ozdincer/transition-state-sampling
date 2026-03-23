#!/bin/bash
#SBATCH --job-name=nr_gad_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-node=0
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/nr_gad_full_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/nr_gad_full_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# Experiment 3: Full v13 NR (95%+ convergence) → GAD from converged minimum
# =============================================================================
#
# The "proper" two-phase approach:
#   Phase 1: Full v13 NR with ALL kicks & late escape, 50k steps
#            (achieves 97.9% at n=0.5, 96.9% at n=1.0, 96.5% at n=1.5,
#             93.0% at n=2.0)
#   Phase 2: GAD from the converged minimum → should have very high TS rate
#            since it starts from a real minimum with clean Hessian
#
# Most samples = "find the minimum, then find TS from the minimum."
# GAD only skipped if NR fails (which is <5% at n≤1.5).
#
# Both GAD variants tested to compare path vs nopath from high-quality minima.
#
# Grid: 4 noise levels × 2 GAD variants = 8 main combos
#       + 2 rescue combos at n=2.0 (where NR fails ~7%)
#
# | # | Config                         | NR Steps | GAD Steps | Noise | Purpose             |
# |---|--------------------------------|----------|-----------|-------|---------------------|
# | 1 | Full v13 + path GAD            | 50k      | 20k       | all   | Main (path)         |
# | 2 | Full v13 + nopath GAD          | 50k      | 20k       | all   | Main (nopath)       |
# | 3 | Full v13 + path GAD + rescue   | 50k      | 20k       | 2.0   | Rescue NR failures  |
# | 4 | Full v13 + nopath GAD + rescue | 50k      | 20k       | 2.0   | Rescue NR failures  |
#
# =============================================================================

mkdir -p /scratch/memoozd/ts-tools-scratch/logs

PROJECT_DIR="/project/rrg-aspuru/memoozd/ts-tools"
SCRATCH_DIR="/scratch/memoozd/ts-tools-scratch"

cd "$SCRATCH_DIR"

module purge
module load StdEnv/2023
module load python/3.11.5

source "$PROJECT_DIR/.venv/bin/activate"

# Environment setup
TMP_BASE="${SLURM_TMPDIR:-${SCRATCH_DIR}/tmp}"
export TMPDIR="${TMP_BASE}/${SLURM_JOB_ID}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="${SCRATCH_DIR}/.cache"
export MPLCONFIGDIR="${SCRATCH_DIR}/.config/matplotlib"
export FONTCONFIG_PATH=/etc/fonts
export FONTCONFIG_FILE=/etc/fonts/fonts.conf

mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

# Thread configuration
TOTAL_CPUS="${SLURM_CPUS_ON_NODE:-192}"
export OMP_NUM_THREADS="$TOTAL_CPUS"
export OPENBLAS_NUM_THREADS="$TOTAL_CPUS"
export MKL_NUM_THREADS="$TOTAL_CPUS"
export NUMEXPR_NUM_THREADS="$TOTAL_CPUS"
export VECLIB_MAXIMUM_THREADS="$TOTAL_CPUS"
export MPLBACKEND=Agg

# Data paths
H5_PATH="/project/rrg-aspuru/memoozd/data/transition1x.h5"
OUT_DIR="${SCRATCH_DIR}/runs/nr_gad_full_v13_${SLURM_JOB_ID}"

mkdir -p "$OUT_DIR"

# =============================================================================
# Fixed configuration
# =============================================================================
MAX_SAMPLES=300
SCINE_FUNCTIONAL="DFTB0"
NOISE_SEED="${SLURM_JOB_ID:-$(date +%s)}"

# Parallel worker config
THREADS_PER_WORKER="${THREADS_PER_WORKER:-4}"
N_WORKERS=$((TOTAL_CPUS / THREADS_PER_WORKER))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

echo "=============================================="
echo "Hybrid NR→GAD — Experiment 3 (full v13 → GAD)"
echo "=============================================="
echo "Date:         $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "CPUs:         $TOTAL_CPUS"
echo "Workers:      $N_WORKERS  (${THREADS_PER_WORKER} threads each)"
echo "=============================================="

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

COMBO=0
FAILED=0

# =============================================================================
# Full v13 NR flags — the 95%+ convergence config (all bells and whistles)
# =============================================================================
NR_FLAGS="\
    --nr-optimizer-mode rfo \
    --nr-polynomial-linesearch \
    --nr-project-gradient-and-v \
    --nr-trust-radius-floor 0.01 \
    --nr-max-atom-disp 1.3 \
    --nr-force-converged 1e-4 \
    --nr-min-interatomic-dist 0.5 \
    --nr-log-spectrum-k 10 \
    --nr-relaxed-eval-threshold 0.01 \
    --nr-accept-relaxed \
    --nr-osc-kick \
    --nr-osc-kick-patience 3 \
    --nr-osc-kick-cooldown 50 \
    --nr-blind-kick \
    --nr-blind-kick-overlap-thresh 0.1 \
    --nr-blind-kick-force-thresh 0.1 \
    --nr-blind-kick-patience 100 \
    --nr-adaptive-kick-scale \
    --nr-adaptive-kick-C 0.1 \
    --nr-blind-kick-probe \
    --nr-late-escape \
    --nr-late-escape-after 15000 \
    --nr-late-escape-alpha 0.1 \
    --nr-late-escape-cooldown 500"

# =============================================================================
# Path-based GAD flags
# =============================================================================
GAD_PATH_FLAGS="\
    --gad-variant path \
    --gad-dt 0.02 \
    --gad-dt-control adaptive \
    --gad-dt-min 1e-6 \
    --gad-dt-max 0.08 \
    --gad-max-atom-disp 0.35 \
    --gad-min-interatomic-dist 0.5 \
    --gad-ts-eps 1e-5 \
    --gad-tr-threshold 8e-3 \
    --gad-project-gradient-and-v \
    --gad-projection-mode eckart_full \
    --gad-log-spectrum-k 10"

# =============================================================================
# Nopath GAD flags
# =============================================================================
GAD_NOPATH_FLAGS="\
    --gad-variant nopath \
    --gad-dt 0.02 \
    --gad-dt-adaptation eigenvalue_clamped \
    --gad-dt-scale-factor 1e-1 \
    --gad-dt-min 1e-6 \
    --gad-dt-max 0.08 \
    --gad-max-atom-disp 0.35 \
    --gad-min-interatomic-dist 0.5 \
    --gad-ts-eps 1e-5 \
    --gad-tr-threshold 8e-3 \
    --gad-project-gradient-and-v \
    --gad-hessian-projection eckart_mw"

# =============================================================================
# Helper: run one hybrid config
# Usage: run_config <noise> <tag> <nr_steps> <gad_steps> <gad_flags> <extra...>
# =============================================================================
run_config() {
    local noise_level="$1"
    local tag="$2"
    local nr_steps="$3"
    local gad_steps="$4"
    local gad_flags="$5"
    shift 5
    local extra_flags="$*"

    local start_from="midpoint_rt_noise${noise_level}A"
    local run_out="${OUT_DIR}/${tag}"
    mkdir -p "$run_out"

    COMBO=$((COMBO + 1))
    echo "[combo ${COMBO}]  ${tag}  (noise=${noise_level}A, NR=${nr_steps}, GAD=${gad_steps})"

    if python "$PROJECT_DIR/src/noisy/v2_tests/runners/run_nr_gad_hybrid_parallel.py" \
            --h5-path "$H5_PATH" \
            --out-dir "$run_out" \
            --scine-functional "$SCINE_FUNCTIONAL" \
            --max-samples "$MAX_SAMPLES" \
            --start-from "$start_from" \
            --noise-seed "$NOISE_SEED" \
            --n-workers "$N_WORKERS" \
            --threads-per-worker "$THREADS_PER_WORKER" \
            --nr-n-steps "$nr_steps" \
            --gad-n-steps "$gad_steps" \
            $NR_FLAGS \
            $gad_flags \
            $extra_flags; then
        echo "  -> OK"
    else
        echo "  -> FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

# =============================================================================
# Config 1: Full v13 (50k) + path GAD (20k) — all noise levels
# Most samples will converge NR, then GAD climbs from the minimum.
# =============================================================================
echo "--- Config 1: Full v13 NR → path GAD (main experiment) ---"
for NOISE in "0.5" "1.0" "1.5" "2.0"; do
    run_config "$NOISE" "n${NOISE}_full_v13_gad_path" \
        "50000" "20000" "$GAD_PATH_FLAGS"
done

# =============================================================================
# Config 2: Full v13 (50k) + nopath GAD (20k) — all noise levels
# Same NR, but Markovian GAD variant for comparison.
# =============================================================================
echo "--- Config 2: Full v13 NR → nopath GAD (Markovian comparison) ---"
for NOISE in "0.5" "1.0" "1.5" "2.0"; do
    run_config "$NOISE" "n${NOISE}_full_v13_gad_nopath" \
        "50000" "20000" "$GAD_NOPATH_FLAGS"
done

# =============================================================================
# Config 3: Rescue mode at n=2.0 (where NR fails ~7%)
# Run GAD even when NR doesn't converge.
# =============================================================================
echo "--- Config 3: Full v13 NR → GAD + rescue at n=2.0 ---"
run_config "2.0" "n2.0_full_v13_gad_path_rescue" \
    "50000" "20000" "$GAD_PATH_FLAGS" \
    --gad-on-nr-failure

run_config "2.0" "n2.0_full_v13_gad_nopath_rescue" \
    "50000" "20000" "$GAD_NOPATH_FLAGS" \
    --gad-on-nr-failure

# =============================================================================
# Summary
# =============================================================================
echo "=============================================="
echo "Experiment 3 (full v13 → GAD) completed at $(date)"
echo "  Total combinations: $COMBO"
echo "  Successful:         $((COMBO - FAILED))"
echo "  Failed:             $FAILED"
echo "  Noise seed:         $NOISE_SEED"
echo "  Results:            $OUT_DIR"
echo "=============================================="
