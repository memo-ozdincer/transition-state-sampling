#!/bin/bash
#SBATCH --job-name=nr_gad_path
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-node=0
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/nr_gad_path_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/nr_gad_path_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# Experiment 1: Hybrid NR→GAD with PATH-BASED trust region (all bells & whistles)
# =============================================================================
#
# iHiSD-inspired two-phase optimizer:
#   Phase 1: Full v13 NR minimization (RFO + PLS + kicks + late escape)
#            → converges to local minimum (n_neg=0, force<1e-4)
#   Phase 2: Path-based GAD with trust-region adaptation
#            → climbs from minimum to index-1 saddle (TS)
#
# Grid: 4 noise levels × 2 configs + 1 ablation = 9 combos
#
# | # | Config                   | NR Steps | GAD Steps | Noise     | Purpose              |
# |---|--------------------------|----------|-----------|-----------|----------------------|
# | 1 | Full v13 + path GAD      | 50k      | 20k       | all       | Main experiment      |
# | 2 | Full v13 + path GAD      | 50k      | 20k       | 2.0       | +gad-on-nr-failure   |
# | 3 | GAD-only (no NR)         | 0        | 20k       | 2.0       | GAD-only control     |
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
OUT_DIR="${SCRATCH_DIR}/runs/nr_gad_path_${SLURM_JOB_ID}"

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
echo "Hybrid NR→GAD — Experiment 1 (path-based)"
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
# Full v13 NR flags (same as v13 SLURM, all bells and whistles)
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
GAD_FLAGS="\
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
# Helper: run one hybrid config
# Usage: run_config <noise> <tag> <nr_steps> <gad_steps> <extra_flags...>
# =============================================================================
run_config() {
    local noise_level="$1"
    local tag="$2"
    local nr_steps="$3"
    local gad_steps="$4"
    shift 4
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
            $GAD_FLAGS \
            $extra_flags; then
        echo "  -> OK"
    else
        echo "  -> FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

# =============================================================================
# Config 1: Full v13 NR (50k) + path GAD (20k) — main experiment, all noise
# =============================================================================
for NOISE in "0.5" "1.0" "1.5" "2.0"; do
    run_config "$NOISE" "n${NOISE}_v13_gad_path_50k_20k" \
        "50000" "20000"
done

# =============================================================================
# Config 2: Same but with --gad-on-nr-failure at n=2.0
# Question: can GAD rescue NR failures by climbing from non-converged geometry?
# =============================================================================
run_config "2.0" "n2.0_v13_gad_path_50k_20k_rescue" \
    "50000" "20000" \
    --gad-on-nr-failure

# =============================================================================
# Config 3: GAD-only control at n=2.0 (NR 0 steps → GAD gets raw noisy geometry)
# =============================================================================
run_config "2.0" "n2.0_gad_only_20k" \
    "0" "20000"

# =============================================================================
# Config 4: Shorter NR (20k) + GAD at n=2.0
# Question: does NR step budget matter for subsequent GAD?
# =============================================================================
run_config "2.0" "n2.0_v13_gad_path_20k_20k" \
    "20000" "20000"

# =============================================================================
# Summary
# =============================================================================
echo "=============================================="
echo "Experiment 1 (path) completed at $(date)"
echo "  Total combinations: $COMBO"
echo "  Successful:         $((COMBO - FAILED))"
echo "  Failed:             $FAILED"
echo "  Noise seed:         $NOISE_SEED"
echo "  Results:            $OUT_DIR"
echo "=============================================="
