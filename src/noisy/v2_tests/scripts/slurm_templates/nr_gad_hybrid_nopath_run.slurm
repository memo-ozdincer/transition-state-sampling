#!/bin/bash
#SBATCH --job-name=nr_gad_nopath
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-node=0
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/nr_gad_nopath_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/nr_gad_nopath_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# Experiment 2: Hybrid NR→GAD — NO PATH information (for generative modelling)
# =============================================================================
#
# Same iHiSD-inspired two-phase structure, but everything is Markovian:
#   Phase 1: Simplified v13 NR (RFO + PLS, NO kicks, fixed trust radius)
#            Every step decision is f(coords, E, F, Hessian) only.
#   Phase 2: State-based GAD (eigenvalue_clamped dt adaptation)
#            dt_eff = dt * scale / clamp(|λ₀|, 1e-2, 1e2)
#
# What's kept (state-based): RFO, PLS, Eckart projection, fixed trust floor
# What's dropped (history-based): all kicks, late escape, adaptive scaling,
#   trust radius growth/shrink
#
# Grid: 4 noise levels × 1 config + 2 ablations at n=2.0 = 6 combos
#
# | # | Config                             | NR Steps | GAD Steps | Noise | Purpose              |
# |---|------------------------------------|----------|-----------|-------|----------------------|
# | 1 | Simplified NR + nopath GAD         | 50k      | 20k       | all   | Main experiment      |
# | 2 | Simplified NR + nopath GAD         | 20k      | 20k       | 2.0   | NR budget ablation   |
# | 3 | GAD-only (no NR)                   | 0        | 20k       | 2.0   | GAD-only control     |
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
OUT_DIR="${SCRATCH_DIR}/runs/nr_gad_nopath_${SLURM_JOB_ID}"

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
echo "Hybrid NR→GAD — Experiment 2 (nopath, Markovian)"
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
# Simplified v13 NR flags — state-based only, no kicks/history
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
    --nr-accept-relaxed"
# NOTE: No --nr-osc-kick, --nr-blind-kick, --nr-late-escape, etc.
# All step decisions are purely state-based for generative modelling.

# =============================================================================
# Nopath GAD flags — state-based adaptive dt only
# =============================================================================
GAD_FLAGS="\
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
# Config 1: Simplified NR (50k) + nopath GAD (20k) — main experiment, all noise
# =============================================================================
for NOISE in "0.5" "1.0" "1.5" "2.0"; do
    run_config "$NOISE" "n${NOISE}_nr_simple_gad_nopath_50k_20k" \
        "50000" "20000"
done

# =============================================================================
# Config 2: Shorter NR (20k) + nopath GAD at n=2.0
# Question: does NR step budget matter for subsequent GAD?
# =============================================================================
run_config "2.0" "n2.0_nr_simple_gad_nopath_20k_20k" \
    "20000" "20000"

# =============================================================================
# Config 3: GAD-only control at n=2.0 (NR 0 steps → GAD gets raw noisy geometry)
# =============================================================================
run_config "2.0" "n2.0_gad_nopath_only_20k" \
    "0" "20000"

# =============================================================================
# Summary
# =============================================================================
echo "=============================================="
echo "Experiment 2 (nopath) completed at $(date)"
echo "  Total combinations: $COMBO"
echo "  Successful:         $((COMBO - FAILED))"
echo "  Failed:             $FAILED"
echo "  Noise seed:         $NOISE_SEED"
echo "  Results:            $OUT_DIR"
echo "=============================================="
