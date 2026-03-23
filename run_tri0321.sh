#!/bin/bash
source /project/rrg-aspuru/memoozd/ts-tools/.venv/bin/activate
cd /project/rrg-aspuru/memoozd/ts-tools
export PYTHONPATH="/project/rrg-aspuru/memoozd/ts-tools:$PYTHONPATH"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLBACKEND=Agg

N_WORKERS=48
THREADS=4
OUT_BASE="/scratch/memoozd/ts-tools-scratch/runs/nr_gad_tri0321"

NR_FULL="--nr-optimizer-mode rfo --nr-polynomial-linesearch --nr-project-gradient-and-v \
    --nr-trust-radius-floor 0.01 --nr-max-atom-disp 1.3 --nr-force-converged 1e-4 \
    --nr-min-interatomic-dist 0.5 --nr-log-spectrum-k 10 --nr-relaxed-eval-threshold 0.01 \
    --nr-accept-relaxed --nr-osc-kick --nr-osc-kick-patience 3 --nr-osc-kick-cooldown 50 \
    --nr-blind-kick --nr-blind-kick-overlap-thresh 0.1 --nr-blind-kick-force-thresh 0.1 \
    --nr-blind-kick-patience 100 --nr-adaptive-kick-scale --nr-adaptive-kick-C 0.1 \
    --nr-blind-kick-probe --nr-late-escape --nr-late-escape-after 15000 \
    --nr-late-escape-alpha 0.1 --nr-late-escape-cooldown 500"

NR_SIMPLE="--nr-optimizer-mode rfo --nr-polynomial-linesearch --nr-project-gradient-and-v \
    --nr-trust-radius-floor 0.01 --nr-max-atom-disp 1.3 --nr-force-converged 1e-4 \
    --nr-min-interatomic-dist 0.5 --nr-log-spectrum-k 10 --nr-relaxed-eval-threshold 0.01 \
    --nr-accept-relaxed"

GAD_PATH="--gad-variant path --gad-dt 0.02 --gad-dt-control adaptive --gad-dt-min 1e-6 \
    --gad-dt-max 0.08 --gad-max-atom-disp 0.35 --gad-min-interatomic-dist 0.5 \
    --gad-ts-eps 1e-5 --gad-tr-threshold 8e-3 --gad-project-gradient-and-v \
    --gad-projection-mode eckart_full --gad-log-spectrum-k 10"

GAD_NOPATH="--gad-variant nopath --gad-dt 0.02 --gad-dt-adaptation eigenvalue_clamped \
    --gad-dt-scale-factor 1e-1 --gad-dt-min 1e-6 --gad-dt-max 0.08 \
    --gad-max-atom-disp 0.35 --gad-min-interatomic-dist 0.5 --gad-ts-eps 1e-5 \
    --gad-tr-threshold 8e-3 --gad-project-gradient-and-v --gad-hessian-projection eckart_mw"

COMMON="--h5-path /project/rrg-aspuru/memoozd/data/transition1x.h5 \
    --scine-functional DFTB0 --noise-seed 42 --n-workers $N_WORKERS --threads-per-worker $THREADS"

run() {
    local tag="$1"; shift
    echo ""; echo "========== $tag =========="; echo ""
    mkdir -p "$OUT_BASE/$tag"
    python src/noisy/v2_tests/runners/run_nr_gad_hybrid_parallel.py \
        $COMMON --out-dir "$OUT_BASE/$tag" "$@"
    echo "Exit: $?"
}

# ---- Single-combo minimal tests (20 samples each) ----
echo "###############################################"
echo "# PHASE 1: Single-combo tests (20 samples)   #"
echo "###############################################"

# Exp 1: path GAD, full NR, n=1.0
run "exp1_path_n1.0_20s" --max-samples 20 --start-from midpoint_rt_noise1.0A \
    --nr-n-steps 50000 --gad-n-steps 20000 $NR_FULL $GAD_PATH

# Exp 2: nopath GAD, simple NR, n=1.0
run "exp2_nopath_n1.0_20s" --max-samples 20 --start-from midpoint_rt_noise1.0A \
    --nr-n-steps 50000 --gad-n-steps 20000 $NR_SIMPLE $GAD_NOPATH

# Exp 3: full v13 NR → path GAD, n=1.0
run "exp3_fullv13_path_n1.0_20s" --max-samples 20 --start-from midpoint_rt_noise1.0A \
    --nr-n-steps 50000 --gad-n-steps 20000 $NR_FULL $GAD_PATH

# Exp 3: full v13 NR → nopath GAD, n=1.0
run "exp3_fullv13_nopath_n1.0_20s" --max-samples 20 --start-from midpoint_rt_noise1.0A \
    --nr-n-steps 50000 --gad-n-steps 20000 $NR_FULL $GAD_NOPATH

echo ""
echo "###############################################"
echo "# PHASE 1 COMPLETE                            #"
echo "###############################################"
