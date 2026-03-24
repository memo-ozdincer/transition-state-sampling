#!/bin/bash
# Remaining experiments from the tri0279 session.
# Run on a reserved compute node: ssh <node> 'nohup bash /project/.../run_remaining.sh > /scratch/.../logs/remaining.log 2>&1 &'
source /project/rrg-aspuru/memoozd/ts-tools/.venv/bin/activate
cd /project/rrg-aspuru/memoozd/ts-tools
export PYTHONPATH="/project/rrg-aspuru/memoozd/ts-tools:$PYTHONPATH"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLBACKEND=Agg

N_WORKERS=48
THREADS=4
OUT_BASE="/scratch/memoozd/ts-tools-scratch/runs/nr_gad_remaining"
H5="/project/rrg-aspuru/memoozd/data/transition1x.h5"
NOISE_SEED=42

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

COMMON="--h5-path $H5 --scine-functional DFTB0 --noise-seed $NOISE_SEED \
    --n-workers $N_WORKERS --threads-per-worker $THREADS"

COMBO=0
FAILED=0

run() {
    local tag="$1"; shift
    COMBO=$((COMBO + 1))
    echo ""
    echo "========== [$COMBO] $tag =========="
    echo "Start: $(date)"
    mkdir -p "$OUT_BASE/$tag"
    if python src/noisy/v2_tests/runners/run_nr_gad_hybrid_parallel.py \
        $COMMON --out-dir "$OUT_BASE/$tag" "$@"; then
        echo "  -> OK ($(date))"
    else
        echo "  -> FAILED exit=$? ($(date))"
        FAILED=$((FAILED + 1))
    fi
}

echo "###############################################"
echo "# Remaining experiments                       #"
echo "###############################################"
echo "Started: $(date)"

# --- Exp 2 nopath: n=1.5 and n=2.0 (incomplete from tri0279) ---
for NOISE in 1.5 2.0; do
    run "exp2_nopath_n${NOISE}_300s" --max-samples 300 \
        --start-from "midpoint_rt_noise${NOISE}A" \
        --nr-n-steps 50000 --gad-n-steps 20000 $NR_SIMPLE $GAD_NOPATH
done

# --- Exp 2 ablation: shorter NR at n=2.0 ---
run "exp2_nopath_n2.0_nr20k_300s" --max-samples 300 \
    --start-from midpoint_rt_noise2.0A \
    --nr-n-steps 20000 --gad-n-steps 20000 $NR_SIMPLE $GAD_NOPATH

# --- Exp 3 cross: Full v13 NR + nopath GAD at all noise levels ---
for NOISE in 0.5 1.0 1.5 2.0; do
    run "exp3_fullnr_nopath_n${NOISE}_300s" --max-samples 300 \
        --start-from "midpoint_rt_noise${NOISE}A" \
        --nr-n-steps 50000 --gad-n-steps 20000 $NR_FULL $GAD_NOPATH
done

# --- GAD-only controls at n=2.0 ---
run "ctrl_gad_path_only_n2.0_300s" --max-samples 300 \
    --start-from midpoint_rt_noise2.0A \
    --nr-n-steps 0 --gad-n-steps 20000 --gad-on-nr-failure $NR_SIMPLE $GAD_PATH

run "ctrl_gad_nopath_only_n2.0_300s" --max-samples 300 \
    --start-from midpoint_rt_noise2.0A \
    --nr-n-steps 0 --gad-n-steps 20000 --gad-on-nr-failure $NR_SIMPLE $GAD_NOPATH

echo ""
echo "###############################################"
echo "# REMAINING COMPLETE                           #"
echo "# Total: $COMBO combos, $FAILED failures       #"
echo "# Finished: $(date)                             #"
echo "###############################################"
