#!/usr/bin/env python3
"""
Phase 3 Results Analysis
========================
Compare control vs force-gated convergence on both LJ and DFTB0.

Run after phase3 experiments complete:
  python src/noisy/v2_tests/investigation/analyze_phase3_results.py

Output: text report + comparison plots
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASE3_BASE = Path("/scratch/memoozd/ts-tools-scratch/runs/phase3_investigation")
OUT = Path(__file__).resolve().parent / "phase3"


def load_results(tag):
    """Load results JSON for a given experiment tag."""
    exp_dir = PHASE3_BASE / tag
    if not exp_dir.exists():
        return None
    results_files = list(exp_dir.glob("minimization_newton_raphson_*_results.json"))
    if not results_files:
        return None
    with open(results_files[0]) as f:
        data = json.load(f)
    return data


def analyze_experiment(tag):
    """Extract key metrics from one experiment."""
    data = load_results(tag)
    if data is None:
        return None

    results = [r for r in data["metrics"]["results"] if r.get("error") is None]
    n = len(results)
    if n == 0:
        return None

    converged = [r for r in results if r.get("converged")]
    forces_conv = [r["final_force_norm"] for r in converged if r.get("final_force_norm") is not None]
    forces_all = [r["final_force_norm"] for r in results if r.get("final_force_norm") is not None]
    evals_conv = [r["final_min_vib_eval"] for r in converged if r.get("final_min_vib_eval") is not None]
    nnegs_conv = [r.get("final_neg_vib", 0) for r in converged]
    steps_conv = [r["converged_step"] for r in converged if r.get("converged_step") is not None]

    strict = [r for r in converged if r.get("final_neg_vib", -1) == 0]
    relaxed = [r for r in converged if r.get("final_neg_vib", 0) > 0]

    out = {
        "tag": tag,
        "n_total": n,
        "n_converged": len(converged),
        "conv_rate": len(converged) / n,
        "n_strict": len(strict),
        "n_relaxed": len(relaxed),
        "mean_steps": np.mean(steps_conv) if steps_conv else None,
    }

    if forces_conv:
        fc = np.array(forces_conv)
        out["force_conv_median"] = float(np.median(fc))
        out["force_conv_min"] = float(np.min(fc))
        out["force_conv_max"] = float(np.max(fc))
        out["force_conv_p5"] = float(np.percentile(fc, 5))
        out["force_conv_p95"] = float(np.percentile(fc, 95))

    if forces_all:
        fa = np.array(forces_all)
        out["force_all_median"] = float(np.median(fa))
        out["force_all_min"] = float(np.min(fa))

    if evals_conv:
        ec = np.array(evals_conv)
        out["eval_conv_median"] = float(np.median(ec))
        out["eval_conv_min"] = float(np.min(ec))

    if nnegs_conv:
        nc = np.array(nnegs_conv)
        out["nneg_conv_median"] = float(np.median(nc))
        out["nneg_conv_max"] = int(np.max(nc))

    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 3 RESULTS ANALYSIS")
    print("=" * 80)
    print()

    # Define experiment pairs (control, treatment)
    experiments = [
        # 3A: DFTB0 force gate
        ("3A_dftb_n2.0_control", "3A_dftb_n2.0_forcegate", "DFTB0 n=2.0"),
        ("3A_dftb_n1.0_control", "3A_dftb_n1.0_forcegate", "DFTB0 n=1.0"),
        # 3B: LJ force gate
        ("3B_lj_n2.0_control", "3B_lj_n2.0_forcegate", "LJ n=2.0"),
        ("3B_lj_n1.0_control", "3B_lj_n1.0_forcegate", "LJ n=1.0"),
        # 3C: LJ tight force
        ("3B_lj_n2.0_control", "3C_lj_n2.0_fc1e-5", "LJ n=2.0 fc=1e-5"),
        ("3B_lj_n2.0_control", "3C_lj_n2.0_fc1e-6", "LJ n=2.0 fc=1e-6"),
    ]

    for ctrl_tag, treat_tag, label in experiments:
        ctrl = analyze_experiment(ctrl_tag)
        treat = analyze_experiment(treat_tag)

        print(f"─── {label} ───")
        if ctrl is None:
            print(f"  Control ({ctrl_tag}): NOT FOUND")
        else:
            print(f"  Control ({ctrl_tag}):")
            print(f"    Conv rate: {ctrl['conv_rate']:.1%} ({ctrl['n_strict']} strict, {ctrl['n_relaxed']} relaxed)")
            if ctrl.get("force_conv_median"):
                print(f"    F̄ (conv): median={ctrl['force_conv_median']:.2e}, "
                      f"range=[{ctrl['force_conv_min']:.2e}, {ctrl['force_conv_max']:.2e}]")
            if ctrl.get("mean_steps"):
                print(f"    Mean steps: {ctrl['mean_steps']:.0f}")

        if treat is None:
            print(f"  Treatment ({treat_tag}): NOT FOUND")
        else:
            print(f"  Treatment ({treat_tag}):")
            print(f"    Conv rate: {treat['conv_rate']:.1%} ({treat['n_strict']} strict, {treat['n_relaxed']} relaxed)")
            if treat.get("force_conv_median"):
                print(f"    F̄ (conv): median={treat['force_conv_median']:.2e}, "
                      f"range=[{treat['force_conv_min']:.2e}, {treat['force_conv_max']:.2e}]")
            if treat.get("force_all_median"):
                print(f"    F̄ (all):  median={treat['force_all_median']:.2e}, min={treat['force_all_min']:.2e}")
            if treat.get("mean_steps"):
                print(f"    Mean steps: {treat['mean_steps']:.0f}")

        if ctrl and treat:
            delta_rate = treat["conv_rate"] - ctrl["conv_rate"]
            print(f"  Δ conv rate: {delta_rate:+.1%}")
            if treat.get("force_conv_median") and ctrl.get("force_conv_median"):
                ratio = treat["force_conv_median"] / ctrl["force_conv_median"]
                print(f"  Force ratio (treat/ctrl): {ratio:.2f}×")

        print()

    # Save summary
    summary = {}
    for ctrl_tag, treat_tag, label in experiments:
        ctrl = analyze_experiment(ctrl_tag)
        treat = analyze_experiment(treat_tag)
        summary[label] = {"control": ctrl, "treatment": treat}

    with open(OUT / "phase3_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved to {OUT / 'phase3_summary.json'}")


if __name__ == "__main__":
    main()
