#!/usr/bin/env python3
"""
Phase 1A Investigation: Force–Eigenvalue–n_neg Relationship
============================================================
Results-JSON-only analyses (fast, no trajectory loading).

Covers:
  1D. 2D threshold sweep (force × eigenvalue) → heatmaps
  1E. Force floor analysis → scatter plots
  1A'. Phase portraits (final state only) → scatter plots
  2A.  Eigenvalue distribution at convergence (LJ vs DFTB0)
  2B.  Force statistics at convergence (LJ vs DFTB0)
       Cascade comparison tables
       Summary comparison table

Input:  Results JSONs from LJ v2 grid and DFTB0 v13 runs.
Output: Text report, JSON summary, PNG plots.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

LJ_BASE  = Path("/scratch/memoozd/ts-tools-scratch/runs/min_nr_lj_v2_grid")
DFTB_BASE = Path("/scratch/memoozd/ts-tools-scratch/runs/min_nr_v13_1149428")

NOISE_LEVELS = ["n0.5", "n1.0", "n1.5", "n2.0"]

# Best-performing combo per noise level (from grid reports)
SELECTED_COMBO = "rfo_pls_both_kicks_adaptive_probe_late_50k"

# Threshold grids for the 2D sweep
FORCE_THRESHOLDS = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
EVAL_THRESHOLDS  = [1e-1, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 0.0]
# EVAL_THRESHOLDS: value τ means convergence requires λ_min ≥ -τ.
# 0.0 means strict: λ_min ≥ 0 (no negative eigenvalues).

# Built-in cascade thresholds (stored in the JSON)
CASCADE_THRESHOLDS = [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.008, 0.01]

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_results(base_dir, noise_level, combo_suffix):
    """Load results JSON for a given combo. Returns (results_list, combo_tag) or (None, tag)."""
    combo_tag = f"{noise_level}_{combo_suffix}"
    combo_dir = base_dir / combo_tag
    if not combo_dir.exists():
        return None, combo_tag
    results_files = list(combo_dir.glob("minimization_newton_raphson_*_results.json"))
    if not results_files:
        return None, combo_tag
    with open(results_files[0]) as f:
        data = json.load(f)
    return data["metrics"]["results"], combo_tag


def pct(arr, ps=[5, 25, 50, 75, 95]):
    """Return percentiles of array."""
    if len(arr) == 0:
        return {}
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}

# ═══════════════════════════════════════════════════════════════════════════
# Analysis 1D: 2D Threshold Sweep
# ═══════════════════════════════════════════════════════════════════════════

def threshold_sweep(results_list):
    """
    For each sample's FINAL state (regardless of convergence flag), check if it
    would pass a joint (force < F_thresh) AND (λ_min ≥ -E_thresh) criterion.
    Returns (heatmap[n_force, n_eval], n_valid).
    """
    forces, min_evals = [], []
    for r in results_list:
        fn = r.get("final_force_norm")
        me = r.get("final_min_vib_eval")
        if fn is not None and me is not None:
            forces.append(fn)
            min_evals.append(me)
    forces    = np.array(forces)
    min_evals = np.array(min_evals)
    n = len(forces)

    hm = np.zeros((len(FORCE_THRESHOLDS), len(EVAL_THRESHOLDS)))
    for i, ft in enumerate(FORCE_THRESHOLDS):
        f_ok = forces < ft
        for j, et in enumerate(EVAL_THRESHOLDS):
            e_ok = min_evals >= -et   # at et=0.0, requires λ_min ≥ 0
            hm[i, j] = np.sum(f_ok & e_ok) / n if n else 0.0
    return hm, n

# ═══════════════════════════════════════════════════════════════════════════
# Analysis 1E: Force Floor
# ═══════════════════════════════════════════════════════════════════════════

def force_floor(results_list):
    """Statistics on final force norm, split by strict/relaxed convergence."""
    out = {}
    def _stats(arr, prefix):
        a = np.array(arr)
        out[f"{prefix}_n"]      = len(a)
        out[f"{prefix}_min"]    = float(np.min(a))     if len(a) else None
        out[f"{prefix}_median"] = float(np.median(a))  if len(a) else None
        out[f"{prefix}_mean"]   = float(np.mean(a))    if len(a) else None
        out[f"{prefix}_p5"]     = float(np.percentile(a, 5))  if len(a) else None
        out[f"{prefix}_p95"]    = float(np.percentile(a, 95)) if len(a) else None
        out[f"{prefix}_max"]    = float(np.max(a))     if len(a) else None

    all_f    = [r["final_force_norm"] for r in results_list if r.get("final_force_norm") is not None]
    conv_f   = [r["final_force_norm"] for r in results_list if r.get("converged") and r.get("final_force_norm") is not None]
    strict_f = [r["final_force_norm"] for r in results_list
                if r.get("converged") and r.get("final_neg_vib", -1) == 0 and r.get("final_force_norm") is not None]
    relax_f  = [r["final_force_norm"] for r in results_list
                if r.get("converged") and r.get("final_neg_vib", 0) > 0 and r.get("final_force_norm") is not None]

    _stats(all_f, "all");  _stats(conv_f, "conv");  _stats(strict_f, "strict");  _stats(relax_f, "relax")

    # Force-nneg correlation among converged
    conv_fn = [r["final_force_norm"] for r in results_list if r.get("converged") and r.get("final_force_norm") is not None]
    conv_nn = [r.get("final_neg_vib", 0) for r in results_list if r.get("converged") and r.get("final_force_norm") is not None]
    if len(conv_fn) > 2 and np.std(conv_nn) > 0:
        out["corr_force_nneg"] = float(np.corrcoef(conv_fn, conv_nn)[0, 1])
    return out

# ═══════════════════════════════════════════════════════════════════════════
# Analysis 2A: Eigenvalue Distribution
# ═══════════════════════════════════════════════════════════════════════════

def eigenvalue_distribution(results_list):
    """Eigenvalue statistics among converged samples."""
    me_list, nn_list = [], []
    for r in results_list:
        if not r.get("converged"):
            continue
        me = r.get("final_min_vib_eval")
        if me is not None:
            me_list.append(me)
            nn_list.append(r.get("final_neg_vib", 0))

    me = np.array(me_list)
    nn = np.array(nn_list)
    out = {"n": len(me)}
    if len(me) == 0:
        return out

    out["lam_min_median"] = float(np.median(me))
    out["lam_min_mean"]   = float(np.mean(me))
    out["lam_min_min"]    = float(np.min(me))
    out["lam_min_max"]    = float(np.max(me))
    out["nneg_median"]    = float(np.median(nn))
    out["nneg_mean"]      = float(np.mean(nn))
    out["nneg_max"]       = int(np.max(nn))
    out["frac_strict"]    = float(np.mean(nn == 0))

    # Magnitude bands for negative eigenvalues
    neg = me[me < 0]
    if len(neg):
        absn = np.abs(neg)
        out["neg_frac"]          = float(len(neg) / len(me))
        out["neg_abs_median"]    = float(np.median(absn))
        out["neg_abs_p95"]       = float(np.percentile(absn, 95))
        out["neg_abs_max"]       = float(np.max(absn))
        out["neg_lt_1e-4"]       = float(np.mean(absn < 1e-4))
        out["neg_1e-4_to_1e-3"]  = float(np.mean((absn >= 1e-4) & (absn < 1e-3)))
        out["neg_1e-3_to_1e-2"]  = float(np.mean((absn >= 1e-3) & (absn < 1e-2)))
        out["neg_ge_1e-2"]       = float(np.mean(absn >= 1e-2))
    else:
        out["neg_frac"] = 0.0
    return out

# ═══════════════════════════════════════════════════════════════════════════
# Analysis: Cascade Comparison
# ═══════════════════════════════════════════════════════════════════════════

def cascade_rates(results_list):
    """Fraction of converged samples with n_neg=0 at each cascade threshold."""
    n_conv = 0
    counts = {str(t): 0 for t in CASCADE_THRESHOLDS}
    for r in results_list:
        if not r.get("converged"):
            continue
        n_conv += 1
        cas = r.get("cascade_at_convergence") or r.get("final_cascade", {})
        for t in CASCADE_THRESHOLDS:
            key = f"n_neg_at_{t}"
            val = cas.get(key)
            if val is None:
                val = cas.get(str(t))
            if val is not None and val == 0:
                counts[str(t)] += 1
    rates = {t: c / n_conv if n_conv else 0 for t, c in counts.items()}
    return {"n_conv": n_conv, "rates": rates}

# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def _eval_label(t):
    if t == 0.0:
        return "strict"
    if t >= 1e-1:
        return f"{t:.1f}"
    exp = int(np.floor(np.log10(t)))
    coeff = t / 10**exp
    if coeff == 1.0:
        return f"1e{exp}"
    return f"{coeff:.0f}e{exp}"


def plot_heatmaps(heatmaps_lj, heatmaps_dftb, output_dir):
    """2D threshold sweep heatmaps: LJ (top) vs DFTB0 (bottom), one column per noise."""
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), constrained_layout=True)
    fig.suptitle("Convergence Rate under Joint (Force, Eigenvalue) Threshold", fontsize=14)

    f_labels = [f"{t:.0e}" for t in FORCE_THRESHOLDS]
    e_labels = [_eval_label(t) for t in EVAL_THRESHOLDS]

    for col, noise in enumerate(NOISE_LEVELS):
        for row, (hms, pes) in enumerate([(heatmaps_lj, "LJ"), (heatmaps_dftb, "DFTB0")]):
            ax = axes[row, col]
            hm = hms.get(noise)
            if hm is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                continue
            im = ax.imshow(hm, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto",
                           interpolation="nearest")
            ax.set_xticks(range(len(EVAL_THRESHOLDS)))
            ax.set_xticklabels(e_labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(FORCE_THRESHOLDS)))
            ax.set_yticklabels(f_labels, fontsize=7)
            if col == 0:
                ax.set_ylabel(f"{pes}\nForce threshold")
            if row == 1:
                ax.set_xlabel("Eigenvalue threshold τ\n(require λ_min ≥ -τ)")
            ax.set_title(f"{noise}" if row == 0 else "")
            # annotate cells
            for i in range(hm.shape[0]):
                for j in range(hm.shape[1]):
                    v = hm[i, j]
                    c = "white" if v < 0.4 else "black"
                    ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=6, color=c)

    fig.colorbar(im, ax=axes, shrink=0.5, label="Convergence rate", pad=0.02)
    path = output_dir / "threshold_sweep_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_force_floor(all_results, output_dir):
    """Force floor: final F̄ vs n_neg, LJ and DFTB0 on same axes."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    fig.suptitle("Force Floor: Final F̄ vs n_neg (converged samples)", fontsize=13)

    for col, noise in enumerate(NOISE_LEVELS):
        ax = axes[col]
        for pes, color, marker in [("lj", "#2166ac", "o"), ("dftb", "#b2182b", "s")]:
            rl = all_results[pes].get(noise, [])
            conv = [r for r in rl if r.get("converged") and r.get("final_force_norm") is not None]
            if not conv:
                continue
            fn = [r["final_force_norm"] for r in conv]
            nn = [r.get("final_neg_vib", 0) for r in conv]
            ax.scatter(nn, fn, alpha=0.35, s=12, c=color, marker=marker,
                       label=pes.upper(), edgecolors="none")

        ax.set_yscale("log")
        ax.axhline(1e-4, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.set_xlabel("n_neg (final)")
        if col == 0:
            ax.set_ylabel("Final F̄")
        ax.set_title(noise)
        ax.legend(fontsize=8, markerscale=1.5)

    path = output_dir / "force_floor_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_phase_portraits(all_results, output_dir):
    """Final-state phase portraits: F̄ vs λ_min, color = n_neg."""
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), constrained_layout=True)
    fig.suptitle("Phase Portrait (final state): F̄  vs  λ_min   (colour = n_neg)", fontsize=14)

    for col, noise in enumerate(NOISE_LEVELS):
        for row, (pes, label) in enumerate([("lj", "LJ"), ("dftb", "DFTB0")]):
            ax = axes[row, col]
            rl = all_results[pes].get(noise, [])
            pts = [(r["final_force_norm"], r["final_min_vib_eval"], r.get("final_neg_vib", 0))
                   for r in rl
                   if r.get("final_force_norm") is not None and r.get("final_min_vib_eval") is not None]
            if not pts:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                continue
            fn, me, nn = zip(*pts)
            vmax = max(10, max(nn))
            sc = ax.scatter(me, fn, c=nn, cmap="viridis", alpha=0.4, s=12,
                            vmin=0, vmax=vmax, edgecolors="none")
            ax.set_yscale("log")
            ax.axhline(1e-4, color="red", ls="--", lw=0.8, alpha=0.5)
            ax.axvline(0,     color="gray", ls="-",  lw=0.6, alpha=0.3)
            ax.axvline(-0.01, color="orange", ls="--", lw=0.8, alpha=0.5)
            if col == 0:
                ax.set_ylabel(f"{label}\nFinal F̄")
            if row == 1:
                ax.set_xlabel("λ_min")
            if row == 0:
                ax.set_title(noise)
            plt.colorbar(sc, ax=ax, shrink=0.8, label="n_neg")

    path = output_dir / "phase_portrait_final.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_eigenvalue_histograms(all_results, output_dir):
    """Histogram of λ_min at convergence: LJ vs DFTB0."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    fig.suptitle("λ_min Distribution at Convergence", fontsize=13)

    for col, noise in enumerate(NOISE_LEVELS):
        ax = axes[col]
        for pes, color, label in [("lj", "#2166ac", "LJ"), ("dftb", "#b2182b", "DFTB0")]:
            rl = all_results[pes].get(noise, [])
            vals = [r["final_min_vib_eval"] for r in rl
                    if r.get("converged") and r.get("final_min_vib_eval") is not None]
            if vals:
                ax.hist(vals, bins=50, alpha=0.5, color=color, label=label, density=True)
        ax.axvline(0, color="gray", ls="-", lw=0.6)
        ax.axvline(-0.01, color="orange", ls="--", lw=0.8, label="τ=0.01")
        ax.set_xlabel("λ_min")
        if col == 0:
            ax.set_ylabel("Density")
        ax.set_title(noise)
        ax.legend(fontsize=8)

    path = output_dir / "eigenvalue_histograms.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════

class Report:
    """Accumulates lines and prints them; can also save to file."""
    def __init__(self):
        self.lines = []
    def __call__(self, line=""):
        self.lines.append(line)
        print(line)
    def save(self, path):
        with open(path, "w") as f:
            f.write("\n".join(self.lines) + "\n")

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    out = Path(__file__).resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    R = Report()

    R("=" * 90)
    R("PHASE 1A INVESTIGATION: Force–Eigenvalue–n_neg Relationship")
    R("Results-JSON analysis on LJ (v2 grid) and DFTB0 (v13)")
    R(f"Combo: *_{SELECTED_COMBO}")
    R("=" * 90)
    R()

    # ── Load ──────────────────────────────────────────────────────────────
    all_results = {"lj": {}, "dftb": {}}
    R("Loading results...")
    for noise in NOISE_LEVELS:
        for pes, base in [("lj", LJ_BASE), ("dftb", DFTB_BASE)]:
            rl, tag = load_results(base, noise, SELECTED_COMBO)
            if rl is not None:
                # Filter out errored samples
                valid = [r for r in rl if r.get("error") is None]
                all_results[pes][noise] = valid
                n_conv = sum(1 for r in valid if r.get("converged"))
                R(f"  {pes.upper():>5s} {tag}: {len(valid)} valid samples, {n_conv} converged ({n_conv/len(valid):.1%})")
            else:
                R(f"  {pes.upper():>5s} {tag}: NOT FOUND")
    R()

    # ══════════════════════════════════════════════════════════════════════
    #  1D. 2D THRESHOLD SWEEP
    # ══════════════════════════════════════════════════════════════════════
    R("─" * 90)
    R("ANALYSIS 1D: 2D Threshold Sweep")
    R("─" * 90)
    R("Convergence = (F̄ < F_thresh) AND (λ_min ≥ -τ).  Applied to ALL valid samples' final state.")
    R()

    heatmaps = {"lj": {}, "dftb": {}}
    e_hdr = " ".join(f"{'τ='+_eval_label(t):>9s}" for t in EVAL_THRESHOLDS)

    for pes in ["lj", "dftb"]:
        R(f"  {pes.upper()}:")
        R(f"  {'':>12s} {e_hdr}")
        for noise in NOISE_LEVELS:
            rl = all_results[pes].get(noise)
            if rl is None:
                continue
            hm, n = threshold_sweep(rl)
            heatmaps[pes][noise] = hm
            for i, ft in enumerate(FORCE_THRESHOLDS):
                row = " ".join(f"{hm[i,j]:>9.1%}" for j in range(len(EVAL_THRESHOLDS)))
                prefix = f"  {noise + ' F<' + f'{ft:.0e}':>12s}"
                R(f"{prefix} {row}")
            R()
        R()

    # ── Marginal analysis: force-only and eval-only ──
    R("  Marginal rates (force-only = rightmost column, eval-only = top row):")
    R()
    for pes in ["lj", "dftb"]:
        R(f"  {pes.upper()}:")
        for noise in NOISE_LEVELS:
            hm = heatmaps[pes].get(noise)
            if hm is None:
                continue
            # Force-only = column with τ=0.1 (most relaxed eval)
            force_only = hm[:, 0]  # τ=0.1
            # Eval-only = row with F<1e-2 (most relaxed force)
            eval_only  = hm[0, :]  # F<1e-2
            R(f"    {noise} force-only (τ=0.1):  " +
              "  ".join(f"F<{ft:.0e}={force_only[i]:.1%}" for i, ft in enumerate(FORCE_THRESHOLDS)))
            R(f"    {noise} eval-only  (F<1e-2): " +
              "  ".join(f"τ={_eval_label(et)}={eval_only[j]:.1%}" for j, et in enumerate(EVAL_THRESHOLDS)))
        R()

    # ══════════════════════════════════════════════════════════════════════
    #  1E. FORCE FLOOR
    # ══════════════════════════════════════════════════════════════════════
    R("─" * 90)
    R("ANALYSIS 1E: Force Floor")
    R("─" * 90)
    R()
    R(f"  {'':>8s} {'':>5s}  {'n':>5s}  {'min':>10s} {'p5':>10s} {'median':>10s} {'p95':>10s} {'max':>10s}  {'corr(F,nn)':>10s}")

    ff_data = {}
    for pes in ["lj", "dftb"]:
        for noise in NOISE_LEVELS:
            rl = all_results[pes].get(noise)
            if rl is None:
                continue
            ff = force_floor(rl)
            ff_data[(pes, noise)] = ff
            for cat in ["strict", "relax", "conv"]:
                n = ff.get(f"{cat}_n", 0)
                if n == 0:
                    R(f"  {pes.upper():>8s} {noise} {cat:>7s}  {0:>5d}  {'—':>10s} {'—':>10s} {'—':>10s} {'—':>10s} {'—':>10s}")
                    continue
                vals = [ff.get(f"{cat}_{k}", 0) for k in ["min", "p5", "median", "p95", "max"]]
                corr_str = f"{ff.get('corr_force_nneg', 0):.3f}" if cat == "conv" and "corr_force_nneg" in ff else ""
                R(f"  {pes.upper():>8s} {noise} {cat:>7s}  {n:>5d}  " +
                  "  ".join(f"{v:>10.2e}" for v in vals) + f"  {corr_str:>10s}")
        R()

    # ══════════════════════════════════════════════════════════════════════
    #  2A. EIGENVALUE DISTRIBUTION
    # ══════════════════════════════════════════════════════════════════════
    R("─" * 90)
    R("ANALYSIS 2A: Eigenvalue Distribution at Convergence")
    R("─" * 90)
    R()

    ev_data = {}
    for pes in ["lj", "dftb"]:
        R(f"  {pes.upper()}:")
        for noise in NOISE_LEVELS:
            rl = all_results[pes].get(noise)
            if rl is None:
                continue
            ev = eigenvalue_distribution(rl)
            ev_data[(pes, noise)] = ev
            R(f"    {noise}: n={ev['n']}, strict={ev.get('frac_strict',0):.1%}")
            R(f"      λ_min: median={ev.get('lam_min_median',0):.2e}, "
              f"min={ev.get('lam_min_min',0):.2e}, max={ev.get('lam_min_max',0):.2e}")
            R(f"      n_neg: median={ev.get('nneg_median',0):.1f}, "
              f"mean={ev.get('nneg_mean',0):.1f}, max={ev.get('nneg_max',0)}")
            nf = ev.get("neg_frac", 0)
            if nf > 0:
                R(f"      Neg eigenvalues ({nf:.1%} of samples): "
                  f"|λ| median={ev.get('neg_abs_median',0):.2e}, "
                  f"p95={ev.get('neg_abs_p95',0):.2e}, max={ev.get('neg_abs_max',0):.2e}")
                R(f"        <1e-4: {ev.get('neg_lt_1e-4',0):.1%}  "
                  f"1e-4–1e-3: {ev.get('neg_1e-4_to_1e-3',0):.1%}  "
                  f"1e-3–1e-2: {ev.get('neg_1e-3_to_1e-2',0):.1%}  "
                  f"≥1e-2: {ev.get('neg_ge_1e-2',0):.1%}")
            else:
                R(f"      No negative eigenvalues at convergence!")
        R()

    # ══════════════════════════════════════════════════════════════════════
    #  CASCADE COMPARISON
    # ══════════════════════════════════════════════════════════════════════
    R("─" * 90)
    R("ANALYSIS: Cascade Convergence Rates (n_neg=0 at threshold)")
    R("─" * 90)
    R()
    thr_hdr = " ".join(f"{t:>8.4f}" for t in CASCADE_THRESHOLDS)
    R(f"  {'':>12s}  {'n':>5s}  {thr_hdr}")
    for pes in ["lj", "dftb"]:
        R(f"  {pes.upper()}:")
        for noise in NOISE_LEVELS:
            rl = all_results[pes].get(noise)
            if rl is None:
                continue
            cr = cascade_rates(rl)
            row = " ".join(f"{cr['rates'].get(str(t), 0):>8.1%}" for t in CASCADE_THRESHOLDS)
            R(f"  {noise:>12s}  {cr['n_conv']:>5d}  {row}")
        R()

    # ══════════════════════════════════════════════════════════════════════
    #  SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════════
    R("─" * 90)
    R("SUMMARY COMPARISON: LJ vs DFTB0 (best combo per noise level)")
    R("─" * 90)
    R()

    header_cols = []
    for pes in ["LJ", "DFTB0"]:
        for noise in NOISE_LEVELS:
            header_cols.append(f"{pes} {noise}")
    R(f"  {'Metric':>28s}  " + "  ".join(f"{h:>12s}" for h in header_cols))
    R(f"  {'─'*28}  " + "  ".join("─"*12 for _ in header_cols))

    def _row(label, fn):
        vals = []
        for pes in ["lj", "dftb"]:
            for noise in NOISE_LEVELS:
                rl = all_results[pes].get(noise, [])
                vals.append(fn(rl))
        R(f"  {label:>28s}  " + "  ".join(f"{v:>12s}" for v in vals))

    _row("Samples (valid)",
         lambda rl: str(len(rl)) if rl else "—")
    _row("Conv. rate",
         lambda rl: f"{sum(1 for r in rl if r.get('converged'))/len(rl):.1%}" if rl else "—")
    _row("Strict rate",
         lambda rl: f"{sum(1 for r in rl if r.get('converged') and r.get('final_neg_vib',-1)==0)/len(rl):.1%}" if rl else "—")
    _row("Median F̄ (conv)",
         lambda rl: f"{np.median([r['final_force_norm'] for r in rl if r.get('converged') and r.get('final_force_norm') is not None]):.2e}" if any(r.get('converged') for r in rl) else "—")
    _row("Median λ_min (conv)",
         lambda rl: f"{np.median([r['final_min_vib_eval'] for r in rl if r.get('converged') and r.get('final_min_vib_eval') is not None]):.2e}" if any(r.get('converged') for r in rl) else "—")
    _row("Median n_neg (conv)",
         lambda rl: f"{np.median([r.get('final_neg_vib',0) for r in rl if r.get('converged')]):.0f}" if any(r.get('converged') for r in rl) else "—")

    R()

    # ══════════════════════════════════════════════════════════════════════
    #  KEY FINDINGS
    # ══════════════════════════════════════════════════════════════════════
    R("=" * 90)
    R("KEY FINDINGS")
    R("=" * 90)
    R()

    # Identify tightest threshold pair giving ≥90% convergence
    R("Tightest (force, eigenvalue) thresholds reaching ≥90% convergence rate:")
    for pes in ["lj", "dftb"]:
        R(f"  {pes.upper()}:")
        for noise in NOISE_LEVELS:
            hm = heatmaps[pes].get(noise)
            if hm is None:
                R(f"    {noise}: no data")
                continue
            # scan from tightest to most relaxed
            best_ft, best_et, best_rate = None, None, 0
            for i in range(len(FORCE_THRESHOLDS)-1, -1, -1):
                for j in range(len(EVAL_THRESHOLDS)-1, -1, -1):
                    if hm[i, j] >= 0.90:
                        ft, et = FORCE_THRESHOLDS[i], EVAL_THRESHOLDS[j]
                        # prefer tighter force, then tighter eval
                        if best_ft is None or ft < best_ft or (ft == best_ft and et < best_et):
                            best_ft, best_et, best_rate = ft, et, hm[i, j]
            if best_ft is not None:
                R(f"    {noise}: F<{best_ft:.0e}, τ={_eval_label(best_et)} → {best_rate:.1%}")
            else:
                R(f"    {noise}: NO combination reaches 90%")
    R()

    # Force-only convergence (no eigenvalue check at all)
    R("Force-only convergence (ignore eigenvalues entirely, τ=0.1):")
    for pes in ["lj", "dftb"]:
        R(f"  {pes.upper()}:")
        for noise in NOISE_LEVELS:
            hm = heatmaps[pes].get(noise)
            if hm is None:
                continue
            # τ=0.1 is column 0
            for i, ft in enumerate(FORCE_THRESHOLDS):
                if hm[i, 0] >= 0.01:  # only show nonzero
                    R(f"    {noise} F<{ft:.0e}: {hm[i,0]:.1%}")
        R()

    R("Strict convergence (n_neg=0, no force check):")
    for pes in ["lj", "dftb"]:
        R(f"  {pes.upper()}:")
        for noise in NOISE_LEVELS:
            hm = heatmaps[pes].get(noise)
            if hm is None:
                continue
            # strict = last column, F<1e-2 (most relaxed force) = first row
            strict_rate = hm[0, -1]  # F<1e-2 and strict
            R(f"    {noise}: {strict_rate:.1%}")
        R()

    # ══════════════════════════════════════════════════════════════════════
    #  PLOTS
    # ══════════════════════════════════════════════════════════════════════
    R("─" * 90)
    R("Generating plots...")
    R("─" * 90)
    R()

    p1 = plot_heatmaps(heatmaps["lj"], heatmaps["dftb"], out)
    R(f"  {p1}")
    p2 = plot_force_floor(all_results, out)
    R(f"  {p2}")
    p3 = plot_phase_portraits(all_results, out)
    R(f"  {p3}")
    p4 = plot_eigenvalue_histograms(all_results, out)
    R(f"  {p4}")
    R()

    # ── Save ──────────────────────────────────────────────────────────────
    R.save(out / "phase1a_report.txt")
    print(f"\nReport: {out / 'phase1a_report.txt'}")

    # JSON summary for downstream scripts
    summary = {
        "config": {
            "combo": SELECTED_COMBO,
            "force_thresholds": FORCE_THRESHOLDS,
            "eval_thresholds": EVAL_THRESHOLDS,
        },
        "heatmaps": {
            pes: {noise: hm.tolist() for noise, hm in hms.items()}
            for pes, hms in heatmaps.items()
        },
        "eigenvalue_distribution": {
            f"{pes}_{noise}": ev
            for (pes, noise), ev in ev_data.items()
        },
        "force_floor": {
            f"{pes}_{noise}": ff
            for (pes, noise), ff in ff_data.items()
        },
    }
    with open(out / "phase1a_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON:   {out / 'phase1a_summary.json'}")


if __name__ == "__main__":
    main()
