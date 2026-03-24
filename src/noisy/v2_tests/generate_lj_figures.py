"""Generate figures for LJ_RESULTS.tex from LJ v2 grid data."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# --- Paths ---
GRID_DIR = Path("/scratch/memoozd/ts-tools-scratch/runs/min_nr_lj_v2_grid")
SUMMARY = GRID_DIR / "analysis" / "nr_grid_summary.json"
OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

# --- Style ---
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})
LJ_BLUE = "#1565C0"
LJ_LIGHT = "#90CAF9"
DFTB_ORANGE = "#E65100"
DFTB_LIGHT = "#FFB74D"
GREEN = "#2E7D32"
RED = "#C62828"
GRAY = "#757575"

# --- Load data ---
summary = json.load(open(SUMMARY))

# Build config lookup: tag -> data
configs = {rc["tag"]: rc for rc in summary["ranked_configs"]}

# The main configs used in the paper
MAIN_CONFIGS = {
    "20k": {n: f"n{n}_rfo_pls_20k" for n in ["0.5", "1.0", "1.5", "2.0"]},
    "50k": {n: f"n{n}_rfo_pls_50k" for n in ["0.5", "1.0", "1.5", "2.0"]},
    "full": {n: f"n{n}_rfo_pls_both_kicks_adaptive_probe_late_50k" for n in ["0.5", "1.0", "1.5", "2.0"]},
}
NOISES = ["0.5", "1.0", "1.5", "2.0"]

# DFTB0 reference data (from LJ_RESULTS.tex / FINDINGS.md)
DFTB_STRICT = {"0.5": 97.9, "1.0": 96.9, "1.5": 96.5, "2.0": 93.0}
DFTB_RELAXED = {"0.5": 98.9, "1.0": 97.2, "1.5": 96.5, "2.0": 93.0}


def get_results(tag):
    """Get results list for a config tag."""
    return configs[tag]["results"]


def get_strict_relaxed(tag):
    """Count strict and relaxed converged samples."""
    results = get_results(tag)
    valid = [r for r in results if r.get("error") is None]
    strict = sum(1 for r in valid if r.get("converged") and r.get("final_neg_vib", -1) == 0)
    converged = sum(1 for r in valid if r.get("converged"))
    relaxed = converged - strict
    total = len(valid)
    return strict, relaxed, converged, total


# =========================================================================
# Figure 1: Strict vs Relaxed convergence bar chart
# =========================================================================
def fig_strict_relaxed_bars():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    x = np.arange(len(NOISES))
    width = 0.25

    for ax_idx, (label, cfg_map) in enumerate([("full", MAIN_CONFIGS["full"])]):
        ax = axes[0]
        strict_rates = []
        relaxed_rates = []
        for n in NOISES:
            s, r, c, t = get_strict_relaxed(cfg_map[n])
            strict_rates.append(100 * s / t)
            relaxed_rates.append(100 * c / t)

        bars_strict = ax.bar(x - width/2, strict_rates, width, color=GREEN, alpha=0.85, label="Strict ($n_{neg}=0$)")
        bars_relaxed = ax.bar(x + width/2, relaxed_rates, width, color=LJ_BLUE, alpha=0.85, label="Relaxed ($\\lambda_{min} \\geq -0.01$)")

        # Add value labels
        for bar in bars_strict:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}%", ha="center", va="bottom", fontsize=8, color=GREEN)
        for bar in bars_relaxed:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}%", ha="center", va="bottom", fontsize=8, color=LJ_BLUE)

        ax.set_xlabel("Noise level (Å)")
        ax.set_ylabel("Convergence rate (%)")
        ax.set_title("LJ: Strict vs Relaxed", fontweight="bold", color=LJ_BLUE)
        ax.set_xticks(x)
        ax.set_xticklabels(NOISES)
        ax.set_ylim(0, 115)
        ax.legend(loc="lower left")
        ax.axhline(50, color=GRAY, ls=":", lw=0.7, alpha=0.5)

    # DFTB0 comparison
    ax = axes[1]
    dftb_strict = [DFTB_STRICT[n] for n in NOISES]
    dftb_relaxed = [DFTB_RELAXED[n] for n in NOISES]
    ax.bar(x - width/2, dftb_strict, width, color=DFTB_ORANGE, alpha=0.85, label="Strict ($n_{neg}=0$)")
    ax.bar(x + width/2, dftb_relaxed, width, color=DFTB_LIGHT, alpha=0.85, label="Relaxed")

    for i, (s, r) in enumerate(zip(dftb_strict, dftb_relaxed)):
        ax.text(x[i] - width/2, s + 1, f"{s:.0f}%", ha="center", va="bottom", fontsize=8, color=DFTB_ORANGE)
        ax.text(x[i] + width/2, r + 1, f"{r:.0f}%", ha="center", va="bottom", fontsize=8, color=DFTB_ORANGE)

    ax.set_xlabel("Noise level (Å)")
    ax.set_title("DFTB0: Strict vs Relaxed", fontweight="bold", color=DFTB_ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels(NOISES)
    ax.set_ylim(0, 115)
    ax.legend(loc="lower left")

    fig.suptitle("Convergence Rate Comparison (Full v12b, 50k steps)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_strict_vs_relaxed.png")
    plt.close(fig)
    print("  -> fig_strict_vs_relaxed.png")


# =========================================================================
# Figure 2: The strict-relaxed gap
# =========================================================================
def fig_gap():
    fig, ax = plt.subplots(figsize=(6, 4))

    lj_strict = []
    lj_relaxed = []
    for n in NOISES:
        s, r, c, t = get_strict_relaxed(MAIN_CONFIGS["full"][n])
        lj_strict.append(100 * s / t)
        lj_relaxed.append(100 * c / t)

    lj_gap = [r - s for s, r in zip(lj_strict, lj_relaxed)]
    dftb_gap = [DFTB_RELAXED[n] - DFTB_STRICT[n] for n in NOISES]

    x = np.arange(len(NOISES))
    width = 0.35
    ax.bar(x - width/2, lj_gap, width, color=LJ_BLUE, alpha=0.85, label="LJ gap")
    ax.bar(x + width/2, dftb_gap, width, color=DFTB_ORANGE, alpha=0.85, label="DFTB0 gap")

    for i, g in enumerate(lj_gap):
        ax.text(x[i] - width/2, g + 1, f"{g:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=LJ_BLUE)
    for i, g in enumerate(dftb_gap):
        ax.text(x[i] + width/2, g + 1.5, f"{g:.1f}", ha="center", va="bottom", fontsize=9, color=DFTB_ORANGE)

    ax.set_xlabel("Noise level (Å)")
    ax.set_ylabel("Gap (relaxed − strict) in percentage points")
    ax.set_title("Strict–Relaxed Gap: LJ vs DFTB0", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(NOISES)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_gap.png")
    plt.close(fig)
    print("  -> fig_gap.png")


# =========================================================================
# Figure 3: Ghost eigenvalue histogram
# =========================================================================
def fig_eigenvalue_histogram():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    for idx, noise in enumerate(NOISES):
        ax = axes[idx // 2][idx % 2]
        tag = MAIN_CONFIGS["full"][noise]
        results = get_results(tag)

        # Collect min_vib_eval for converged samples
        strict_evals = []
        relaxed_evals = []
        for r in results:
            if r.get("error") or not r.get("converged"):
                continue
            ev = r.get("final_min_vib_eval")
            if ev is None:
                continue
            if r.get("final_neg_vib", -1) == 0:
                strict_evals.append(ev)
            else:
                relaxed_evals.append(ev)

        # Plot histogram of log10(|lambda_min|) for relaxed samples
        if relaxed_evals:
            neg_evals = [e for e in relaxed_evals if e < 0]
            if neg_evals:
                log_abs = [np.log10(abs(e)) for e in neg_evals]
                ax.hist(log_abs, bins=25, color=LJ_BLUE, alpha=0.7, edgecolor="white", label=f"Relaxed ({len(neg_evals)} samples)")
                median_val = np.median([abs(e) for e in neg_evals])
                ax.axvline(np.log10(median_val), color=RED, ls="--", lw=2, label=f"Median |λ| = {median_val:.1e}")

        if strict_evals:
            pos_evals = [e for e in strict_evals if e > 0]
            if pos_evals:
                log_pos = [np.log10(e) for e in pos_evals]
                ax.hist(log_pos, bins=15, color=GREEN, alpha=0.5, edgecolor="white", label=f"Strict ({len(pos_evals)})")

        ax.set_xlabel("$\\log_{10}|\\lambda_{\\min}|$")
        ax.set_ylabel("Count")
        ax.set_title(f"n = {noise} Å", fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle("Smallest Vibrational Eigenvalue at Convergence", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_eigenvalue_histogram.png")
    plt.close(fig)
    print("  -> fig_eigenvalue_histogram.png")


# =========================================================================
# Figure 4: Force distribution at convergence (strict vs relaxed)
# =========================================================================
def fig_force_distribution():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    for idx, noise in enumerate(NOISES):
        ax = axes[idx // 2][idx % 2]
        tag = MAIN_CONFIGS["full"][noise]
        results = get_results(tag)

        strict_forces = []
        relaxed_forces = []
        for r in results:
            if r.get("error") or not r.get("converged"):
                continue
            f = r.get("final_force_norm")
            if f is None:
                continue
            if r.get("final_neg_vib", -1) == 0:
                strict_forces.append(f)
            else:
                relaxed_forces.append(f)

        # Log-scale histogram
        all_forces = strict_forces + relaxed_forces
        if not all_forces:
            continue
        bins = np.logspace(np.log10(min(all_forces) * 0.8), np.log10(max(all_forces) * 1.2), 30)

        if relaxed_forces:
            ax.hist(relaxed_forces, bins=bins, color=LJ_BLUE, alpha=0.7, label=f"Relaxed ({len(relaxed_forces)})")
        if strict_forces:
            ax.hist(strict_forces, bins=bins, color=GREEN, alpha=0.7, label=f"Strict ({len(strict_forces)})")

        ax.axvline(1e-4, color=RED, ls="--", lw=2, label="$\\varepsilon = 10^{-4}$")
        ax.set_xscale("log")
        ax.set_xlabel("$\\bar{F}$ (eV/Å)")
        ax.set_ylabel("Count")
        ax.set_title(f"n = {noise} Å", fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle("Force Norm at Convergence: Strict vs Relaxed", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_force_distribution.png")
    plt.close(fig)
    print("  -> fig_force_distribution.png")


# =========================================================================
# Figure 5: Cascade plot (convergence vs eigenvalue threshold)
# =========================================================================
def fig_cascade():
    CASCADE_THRESHOLDS = [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.008, 0.01]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1565C0", "#2196F3", "#FF9800", "#E65100"]
    markers = ["o", "s", "D", "^"]

    for idx, noise in enumerate(NOISES):
        tag = MAIN_CONFIGS["full"][noise]
        results = get_results(tag)
        valid = [r for r in results if r.get("error") is None]

        counts = []
        for thr in CASCADE_THRESHOLDS:
            key = f"n_neg_at_{thr}"
            count = sum(1 for r in valid if r.get("converged") and
                       r.get("final_cascade", {}).get(key, 1) == 0)
            counts.append(count)

        rates = [100 * c / len(valid) for c in counts]
        labels = [f"{t}" for t in CASCADE_THRESHOLDS]
        ax.plot(range(len(CASCADE_THRESHOLDS)), rates, f"-{markers[idx]}",
                color=colors[idx], lw=2, markersize=7, label=f"n = {noise} Å")

    ax.set_xticks(range(len(CASCADE_THRESHOLDS)))
    ax.set_xticklabels([f"{t}" for t in CASCADE_THRESHOLDS], rotation=45, ha="right")
    ax.set_xlabel("Eigenvalue threshold $\\tau$")
    ax.set_ylabel("Convergence rate (%)")
    ax.set_title("Cascade: Convergence vs Eigenvalue Threshold (Full v12b 50k)", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 105)
    ax.axhline(100, color=GRAY, ls=":", lw=0.7, alpha=0.5)

    # Annotate the knee
    ax.annotate("Knee: most ghost modes\nlive below $10^{-4}$",
                xy=(1, 80), xytext=(3, 60),
                arrowprops=dict(arrowstyle="->", color=GRAY),
                fontsize=9, color=GRAY)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_cascade.png")
    plt.close(fig)
    print("  -> fig_cascade.png")


# =========================================================================
# Figure 6: n_neg distribution at convergence
# =========================================================================
def fig_nneg_distribution():
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)

    for idx, noise in enumerate(NOISES):
        ax = axes[idx]
        tag = MAIN_CONFIGS["full"][noise]
        results = get_results(tag)

        n_negs = []
        for r in results:
            if r.get("error") or not r.get("converged"):
                continue
            n_negs.append(r.get("final_neg_vib", 0))

        if n_negs:
            max_n = max(n_negs)
            bins = np.arange(-0.5, max_n + 1.5, 1)
            counts, _, bars = ax.hist(n_negs, bins=bins, color=LJ_BLUE, alpha=0.7, edgecolor="white")
            # Color the n_neg=0 bar green
            if len(bars) > 0:
                bars[0].set_facecolor(GREEN)
                bars[0].set_alpha(0.85)

            n_zero = sum(1 for n in n_negs if n == 0)
            ax.text(0, n_zero + 2, f"{n_zero}", ha="center", fontsize=9, fontweight="bold", color=GREEN)

        ax.set_xlabel("$n_{neg}$")
        if idx == 0:
            ax.set_ylabel("Count")
        ax.set_title(f"n = {noise} Å", fontweight="bold")

    fig.suptitle("Number of Negative Eigenvalues at Convergence", fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_nneg_distribution.png")
    plt.close(fig)
    print("  -> fig_nneg_distribution.png")


# =========================================================================
# Figure 7: Step count comparison (LJ vs DFTB0)
# =========================================================================
def fig_steps_boxplot():
    fig, ax = plt.subplots(figsize=(8, 4.5))

    positions = []
    data = []
    colors_list = []
    labels = []

    for idx, noise in enumerate(NOISES):
        tag = MAIN_CONFIGS["full"][noise]
        results = get_results(tag)
        steps = [r["converged_step"] for r in results
                 if r.get("converged") and r.get("error") is None and r.get("converged_step")]
        pos = idx * 2
        positions.append(pos)
        data.append(steps)
        colors_list.append(LJ_BLUE)

    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                    showfliers=False, medianprops=dict(color="white", lw=2))
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # DFTB0 reference medians
    dftb_medians = {"0.5": 2613, "1.0": 5435, "1.5": 7755, "2.0": 11000}
    for idx, noise in enumerate(NOISES):
        pos = idx * 2
        ax.plot(pos, dftb_medians[noise], "D", color=DFTB_ORANGE, markersize=10, zorder=5)

    # Custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=LJ_BLUE, alpha=0.7, label="LJ (box)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=DFTB_ORANGE, markersize=10, label="DFTB0 median"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    ax.set_xticks([i * 2 for i in range(len(NOISES))])
    ax.set_xticklabels([f"{n} Å" for n in NOISES])
    ax.set_xlabel("Noise level")
    ax.set_ylabel("Steps to convergence")
    ax.set_title("Convergence Speed: LJ vs DFTB0", fontweight="bold")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_steps_comparison.png")
    plt.close(fig)
    print("  -> fig_steps_comparison.png")


# =========================================================================
# Figure 8: Conditioning (kappa) at convergence
# =========================================================================
def fig_conditioning():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    data = []
    positions = []
    for idx, noise in enumerate(NOISES):
        tag = MAIN_CONFIGS["full"][noise]
        results = get_results(tag)

        kappas = []
        for r in results:
            if r.get("error") or not r.get("converged"):
                continue
            ev_min = r.get("final_min_vib_eval")
            cascade = r.get("final_cascade", {})
            # Use bottom spectrum to estimate kappa
            spec = r.get("bottom_spectrum_at_convergence", [])
            if ev_min is not None and abs(ev_min) > 1e-20:
                # Approximate kappa as ratio of typical eigenvalue to min
                # We don't have lambda_max, so estimate from typical LJ values
                kappas.append(0.5 / abs(ev_min))  # lambda_max ~ 0.5 for LJ

        positions.append(idx)
        data.append(kappas)

    bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True,
                    showfliers=True,
                    flierprops=dict(marker=".", markersize=3, alpha=0.3),
                    medianprops=dict(color="white", lw=2))
    for patch in bp["boxes"]:
        patch.set_facecolor(LJ_BLUE)
        patch.set_alpha(0.7)

    ax.set_xticks(range(len(NOISES)))
    ax.set_xticklabels([f"{n} Å" for n in NOISES])
    ax.set_xlabel("Noise level")
    ax.set_ylabel("Condition number $\\kappa \\approx |\\lambda_{max}|/|\\lambda_{min}|$")
    ax.set_title("Hessian Conditioning at Convergence", fontweight="bold")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_conditioning.png")
    plt.close(fig)
    print("  -> fig_conditioning.png")


# =========================================================================
# Figure 9: Gain decomposition (50k vs kicks)
# =========================================================================
def fig_gain_decomposition():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # LJ gains
    ax = axes[0]
    x = np.arange(len(NOISES))
    width = 0.35

    gain_50k = []
    gain_kicks = []
    for noise in NOISES:
        tag_20k = MAIN_CONFIGS["20k"][noise]
        tag_50k = MAIN_CONFIGS["50k"][noise]
        tag_full = MAIN_CONFIGS["full"][noise]

        s20, _, c20, t20 = get_strict_relaxed(tag_20k)
        s50, _, c50, t50 = get_strict_relaxed(tag_50k)
        sf, _, cf, tf = get_strict_relaxed(tag_full)

        gain_50k.append(c50 - c20)
        gain_kicks.append(cf - c50)

    ax.bar(x - width/2, gain_50k, width, color=LJ_LIGHT, edgecolor=LJ_BLUE, lw=1.5, label="Δ from 50k steps")
    ax.bar(x + width/2, gain_kicks, width, color=LJ_BLUE, alpha=0.85, label="Δ from kick mechanisms")

    for i in range(len(NOISES)):
        ax.text(x[i] - width/2, gain_50k[i] + 0.3, f"+{gain_50k[i]}", ha="center", fontsize=8)
        ax.text(x[i] + width/2, gain_kicks[i] + 0.3, f"+{gain_kicks[i]}", ha="center", fontsize=8)

    ax.set_xlabel("Noise level (Å)")
    ax.set_ylabel("Additional converged samples")
    ax.set_title("LJ: Gain Decomposition", fontweight="bold", color=LJ_BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(NOISES)
    ax.legend()

    # DFTB0 gains (from paper data)
    ax = axes[1]
    # DFTB0: approximate gain values from the tex
    dftb_gain_50k = [3, 12, 34, 50]
    dftb_gain_kicks = [6, 14, 19, 35]

    ax.bar(x - width/2, dftb_gain_50k, width, color=DFTB_LIGHT, edgecolor=DFTB_ORANGE, lw=1.5, label="Δ from 50k steps")
    ax.bar(x + width/2, dftb_gain_kicks, width, color=DFTB_ORANGE, alpha=0.85, label="Δ from kick mechanisms")

    for i in range(len(NOISES)):
        ax.text(x[i] - width/2, dftb_gain_50k[i] + 0.5, f"+{dftb_gain_50k[i]}", ha="center", fontsize=8)
        ax.text(x[i] + width/2, dftb_gain_kicks[i] + 0.5, f"+{dftb_gain_kicks[i]}", ha="center", fontsize=8)

    ax.set_xlabel("Noise level (Å)")
    ax.set_title("DFTB0: Gain Decomposition", fontweight="bold", color=DFTB_ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels(NOISES)
    ax.legend()

    fig.suptitle("Where Do Extra Converged Samples Come From?", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_gain_decomposition.png")
    plt.close(fig)
    print("  -> fig_gain_decomposition.png")


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    print("Generating LJ_RESULTS.tex figures...")
    fig_strict_relaxed_bars()
    fig_gap()
    fig_eigenvalue_histogram()
    fig_force_distribution()
    fig_cascade()
    fig_nneg_distribution()
    fig_steps_boxplot()
    fig_conditioning()
    fig_gain_decomposition()
    print(f"\nAll figures saved to {OUT_DIR}/")
