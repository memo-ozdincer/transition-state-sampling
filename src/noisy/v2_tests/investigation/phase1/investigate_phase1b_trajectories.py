#!/usr/bin/env python3
"""
Phase 1B Investigation: Trajectory-Based Analysis
===================================================
Loads per-step trajectory data to analyze convergence dynamics.

Covers:
  1B. Convergence ordering — which converges first: force or eigenvalues?
  1C. Ghost mode onset — at what force level do ghost modes appear?
  1F. Conditioning trajectory analysis
  2C. DFTB0 conditioning comparison

Subsamples trajectories (max N per combo) to keep runtime reasonable.

Output: Text report, JSON data, PNG plots.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

LJ_BASE  = Path("/scratch/memoozd/ts-tools-scratch/runs/min_nr_lj_v2_grid")
DFTB_BASE = Path("/scratch/memoozd/ts-tools-scratch/runs/min_nr_v13_1149428")

NOISE_LEVELS = ["n0.5", "n1.0", "n1.5", "n2.0"]
SELECTED_COMBO = "rfo_pls_both_kicks_adaptive_probe_late_50k"

MAX_TRAJ_PER_COMBO = 80   # subsample to keep runtime manageable
FORCE_CONV_THRESHOLD = 1e-4  # the configured force convergence threshold

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_results(base_dir, noise_level, combo_suffix):
    """Load results JSON, return list of per-sample dicts."""
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


def load_trajectory(base_dir, noise_level, combo_suffix, sample_idx):
    """Load a single trajectory JSON. Returns list of step dicts, or None."""
    combo_tag = f"{noise_level}_{combo_suffix}"
    traj_path = (base_dir / combo_tag / "diagnostics" /
                 f"sample_{sample_idx:03d}_newton_raphson_trajectory.json")
    if not traj_path.exists():
        return None
    try:
        with open(traj_path) as f:
            data = json.load(f)
        if isinstance(data, dict) and "trajectory" in data:
            return data["trajectory"]
        elif isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return None


def extract_convergence_ordering(trajectory):
    """
    For a single trajectory, find:
      - first step where force < threshold
      - first step where n_neg == 0
      - first step where BOTH are satisfied
      - force at the step where n_neg first hits 0
      - n_neg at the step where force first drops below threshold
    Returns dict or None if trajectory is empty.
    """
    if not trajectory:
        return None

    first_force = None   # first step with force < FORCE_CONV_THRESHOLD
    first_nneg0 = None   # first step with n_neg_evals == 0
    force_at_nneg0 = None
    nneg_at_force = None
    first_both = None
    total_steps = len(trajectory)

    for step in trajectory:
        s = step.get("step", 0)
        fn = step.get("force_norm")
        nn = step.get("n_neg_evals")
        if fn is None or nn is None:
            continue

        if first_force is None and fn < FORCE_CONV_THRESHOLD:
            first_force = s
            nneg_at_force = nn

        if first_nneg0 is None and nn == 0:
            first_nneg0 = s
            force_at_nneg0 = fn

        if first_both is None and fn < FORCE_CONV_THRESHOLD and nn == 0:
            first_both = s

    return {
        "total_steps": total_steps,
        "first_force_conv": first_force,
        "first_nneg0": first_nneg0,
        "first_both": first_both,
        "force_at_nneg0": force_at_nneg0,
        "nneg_at_force_conv": nneg_at_force,
    }


def extract_ghost_onset(trajectory):
    """
    Track when ghost eigenvalues (|λ| < 1e-4) first become the dominant negative modes.
    Uses cascade: n_neg_at_0.0 - n_neg_at_0.0001 = number of ghost modes.
    Returns dict with ghost onset information.
    """
    if not trajectory:
        return None

    first_ghost_step = None
    force_at_ghost_onset = None
    ghost_history = []

    for step in trajectory:
        s = step.get("step", 0)
        fn = step.get("force_norm")
        n0 = step.get("n_neg_at_0.0")
        n1 = step.get("n_neg_at_0.0001")
        if n0 is None or n1 is None:
            continue
        n_ghost = n0 - n1  # eigenvalues in (-1e-4, 0)
        ghost_history.append((s, n_ghost, fn, n0))

        if first_ghost_step is None and n_ghost > 0:
            first_ghost_step = s
            force_at_ghost_onset = fn

    return {
        "first_ghost_step": first_ghost_step,
        "force_at_ghost_onset": force_at_ghost_onset,
        "ghost_history": ghost_history,  # for plotting
    }


def extract_conditioning(trajectory):
    """Extract condition number trajectory and related quantities."""
    if not trajectory:
        return None

    steps, conds, forces, min_evals, nnegs = [], [], [], [], []
    for step in trajectory:
        s = step.get("step", 0)
        cn = step.get("cond_num")
        fn = step.get("force_norm")
        me = step.get("min_vib_eval")
        nn = step.get("n_neg_evals")
        if cn is not None and np.isfinite(cn) and cn > 0:
            steps.append(s)
            conds.append(cn)
            forces.append(fn)
            min_evals.append(me)
            nnegs.append(nn)

    if not steps:
        return None

    return {
        "steps": steps,
        "cond_nums": conds,
        "forces": forces,
        "min_evals": min_evals,
        "n_negs": nnegs,
        "final_cond": conds[-1],
        "max_cond": max(conds),
        "median_cond": float(np.median(conds)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_convergence_ordering(ordering_data, output_dir):
    """
    Plot convergence ordering: histogram of (step_nneg0 - step_force_conv).
    Positive = eigenvalues converge after forces (LJ behavior expected).
    Negative = eigenvalues converge before forces (DFTB0 behavior expected).
    """
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), constrained_layout=True)
    fig.suptitle("Convergence Ordering: step(n_neg=0) − step(F̄<1e-4)\n"
                 "Positive = eigenvalues converge LATER; Negative = eigenvalues converge FIRST",
                 fontsize=13)

    for col, noise in enumerate(NOISE_LEVELS):
        for row, (pes, label) in enumerate([("lj", "LJ"), ("dftb", "DFTB0")]):
            ax = axes[row, col]
            data = ordering_data.get((pes, noise), [])
            if not data:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                continue

            # Cases where both events occurred
            both = [d for d in data if d["first_force_conv"] is not None and d["first_nneg0"] is not None]
            force_only = [d for d in data if d["first_force_conv"] is not None and d["first_nneg0"] is None]
            nneg_only = [d for d in data if d["first_force_conv"] is None and d["first_nneg0"] is not None]
            neither = [d for d in data if d["first_force_conv"] is None and d["first_nneg0"] is None]

            if both:
                diffs = [d["first_nneg0"] - d["first_force_conv"] for d in both]
                ax.hist(diffs, bins=30, alpha=0.7, color="#2166ac" if pes == "lj" else "#b2182b",
                        edgecolor="black", linewidth=0.5)
                med = np.median(diffs)
                ax.axvline(med, color="black", ls="--", lw=1.2)
                ax.axvline(0, color="gray", ls="-", lw=0.8, alpha=0.5)

            # Annotation
            n_total = len(data)
            txt = f"n={n_total}\nboth: {len(both)}\nF only: {len(force_only)}\n"
            txt += f"n_neg only: {len(nneg_only)}\nneither: {len(neither)}"
            ax.text(0.98, 0.98, txt, transform=ax.transAxes, ha="right", va="top",
                    fontsize=7, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            if col == 0:
                ax.set_ylabel(f"{label}\nCount")
            if row == 1:
                ax.set_xlabel("step(n_neg=0) − step(F̄<1e-4)")
            if row == 0:
                ax.set_title(noise)

    path = output_dir / "convergence_ordering.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_force_at_nneg0(ordering_data, output_dir):
    """Plot the force norm at the step when n_neg first hits 0."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    fig.suptitle("Force Norm When n_neg First Reaches 0", fontsize=13)

    for col, noise in enumerate(NOISE_LEVELS):
        ax = axes[col]
        for pes, color, label in [("lj", "#2166ac", "LJ"), ("dftb", "#b2182b", "DFTB0")]:
            data = ordering_data.get((pes, noise), [])
            vals = [d["force_at_nneg0"] for d in data
                    if d["force_at_nneg0"] is not None]
            if vals:
                ax.hist(vals, bins=30, alpha=0.5, color=color, label=label)

        ax.set_yscale("linear")
        ax.set_xlabel("F̄ at first n_neg=0")
        if col == 0:
            ax.set_ylabel("Count")
        ax.set_title(noise)
        ax.axvline(FORCE_CONV_THRESHOLD, color="gray", ls="--", lw=1, label=f"F̄={FORCE_CONV_THRESHOLD}")
        ax.legend(fontsize=8)

    path = output_dir / "force_at_nneg0.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_ghost_onset(ghost_data, output_dir):
    """Plot force norm at ghost onset (LJ only, since DFTB0 has no ghosts)."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    fig.suptitle("LJ: Force Norm at Ghost Mode Onset (|λ| < 1e-4 eigenvalues first appear)", fontsize=13)

    for col, noise in enumerate(NOISE_LEVELS):
        ax = axes[col]
        data = ghost_data.get(("lj", noise), [])
        vals = [d["force_at_ghost_onset"] for d in data
                if d["force_at_ghost_onset"] is not None]
        if vals:
            ax.hist(vals, bins=30, alpha=0.7, color="#2166ac", edgecolor="black", linewidth=0.5)
            med = np.median(vals)
            ax.axvline(med, color="red", ls="--", lw=1.2, label=f"median={med:.2e}")
            ax.axvline(FORCE_CONV_THRESHOLD, color="gray", ls="--", lw=1, label="F̄=1e-4")
        else:
            ax.text(0.5, 0.5, f"No ghost\nonset\n(n={len(data)})", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
        ax.set_xlabel("F̄ at ghost onset")
        if col == 0:
            ax.set_ylabel("Count")
        ax.set_title(noise)
        ax.legend(fontsize=8)

    path = output_dir / "ghost_onset_force.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_conditioning_comparison(cond_data, output_dir):
    """Box plot of final condition numbers: LJ vs DFTB0."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    fig.suptitle("Hessian Condition Number at Convergence", fontsize=13)

    for col, noise in enumerate(NOISE_LEVELS):
        ax = axes[col]
        plot_data = []
        labels = []
        colors = []
        for pes, color, label in [("lj", "#2166ac", "LJ"), ("dftb", "#b2182b", "DFTB0")]:
            data = cond_data.get((pes, noise), [])
            finals = [d["final_cond"] for d in data if d is not None]
            if finals:
                plot_data.append(finals)
                labels.append(label)
                colors.append(color)

        if plot_data:
            bp = ax.boxplot(plot_data, labels=labels, patch_artist=True, showfliers=True,
                           flierprops=dict(markersize=2, alpha=0.3))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)

        ax.set_yscale("log")
        ax.set_ylabel("κ (condition number)" if col == 0 else "")
        ax.set_title(noise)

    path = output_dir / "conditioning_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════

class Report:
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
    R("PHASE 1B INVESTIGATION: Trajectory-Based Analysis")
    R("=" * 90)
    R()

    # ── Identify samples to load ──────────────────────────────────────────
    ordering_data  = {}
    ghost_data     = {}
    cond_data      = {}

    for pes, base in [("lj", LJ_BASE), ("dftb", DFTB_BASE)]:
        for noise in NOISE_LEVELS:
            rl, tag = load_results(base, noise, SELECTED_COMBO)
            if rl is None:
                R(f"  {pes.upper()} {tag}: NOT FOUND, skipping")
                continue

            # Select sample indices (prefer converged, subsample if needed)
            valid = [(i, r) for i, r in enumerate(rl) if r.get("error") is None]
            converged = [(i, r) for i, r in valid if r.get("converged")]
            nonconv   = [(i, r) for i, r in valid if not r.get("converged")]

            # Take up to MAX_TRAJ_PER_COMBO converged, plus some non-converged
            n_conv = min(len(converged), MAX_TRAJ_PER_COMBO)
            n_nonc = min(len(nonconv), MAX_TRAJ_PER_COMBO // 4)
            selected = converged[:n_conv] + nonconv[:n_nonc]

            R(f"  {pes.upper()} {tag}: loading {len(selected)} trajectories "
              f"({n_conv} converged, {n_nonc} non-converged)...")

            orders = []
            ghosts = []
            conds  = []
            loaded = 0

            for idx, r in selected:
                sample_idx = r.get("sample_idx", idx)
                traj = load_trajectory(base, noise, SELECTED_COMBO, sample_idx)
                if traj is None:
                    continue
                loaded += 1

                # 1B: Convergence ordering
                o = extract_convergence_ordering(traj)
                if o:
                    o["converged"] = r.get("converged", False)
                    orders.append(o)

                # 1C: Ghost onset (LJ only, DFTB0 has no ghosts)
                if pes == "lj":
                    g = extract_ghost_onset(traj)
                    if g:
                        ghosts.append(g)

                # 1F: Conditioning
                c = extract_conditioning(traj)
                if c:
                    conds.append(c)

            R(f"    Loaded {loaded}/{len(selected)} trajectories")
            ordering_data[(pes, noise)] = orders
            ghost_data[(pes, noise)]    = ghosts
            cond_data[(pes, noise)]     = conds

    R()

    # ══════════════════════════════════════════════════════════════════════
    #  1B: CONVERGENCE ORDERING
    # ══════════════════════════════════════════════════════════════════════
    R("─" * 90)
    R("ANALYSIS 1B: Convergence Ordering")
    R("─" * 90)
    R("Q: Does force converge before or after eigenvalues?")
    R(f"   Force threshold = {FORCE_CONV_THRESHOLD:.0e}, eigenvalue threshold = n_neg=0")
    R()

    R(f"  {'PES':>5s} {'noise':>5s}  {'n':>4s}  {'both':>5s} {'F only':>7s} {'nn only':>7s} {'neither':>7s}  "
      f"{'med(Δstep)':>11s}  {'med F@ nn=0':>12s}  {'med nn@ F':>10s}")
    R(f"  {'─'*5} {'─'*5}  {'─'*4}  {'─'*5} {'─'*7} {'─'*7} {'─'*7}  {'─'*11}  {'─'*12}  {'─'*10}")

    for pes in ["lj", "dftb"]:
        for noise in NOISE_LEVELS:
            data = ordering_data.get((pes, noise), [])
            if not data:
                continue
            n = len(data)
            both = [d for d in data if d["first_force_conv"] is not None and d["first_nneg0"] is not None]
            f_only = [d for d in data if d["first_force_conv"] is not None and d["first_nneg0"] is None]
            nn_only = [d for d in data if d["first_force_conv"] is None and d["first_nneg0"] is not None]
            neither = [d for d in data if d["first_force_conv"] is None and d["first_nneg0"] is None]

            if both:
                diffs = [d["first_nneg0"] - d["first_force_conv"] for d in both]
                med_diff = f"{np.median(diffs):.0f}"
            else:
                med_diff = "—"

            forces_at_nn0 = [d["force_at_nneg0"] for d in data if d["force_at_nneg0"] is not None]
            med_f_at_nn0 = f"{np.median(forces_at_nn0):.2e}" if forces_at_nn0 else "—"

            nnegs_at_f = [d["nneg_at_force_conv"] for d in data
                         if d["nneg_at_force_conv"] is not None]
            med_nn_at_f = f"{np.median(nnegs_at_f):.0f}" if nnegs_at_f else "—"

            R(f"  {pes.upper():>5s} {noise:>5s}  {n:>4d}  {len(both):>5d} {len(f_only):>7d} {len(nn_only):>7d} "
              f"{len(neither):>7d}  {med_diff:>11s}  {med_f_at_nn0:>12s}  {med_nn_at_f:>10s}")
    R()

    # Detailed breakdown
    R("  Detailed: Force at the moment n_neg first hits 0:")
    for pes in ["lj", "dftb"]:
        R(f"  {pes.upper()}:")
        for noise in NOISE_LEVELS:
            data = ordering_data.get((pes, noise), [])
            vals = [d["force_at_nneg0"] for d in data if d["force_at_nneg0"] is not None]
            if not vals:
                continue
            a = np.array(vals)
            R(f"    {noise}: n={len(vals)}, min={np.min(a):.2e}, p5={np.percentile(a,5):.2e}, "
              f"median={np.median(a):.2e}, p95={np.percentile(a,95):.2e}, max={np.max(a):.2e}")
            R(f"          F̄ < 1e-4: {np.mean(a < 1e-4):.1%}, "
              f"F̄ < 1e-3: {np.mean(a < 1e-3):.1%}, "
              f"F̄ < 1e-2: {np.mean(a < 1e-2):.1%}")
        R()

    # ══════════════════════════════════════════════════════════════════════
    #  1C: GHOST MODE ONSET (LJ only)
    # ══════════════════════════════════════════════════════════════════════
    R("─" * 90)
    R("ANALYSIS 1C: Ghost Mode Onset (LJ only)")
    R("─" * 90)
    R("Q: At what force level do ghost eigenvalues (|λ| < 1e-4) first appear?")
    R()

    for noise in NOISE_LEVELS:
        data = ghost_data.get(("lj", noise), [])
        if not data:
            continue
        onset_forces = [d["force_at_ghost_onset"] for d in data if d["force_at_ghost_onset"] is not None]
        no_ghosts = [d for d in data if d["first_ghost_step"] is None]
        R(f"  {noise}: {len(onset_forces)} with ghost onset, {len(no_ghosts)} without ghosts")
        if onset_forces:
            a = np.array(onset_forces)
            R(f"    Force at ghost onset: min={np.min(a):.2e}, median={np.median(a):.2e}, "
              f"p95={np.percentile(a,95):.2e}, max={np.max(a):.2e}")
            R(f"    Ghosts appear before force < 1e-4: {np.mean(a > 1e-4):.1%}")
            R(f"    Ghosts appear before force < 1e-3: {np.mean(a > 1e-3):.1%}")
            R(f"    Ghosts appear before force < 1e-2: {np.mean(a > 1e-2):.1%}")
    R()

    # ══════════════════════════════════════════════════════════════════════
    #  1F: CONDITIONING
    # ══════════════════════════════════════════════════════════════════════
    R("─" * 90)
    R("ANALYSIS 1F: Hessian Conditioning")
    R("─" * 90)
    R()

    R(f"  {'PES':>5s} {'noise':>5s}  {'n':>4s}  {'med κ_final':>12s} {'med κ_max':>12s} {'med κ_med':>12s}")
    R(f"  {'─'*5} {'─'*5}  {'─'*4}  {'─'*12} {'─'*12} {'─'*12}")

    for pes in ["lj", "dftb"]:
        for noise in NOISE_LEVELS:
            data = cond_data.get((pes, noise), [])
            if not data:
                continue
            finals = [d["final_cond"] for d in data]
            maxes  = [d["max_cond"] for d in data]
            meds   = [d["median_cond"] for d in data]
            R(f"  {pes.upper():>5s} {noise:>5s}  {len(data):>4d}  "
              f"{np.median(finals):>12.2e} {np.median(maxes):>12.2e} {np.median(meds):>12.2e}")
    R()

    # ══════════════════════════════════════════════════════════════════════
    #  KEY FINDINGS
    # ══════════════════════════════════════════════════════════════════════
    R("=" * 90)
    R("KEY FINDINGS (Phase 1B — Trajectory Analysis)")
    R("=" * 90)
    R()

    # Convergence ordering summary
    R("1. CONVERGENCE ORDERING:")
    for pes in ["lj", "dftb"]:
        R(f"   {pes.upper()}:")
        for noise in NOISE_LEVELS:
            data = ordering_data.get((pes, noise), [])
            if not data:
                continue
            both = [d for d in data if d["first_force_conv"] is not None and d["first_nneg0"] is not None]
            nn_only = [d for d in data if d["first_force_conv"] is None and d["first_nneg0"] is not None]
            f_only = [d for d in data if d["first_force_conv"] is not None and d["first_nneg0"] is None]
            if both:
                diffs = [d["first_nneg0"] - d["first_force_conv"] for d in both]
                if np.median(diffs) > 0:
                    R(f"     {noise}: eigenvalues lag forces by ~{np.median(diffs):.0f} steps (force-first)")
                else:
                    R(f"     {noise}: eigenvalues lead forces by ~{-np.median(diffs):.0f} steps (eigenvalue-first)")
            if nn_only:
                R(f"     {noise}: {len(nn_only)} samples reach n_neg=0 but NEVER reach F̄<1e-4")
            if f_only:
                R(f"     {noise}: {len(f_only)} samples reach F̄<1e-4 but NEVER reach n_neg=0")
    R()

    R("2. FORCE AT EIGENVALUE CONVERGENCE (n_neg=0):")
    R("   This is the force the optimizer would declare 'converged' at under strict criterion:")
    for pes in ["lj", "dftb"]:
        for noise in NOISE_LEVELS:
            data = ordering_data.get((pes, noise), [])
            vals = [d["force_at_nneg0"] for d in data if d["force_at_nneg0"] is not None]
            if vals:
                R(f"   {pes.upper()} {noise}: median F̄ = {np.median(vals):.2e} "
                  f"(range {np.min(vals):.2e} – {np.max(vals):.2e})")
    R()

    R("3. GHOST MODE ONSET (LJ):")
    for noise in NOISE_LEVELS:
        data = ghost_data.get(("lj", noise), [])
        onset = [d["force_at_ghost_onset"] for d in data if d["force_at_ghost_onset"] is not None]
        if onset:
            R(f"   {noise}: ghosts appear at median F̄ = {np.median(onset):.2e}")
    R()

    # ══════════════════════════════════════════════════════════════════════
    #  PLOTS
    # ══════════════════════════════════════════════════════════════════════
    R("─" * 90)
    R("Generating plots...")
    R("─" * 90)
    R()
    for fn in [plot_convergence_ordering, plot_force_at_nneg0, plot_ghost_onset, plot_conditioning_comparison]:
        try:
            data_arg = ordering_data if "ordering" in fn.__name__ or "force_at" in fn.__name__ else (
                ghost_data if "ghost" in fn.__name__ else cond_data)
            p = fn(data_arg, out)
            R(f"  {p}")
        except Exception as e:
            R(f"  Error in {fn.__name__}: {e}")
    R()

    # ── Save ──────────────────────────────────────────────────────────────
    R.save(out / "phase1b_report.txt")
    print(f"\nReport: {out / 'phase1b_report.txt'}")

    # JSON summary
    summary = {}
    for (pes, noise), data in ordering_data.items():
        key = f"{pes}_{noise}"
        summary[key] = {
            "n": len(data),
            "n_both": len([d for d in data if d["first_force_conv"] is not None and d["first_nneg0"] is not None]),
            "n_force_only": len([d for d in data if d["first_force_conv"] is not None and d["first_nneg0"] is None]),
            "n_nneg_only": len([d for d in data if d["first_force_conv"] is None and d["first_nneg0"] is not None]),
            "force_at_nneg0_stats": {
                "vals": [d["force_at_nneg0"] for d in data if d["force_at_nneg0"] is not None]
            },
        }
    with open(out / "phase1b_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)
    print(f"JSON:   {out / 'phase1b_summary.json'}")


if __name__ == "__main__":
    main()
