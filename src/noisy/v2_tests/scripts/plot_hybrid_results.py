#!/usr/bin/env python3
"""Generate informative plots for the NR→GAD hybrid results."""

import json
import glob
import os
import sys
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# ── Configuration ──────────────────────────────────────────────────────────
OUT_DIR = "/scratch/memoozd/ts-tools-scratch/runs/nr_gad_plots"
os.makedirs(OUT_DIR, exist_ok=True)

NOPATH_DIRS = {
    '0.5': '/scratch/memoozd/ts-tools-scratch/runs/nr_gad_tri0279/exp2_nopath_n0.5_300s',
    '1.0': '/scratch/memoozd/ts-tools-scratch/runs/nr_gad_tri0279/exp2_nopath_n1.0_300s',
    '1.5': '/scratch/memoozd/ts-tools-scratch/runs/nr_gad_tri0279/exp2_nopath_n1.5_300s',
}
PATH_DIRS = {
    '0.5': '/scratch/memoozd/ts-tools-scratch/runs/nr_gad_tri0279/exp1_path_n0.5_300s',
    '1.0': '/scratch/memoozd/ts-tools-scratch/runs/nr_gad_tri0279/exp1_path_n1.0_300s',
    '1.5': '/scratch/memoozd/ts-tools-scratch/runs/nr_gad_tri0279/exp1_path_n1.5_300s',
    '2.0': '/scratch/memoozd/ts-tools-scratch/runs/nr_gad_tri0279/exp1_path_n2.0_300s',
}

COLORS = {'0.5': '#2196F3', '1.0': '#4CAF50', '1.5': '#FF9800', '2.0': '#F44336'}


def load_summaries(base_path):
    files = sorted(glob.glob(os.path.join(base_path, 'diagnostics', 'sample_*_summary.json')))
    data = []
    for f in files:
        d = json.load(open(f))
        data.append(d)
    return data


def load_trajectory(traj_path):
    return json.load(open(traj_path))


# ── Figure 1: Path vs Nopath TS Rate Bar Chart ────────────────────────────
def plot_ts_rate_comparison():
    fig, ax = plt.subplots(figsize=(8, 5))
    noise_levels = ['0.5', '1.0', '1.5', '2.0']
    x = np.arange(len(noise_levels))
    width = 0.35

    path_rates = []
    nopath_rates = []
    for n in noise_levels:
        if n in PATH_DIRS:
            data = load_summaries(PATH_DIRS[n])
            ts = sum(1 for d in data if d.get('converged_to_ts', False))
            path_rates.append(100 * ts / len(data) if data else 0)
        else:
            path_rates.append(0)
        if n in NOPATH_DIRS:
            data = load_summaries(NOPATH_DIRS[n])
            ts = sum(1 for d in data if d.get('converged_to_ts', False))
            nopath_rates.append(100 * ts / len(data) if data else 0)
        else:
            nopath_rates.append(0)

    bars1 = ax.bar(x - width/2, path_rates, width, label='Path GAD', color='#E57373', edgecolor='#C62828')
    bars2 = ax.bar(x + width/2, nopath_rates, width, label='Nopath GAD', color='#81C784', edgecolor='#2E7D32')

    for bar, val in zip(bars1, path_rates):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, nopath_rates):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Noise level σ (Å)', fontsize=12)
    ax.set_ylabel('TS found rate (%)', fontsize=12)
    ax.set_title('Path vs Nopath GAD: TS Rate Given NR Convergence', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f'σ={n}' for n in noise_levels])
    ax.set_ylim(0, 115)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig1_ts_rate_comparison.png'), dpi=200)
    plt.close(fig)
    print("  -> fig1_ts_rate_comparison.png")


# ── Figure 2: GAD Step Distribution (Nopath) ──────────────────────────────
def plot_gad_step_distribution():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for i, (noise, path) in enumerate(sorted(NOPATH_DIRS.items(), key=lambda x: float(x[0]))):
        ax = axes[i]
        data = load_summaries(path)
        conv_steps = [d['total_steps'] for d in data if d.get('converged_to_ts', False)]
        if not conv_steps:
            continue

        # Log-scale histogram
        bins = np.logspace(0, np.log10(max(conv_steps) + 1), 40)
        ax.hist(conv_steps, bins=bins, color=COLORS[noise], alpha=0.8, edgecolor='white', linewidth=0.5)
        ax.set_xscale('log')
        ax.axvline(statistics.median(conv_steps), color='red', linestyle='--', linewidth=2,
                    label=f'median={statistics.median(conv_steps):.0f}')
        ax.axvline(sorted(conv_steps)[int(0.95 * len(conv_steps))], color='orange',
                    linestyle=':', linewidth=2,
                    label=f'p95={sorted(conv_steps)[int(0.95*len(conv_steps))]:.0f}')
        ax.set_xlabel('GAD steps to TS', fontsize=11)
        ax.set_title(f'σ = {noise} Å  (n={len(conv_steps)})', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel('Count', fontsize=11)
    fig.suptitle('Nopath GAD: Step Count Distribution (Converged Samples)', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig2_nopath_step_distribution.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  -> fig2_nopath_step_distribution.png")


# ── Figure 3: Path GAD Failure Morse Index Distribution ───────────────────
def plot_path_failure_morse():
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    noise_levels = ['0.5', '1.0', '1.5', '2.0']

    for i, noise in enumerate(noise_levels):
        ax = axes[i]
        if noise not in PATH_DIRS:
            continue
        data = load_summaries(PATH_DIRS[noise])
        failures = [d for d in data if not d.get('converged_to_ts', False)]
        morse_counts = defaultdict(int)
        for d in failures:
            fa = d.get('failure_analysis', {})
            mi = fa.get('final_morse_index', d.get('final_morse_index', -1))
            morse_counts[mi] = morse_counts.get(mi, 0) + 1

        if not morse_counts:
            continue

        indices = sorted(morse_counts.keys())
        counts = [morse_counts[m] for m in indices]
        colors_bar = ['#BBDEFB' if m == 0 else '#FFCDD2' if m == 1 else '#EF5350' for m in indices]
        ax.bar([str(m) for m in indices], counts, color=colors_bar, edgecolor='#333')
        ax.set_xlabel('Final Morse index', fontsize=11)
        ax.set_title(f'σ = {noise} Å\n{len(failures)} failures', fontsize=11)
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel('Count', fontsize=11)
    fig.suptitle('Path GAD: Morse Index Distribution of Failed Samples', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig3_path_failure_morse.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  -> fig3_path_failure_morse.png")


# ── Figure 4: Nopath Trajectory Examples ──────────────────────────────────
def plot_nopath_trajectory_examples():
    """Show eigenvalue evolution for fast, medium, and slow convergence."""
    base = NOPATH_DIRS['1.0']
    data = load_summaries(base)
    conv = [(d, d['total_steps']) for d in data if d.get('converged_to_ts', False)]
    conv.sort(key=lambda x: x[1])

    # Pick fast (~10 steps), medium (p50), slow (max) — avoid trivial 1-2 step samples
    fast_candidates = [(d, s) for d, s in conv if 5 <= s <= 20]
    picks = [
        ('Fast (~10 steps)', fast_candidates[0] if fast_candidates else conv[int(0.1 * len(conv))]),
        ('Median', conv[len(conv) // 2]),
        ('Slowest converged', conv[-1]),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))

    for row, (label, (summary, steps)) in enumerate(picks):
        sid = summary['sample_id']
        traj_path = os.path.join(base, 'diagnostics', f'{sid}_trajectory.json')
        if not os.path.exists(traj_path):
            continue
        traj = load_trajectory(traj_path)

        step_arr = np.array(traj['step'])
        eig0 = np.array(traj['eig_0'])
        eig1 = np.array(traj['eig_1'])
        morse = np.array(traj['morse_index'])
        dt_eff = np.array(traj['step_size_eff'])

        # Panel 1: Eigenvalues
        ax = axes[row, 0]
        ax.plot(step_arr, eig0, 'b-', linewidth=1.2, label='λ₀')
        ax.plot(step_arr, eig1, 'r-', linewidth=1.2, label='λ₁')
        ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
        ax.set_ylabel('Eigenvalue (Eh/Å²)')
        ax.set_title(f'{label}: {sid} ({steps} steps)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Panel 2: Morse index
        ax = axes[row, 1]
        ax.plot(step_arr, morse, 'k-', linewidth=1)
        ax.axhline(1, color='green', linestyle='--', linewidth=1.5, label='Target (index 1)')
        ax.set_ylabel('Morse index')
        ax.set_title(f'Morse index evolution')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Panel 3: Effective step size
        ax = axes[row, 2]
        ax.plot(step_arr, dt_eff, 'purple', linewidth=1)
        ax.set_yscale('log')
        ax.set_ylabel('dt_eff')
        ax.set_title(f'Adaptive step size')
        ax.grid(alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel('GAD step')

    fig.suptitle('Nopath GAD Trajectory Examples (σ = 1.0 Å): Eigenvalue-Clamped dt Dynamics',
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig4_nopath_trajectory_examples.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  -> fig4_nopath_trajectory_examples.png")


# ── Figure 5: Eigenvalue product at convergence ──────────────────────────
def plot_eig_product_at_convergence():
    """Diagnostic: eigenvalue product distribution at convergence.

    NOTE: Convergence gate is n_neg == 1, NOT eig_product.  This plot shows
    the eigenvalue product as a diagnostic measure of how cleanly separated
    the negative eigenvalue is from the positive spectrum.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for i, (noise, path) in enumerate(sorted(NOPATH_DIRS.items(), key=lambda x: float(x[0]))):
        ax = axes[i]
        traj_files = sorted(glob.glob(os.path.join(path, 'diagnostics', 'sample_*_trajectory.json')))
        summary_files = sorted(glob.glob(os.path.join(path, 'diagnostics', 'sample_*_summary.json')))

        # Build set of converged sample IDs
        converged_ids = set()
        for sf in summary_files:
            d = json.load(open(sf))
            if d.get('converged_to_ts', False):
                converged_ids.add(d['sample_id'])

        final_products = []
        for tf in traj_files:
            sid = os.path.basename(tf).replace('_trajectory.json', '')
            if sid not in converged_ids:
                continue
            traj = json.load(open(tf))
            eig0 = traj.get('eig_0', [])
            eig1 = traj.get('eig_1', [])
            if not eig0 or not eig1:
                continue
            # Find the first step where product < -ts_eps (convergence point)
            for j in range(len(eig0)):
                prod = eig0[j] * eig1[j]
                if np.isfinite(prod) and prod < -1e-5:
                    final_products.append(abs(prod))
                    break
            else:
                # Fallback: use last step
                prod = eig0[-1] * eig1[-1]
                if np.isfinite(prod) and prod < 0:
                    final_products.append(abs(prod))

        if not final_products:
            continue

        bins = np.logspace(np.log10(min(final_products)), np.log10(max(final_products)), 40)
        ax.hist(final_products, bins=bins, color=COLORS[noise], alpha=0.8, edgecolor='white')
        ax.axvline(1e-5, color='red', linestyle='--', linewidth=2, label='ts_eps = 1e-5')
        ax.set_xscale('log')
        ax.set_xlabel('|λ₀ × λ₁| at convergence', fontsize=11)
        ax.set_title(f'σ = {noise} Å  (n={len(final_products)})', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel('Count', fontsize=11)
    fig.suptitle('Nopath GAD: Eigenvalue Product at Convergence (diagnostic, not gate)',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig5_eig_product_at_convergence.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  -> fig5_eig_product_at_convergence.png")


# ── Figure 6: Path vs Nopath side-by-side trajectory (same sample) ───────
def plot_path_vs_nopath_same_sample():
    """Show eigenvalue evolution for the same sample under both GAD variants."""
    # Use σ=1.0, find samples that converge with nopath but fail with path
    path_base = PATH_DIRS['1.0']
    nopath_base = NOPATH_DIRS['1.0']

    path_summaries = {}
    for f in glob.glob(os.path.join(path_base, 'diagnostics', 'sample_*_summary.json')):
        d = json.load(open(f))
        path_summaries[d['sample_id']] = d

    nopath_summaries = {}
    for f in glob.glob(os.path.join(nopath_base, 'diagnostics', 'sample_*_summary.json')):
        d = json.load(open(f))
        nopath_summaries[d['sample_id']] = d

    # Find samples where nopath succeeds and path fails
    divergent = []
    for sid in nopath_summaries:
        if sid in path_summaries:
            np_conv = nopath_summaries[sid].get('converged_to_ts', False)
            p_conv = path_summaries[sid].get('converged_to_ts', False)
            if np_conv and not p_conv:
                np_steps = nopath_summaries[sid].get('total_steps', 0)
                divergent.append((sid, np_steps))

    divergent.sort(key=lambda x: x[1])  # Sort by nopath step count

    if len(divergent) < 3:
        print("  -> Not enough divergent samples for fig6")
        return

    # Pick 3 representative examples
    picks = [divergent[0], divergent[len(divergent)//2], divergent[-1]]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    for row, (sid, np_steps) in enumerate(picks):
        # Path trajectory
        path_traj_f = os.path.join(path_base, 'diagnostics', f'{sid}_trajectory.json')
        nopath_traj_f = os.path.join(nopath_base, 'diagnostics', f'{sid}_trajectory.json')
        if not os.path.exists(path_traj_f) or not os.path.exists(nopath_traj_f):
            continue

        ptraj = load_trajectory(path_traj_f)
        ntraj = load_trajectory(nopath_traj_f)

        # Path panel
        ax = axes[row, 0]
        step_p = np.array(ptraj['step'][:500])  # Limit to first 500 for readability
        morse_p = np.array(ptraj['morse_index'][:500])
        ax.plot(step_p, morse_p, 'r-', linewidth=0.8)
        ax.axhline(1, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_ylabel('Morse index')
        p_fa = path_summaries[sid].get('failure_analysis', {})
        p_final_morse = p_fa.get('final_morse_index', '?')
        ax.set_title(f'Path GAD: {sid} — FAILS (final Morse {p_final_morse})', color='red')
        ax.set_ylim(-0.5, max(morse_p) + 1)
        ax.grid(alpha=0.3)

        # Nopath panel
        ax = axes[row, 1]
        step_n = np.array(ntraj['step'])
        morse_n = np.array(ntraj['morse_index'])
        ax.plot(step_n, morse_n, 'g-', linewidth=0.8)
        ax.axhline(1, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_ylabel('Morse index')
        ax.set_title(f'Nopath GAD: {sid} — CONVERGES ({np_steps} steps)', color='green')
        ax.set_ylim(-0.5, max(max(morse_n), max(morse_p)) + 1)
        ax.grid(alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel('GAD step')

    fig.suptitle('Same Samples, Different Outcomes (σ = 1.0 Å): Path vs Nopath GAD',
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig6_path_vs_nopath_same_sample.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  -> fig6_path_vs_nopath_same_sample.png")


# ── Figure 7: Nopath dt_eff self-regulation mechanism ────────────────────
def plot_dt_self_regulation():
    """Show how dt_eff correlates with |λ₀| — the self-regulation mechanism."""
    base = NOPATH_DIRS['1.0']
    data = load_summaries(base)
    conv = [(d, d['total_steps']) for d in data if d.get('converged_to_ts', False)]
    conv.sort(key=lambda x: x[1])

    # Pick a medium-speed sample with interesting dynamics
    mid = conv[len(conv) // 2]
    sid = mid[0]['sample_id']

    traj_path = os.path.join(base, 'diagnostics', f'{sid}_trajectory.json')
    traj = load_trajectory(traj_path)

    step_arr = np.array(traj['step'])
    eig0 = np.array(traj['eig_0'])
    dt_eff = np.array(traj['step_size_eff'])
    morse = np.array(traj['morse_index'])

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Panel 1: |λ₀|
    ax = axes[0]
    ax.plot(step_arr, np.abs(eig0), 'b-', linewidth=1.2)
    ax.set_yscale('log')
    ax.set_ylabel('|λ₀| (Eh/Å²)')
    ax.set_title(f'Nopath GAD self-regulation: {sid} (σ=1.0, {mid[1]} steps)')
    ax.grid(alpha=0.3)

    # Panel 2: dt_eff
    ax = axes[1]
    ax.plot(step_arr, dt_eff, 'purple', linewidth=1.2)
    ax.set_yscale('log')
    ax.set_ylabel('dt_eff')
    ax.grid(alpha=0.3)
    # Annotate the relationship
    ax.text(0.02, 0.95, r'$dt_{\rm eff} = dt \cdot s\, /\, {\rm clamp}(|\lambda_0|)$',
            transform=ax.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 3: Morse index
    ax = axes[2]
    ax.plot(step_arr, morse, 'k-', linewidth=1)
    ax.axhline(1, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Target')
    ax.set_ylabel('Morse index')
    ax.set_xlabel('GAD step')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig7_dt_self_regulation.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  -> fig7_dt_self_regulation.png")


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Generating plots in {OUT_DIR}")
    print()

    print("[1/7] TS rate comparison...")
    plot_ts_rate_comparison()

    print("[2/7] Nopath step distribution...")
    plot_gad_step_distribution()

    print("[3/7] Path failure Morse distribution...")
    plot_path_failure_morse()

    print("[4/7] Nopath trajectory examples...")
    plot_nopath_trajectory_examples()

    print("[5/7] Eigenvalue product at convergence...")
    plot_eig_product_at_convergence()

    print("[6/7] Path vs Nopath same-sample comparison...")
    plot_path_vs_nopath_same_sample()

    print("[7/7] dt self-regulation mechanism...")
    plot_dt_self_regulation()

    print(f"\nAll plots saved to: {OUT_DIR}")
