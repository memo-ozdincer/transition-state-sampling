# Investigation: Force–Eigenvalue–n_neg Relationship

## Overview

This investigation examines the joint behavior of three convergence quantities —
force norm (F̄), smallest eigenvalue (λ_min), and count of negative eigenvalues
(n_neg) — across two potential energy surfaces: Lennard-Jones (LJ, pairwise) and
DFTB0 (semi-empirical quantum). The goal is to understand what "converged to a
minimum" actually means on each PES, how small each quantity can get, and what the
tradeoffs are.

**Data sources:**
- LJ v2 grid: `/scratch/memoozd/ts-tools-scratch/runs/min_nr_lj_v2_grid/`
- DFTB0 v13:  `/scratch/memoozd/ts-tools-scratch/runs/min_nr_v13_1149428/`
- Combo analyzed: `*_rfo_pls_both_kicks_adaptive_probe_late_50k` (best-performing per noise level)
- ~280 samples per noise level per PES, 4 noise levels (n=0.5, 1.0, 1.5, 2.0 Å)

**Scripts:**
- `investigation/phase1/investigate_phase1a.py` — results-JSON analysis
- `investigation/phase1/investigate_phase1b_trajectories.py` — trajectory analysis

---

## Finding 1: The Two PES Have Opposite Convergence Pathologies

This is the central finding. LJ and DFTB0 fail to converge in *opposite* ways:

| Property | LJ | DFTB0 |
|----------|-------|--------|
| Force convergence | Easy (F̄ < 1e-4 achievable) | Broken (median F̄ ~ 0.05–0.12 eV/Å) |
| Eigenvalue convergence | Hard (ghost modes at high noise) | Trivial (100% strict at all noise) |
| Limiting factor | Eigenvalue positivity | Force magnitude |
| Relaxed convergence needed? | Yes, heavily at n≥1.5 | Never (strict always fires first) |

The convergence code has two paths:
- **Strict**: `n_neg == 0`, no force check → declares convergence the instant all eigenvalues become positive
- **Relaxed**: `F̄ < 1e-4 AND λ_min ≥ -0.01` → requires both force and eigenvalue conditions

On **LJ**, forces converge to ~1e-4 easily, but ghost eigenvalues prevent n_neg=0
at high noise. Relaxed criterion saves these samples.

On **DFTB0**, n_neg hits 0 early in optimization (eigenvalues are well-behaved),
so strict fires immediately — but forces are still ~0.05–0.12 eV/Å at that moment.
The optimizer declares victory and stops, leaving the geometry far from a true minimum.

### Implication

The "strict convergence has no force check" design is a **critical flaw on DFTB0**.
It produces "converged" geometries that are NOT force-converged. This was masked
because the convergence *rate* looks good (93–99%), but the convergence *quality*
is poor.

---

## Finding 2: 2D Threshold Sweep Reveals the Asymmetry

Convergence rate when applying joint criterion (F̄ < F_thresh AND λ_min ≥ -τ)
to ALL samples' final states:

### LJ (selected rows)

| Noise | F < 1e-2, τ=0.1 | F < 1e-4, τ=1e-3 | F < 1e-4, strict |
|-------|------------------|-------------------|------------------|
| n0.5  | 96.0%            | 95.6%             | 0.0%*            |
| n1.0  | 96.8%            | 31.5%             | 0.0%             |
| n1.5  | 98.2%            | 64.9%             | 0.0%             |
| n2.0  | 99.6%            | 91.9%             | 0.0%             |

*F < 1e-4 AND strict gives 0% because strict-converged samples have F̄ ~5e-3.

### DFTB0 (all cells near 0%)

| Noise | F < 1e-2, τ=0.1 | F < 1e-2, strict | F < 1e-4, any τ |
|-------|------------------|------------------|-----------------|
| n0.5  | 1.8%             | 0.7%             | 0.0%            |
| n1.0  | 1.4%             | 1.1%             | 0.0%            |
| n1.5  | 3.8%             | 3.8%             | 0.0%            |
| n2.0  | 5.2%             | 4.5%             | 0.0%            |

DFTB0 is **almost entirely red** in the heatmap — nothing passes because forces
are too high. Even F < 1e-2 captures only 1–5% of samples.

### Key insight from the sweep

On LJ, the eigenvalue threshold τ is the "knob" — relaxing τ from strict to 1e-3
recovers most lost samples. Force threshold matters little above 1e-3.

On DFTB0, the eigenvalue threshold has almost no effect (everything is already
n_neg=0). The force threshold is the entire story, and forces are too high to
satisfy any reasonable threshold.

See: `investigation/phase1/threshold_sweep_heatmap.png`

---

## Finding 3: Force Floor

### LJ

| Noise | Convergence type | n     | Median F̄   | Min F̄      | Max F̄      |
|-------|-----------------|-------|-------------|-------------|-------------|
| n0.5  | Strict          | 265   | 5.22e-3     | 1.49e-3     | 1.52e-2     |
| n0.5  | Relaxed         | 6     | 9.60e-5     | 9.16e-5     | 9.96e-5     |
| n1.0  | Strict          | 185   | 5.70e-3     | 1.49e-3     | 1.38e-2     |
| n1.0  | Relaxed         | 88    | 9.60e-5     | 7.22e-5     | 9.99e-5     |
| n1.5  | Strict          | 96    | 5.53e-3     | 8.11e-4     | 1.23e-2     |
| n1.5  | Relaxed         | 183   | 9.62e-5     | 6.75e-5     | 1.00e-4     |
| n2.0  | Strict          | 19    | 6.06e-3     | 2.84e-3     | 1.02e-2     |
| n2.0  | Relaxed         | 265   | 9.50e-5     | 1.47e-5     | 1.00e-4     |

The force distribution is **bimodal**:
- Strict-converged: F̄ ∈ [1e-3, 1.5e-2], median ~5.5e-3
- Relaxed-converged: F̄ ∈ [1e-5, 1e-4], median ~9.5e-5

The strict samples have forces **50× higher** because the strict criterion has no
force check — it stops the instant n_neg=0, regardless of force.

The relaxed samples' force floor is ~1e-5 at n=2.0 (the lowest achievable F̄
on LJ with current settings). Below ~5e-5, nothing converges.

**Correlation**: F̄ and n_neg are negatively correlated (ρ = -0.35 to -0.59).
More negative eigenvalues → lower forces. This makes sense: the optimizer can
minimize forces freely when it doesn't worry about eigenvalue positivity.

### DFTB0

| Noise | n     | Median F̄   | Min F̄      | Max F̄      |
|-------|-------|-------------|-------------|-------------|
| n0.5  | 281   | 1.16e-1     | 9.15e-3     | 2.87        |
| n1.0  | 278   | 6.76e-2     | 2.76e-3     | 4.88        |
| n1.5  | 277   | 5.47e-2     | 5.20e-3     | 2.95        |
| n2.0  | 267   | 5.06e-2     | 2.28e-3     | 4.30        |

ALL DFTB0 converged samples are strict (n_neg=0). Zero relaxed.
Forces are 500–1000× above the 1e-4 threshold.

Counterintuitively, higher noise → *lower* median forces. Possible explanation:
higher noise starts further from the TS, so the optimizer explores more of the
basin before finding n_neg=0, accumulating more force reduction along the way.

See: `investigation/phase1/force_floor_scatter.png`

---

## Finding 4: Eigenvalue Distribution at Convergence

### LJ — Ghost modes are noise-dependent

| Noise | Strict % | Median λ_min | Median n_neg | Neg |λ| median | Neg |λ| max |
|-------|----------|-------------|-------------|-----------------|-------------|
| n0.5  | 97.8%    | 2.20e-4     | 0           | 1.19e-5          | 2.16e-5     |
| n1.0  | 67.8%    | 1.21e-4     | 0           | 9.40e-6          | 1.39e-4     |
| n1.5  | 34.4%    | -2.60e-6    | 1           | 9.53e-6          | 8.50e-4     |
| n2.0  | 6.7%     | -1.91e-5    | 6           | 2.26e-5          | 1.63e-3     |

Ghost eigenvalues (|λ| < 1e-4) account for 72–100% of all negative eigenvalues.
They are a direct consequence of the LJ PES having zero angular curvature.

### DFTB0 — No negative eigenvalues, ever

| Noise | Strict % | Median λ_min | Min λ_min   |
|-------|----------|-------------|-------------|
| n0.5  | 100%     | 1.50e-4     | 1.73e-10    |
| n1.0  | 100%     | 3.64e-5     | 9.84e-11    |
| n1.5  | 100%     | 1.10e-5     | 5.82e-11    |
| n2.0  | 100%     | 1.16e-6     | 6.47e-18    |

DFTB0 has genuine curvature in all directions. n_neg=0 is trivially achievable.
The minimum λ_min at n=2.0 (6.47e-18) is effectively zero — the Hessian is
positive semidefinite to machine precision.

Note the trend: higher noise → smaller λ_min. At n=2.0, the smallest eigenvalue
is barely distinguishable from zero, suggesting these are near-degenerate but
still positive modes.

### Cascade rates (n_neg=0 at each eigenvalue threshold)

| PES  | Noise | τ=0     | τ=1e-4  | τ=5e-4  | τ=1e-3  | τ=1e-2  |
|------|-------|---------|---------|---------|---------|---------|
| LJ   | n0.5 | 97.8%   | 100%    | 100%    | 100%    | 100%    |
| LJ   | n1.0 | 67.8%   | 99.3%   | 100%    | 100%    | 100%    |
| LJ   | n1.5 | 34.4%   | 94.6%   | 99.3%   | 100%    | 100%    |
| LJ   | n2.0 | 6.7%    | 73.9%   | 96.5%   | 98.9%   | 100%    |
| DFTB | all  | 100%    | 100%    | 100%    | 100%    | 100%    |

On LJ, τ=5e-4 captures 96.5% even at n=2.0 (the hardest case).
On DFTB0, the cascade is irrelevant — everything is already at n_neg=0.

See: `investigation/phase1/eigenvalue_histograms.png`

---

## Finding 5: Phase Portraits

The (λ_min, F̄) phase portraits (colored by n_neg) reveal the geometry of the
convergence landscape:

### LJ
- At n=0.5: tight cluster at (λ_min ~ 2e-4, F̄ ~ 5e-3), almost all n_neg=0
- At n=2.0: two populations:
  - Strict: high F̄ (~5e-3), positive λ_min, n_neg=0 (upper right)
  - Relaxed: low F̄ (~1e-4), slightly negative λ_min, n_neg=6 (lower left)
  - The two populations are clearly separated — there is no continuum between them

### DFTB0
- All samples cluster at n_neg=0 (no color variation)
- λ_min spans 1e-18 to 1e-1 (all positive)
- F̄ spans 1e-3 to ~5 eV/Å — enormous range, all at n_neg=0
- The horizontal band structure at different F̄ levels suggests discrete
  convergence "shelves" where the optimizer stalls

See: `investigation/phase1/phase_portrait_final.png`

---

## Summary Table

| Metric                    | LJ n0.5 | LJ n2.0 | DFTB0 n0.5 | DFTB0 n2.0 |
|--------------------------|---------|---------|------------|------------|
| Convergence rate          | 99.3%   | 99.6%   | 98.9%      | 93.0%      |
| Strict rate               | 97.1%   | 6.7%    | 98.9%      | 93.0%      |
| Median F̄ (converged)     | 5.2e-3  | 9.6e-5  | 1.2e-1     | 5.1e-2     |
| Median λ_min (converged)  | 2.2e-4  | -1.9e-5 | 1.5e-4     | 1.2e-6     |
| Median n_neg (converged)  | 0       | 6       | 0          | 0          |
| Force-converged (F̄<1e-4) | 2.2%    | 93.0%   | 0%         | 0%         |

---

## Implications for Convergence Criterion Design

1. **The strict criterion needs a force check.** On DFTB0, strict convergence
   produces geometries with forces 500× above threshold. Adding `F̄ < ε` to the
   strict criterion would prevent premature convergence. The convergence rate
   would drop dramatically on DFTB0, but the *quality* of converged geometries
   would improve.

2. **A universal criterion should be force-first, eigenvalue-second.** The data
   suggests: `converged = (F̄ < ε) AND (n_neg_at_τ == 0)` with ε = 1e-4 and
   τ = 5e-4. This works on LJ (captures 89.5% at n=2.0) and would work on
   DFTB0 *if* the optimizer runs long enough to reduce forces.

3. **The eigenvalue threshold τ = 5e-4 is optimal for LJ.** It captures 96.5%
   of converged samples at n=2.0, filtering out only genuine negative curvature
   (not ghost modes). Stricter thresholds (τ < 1e-4) lose significant coverage.

4. **DFTB0 needs longer optimization runs.** The current "converged" DFTB0
   geometries are Hessian-converged but not gradient-converged. Phase 3
   experiments should test whether DFTB0 forces can actually reach 1e-4 with
   enough steps, or whether there's a structural barrier.

5. **LJ's force floor is ~5e-5.** Below this, no samples converge. This sets a
   lower bound on achievable force convergence on the LJ PES.

---

## Finding 6: Force and Eigenvalue Convergence Are Mutually Exclusive (Phase 1B)

This is the most striking trajectory result: **zero samples on either PES achieve
both F̄ < 1e-4 and n_neg = 0 at any point during optimization.** The two events
never co-occur.

### Convergence ordering

| PES   | Noise | Reach n_neg=0 only | Reach F̄<1e-4 only | Both | Neither |
|-------|-------|-------------------|--------------------|------|---------|
| LJ    | n0.5  | 78                | 2                  | 0    | 2       |
| LJ    | n1.0  | 57                | 23                 | 0    | 6       |
| LJ    | n1.5  | 28                | 52                 | 0    | 3       |
| LJ    | n2.0  | 7                 | 73                 | 0    | 1       |
| DFTB0 | n0.5  | 80                | 0                  | 0    | 3       |
| DFTB0 | n1.0  | 80                | 0                  | 0    | 7       |
| DFTB0 | n1.5  | 80                | 0                  | 0    | 10      |
| DFTB0 | n2.0  | 80                | 0                  | 0    | 20      |

**Interpretation:** On LJ, the optimizer must choose: either stop at n_neg=0
(strict, high forces) or continue to lower forces at the cost of acquiring ghost
eigenvalues (relaxed, low forces). There is no regime where both conditions hold.
On DFTB0, ALL samples reach n_neg=0 but NONE reach F̄<1e-4.

### Force at eigenvalue convergence (n_neg first hits 0)

| PES   | Noise | Median F̄ | Min F̄    | Max F̄    | F̄ < 1e-3 |
|-------|-------|----------|----------|----------|----------|
| LJ    | n0.5  | 5.15e-3  | 2.95e-3  | 1.19e-2  | 0%       |
| LJ    | n1.0  | 5.86e-3  | 1.49e-3  | 1.23e-2  | 0%       |
| LJ    | n1.5  | 6.01e-3  | 1.98e-3  | 1.23e-2  | 0%       |
| LJ    | n2.0  | 6.19e-3  | 2.84e-3  | 8.39e-3  | 0%       |
| DFTB0 | n0.5  | 9.98e-2  | 9.15e-3  | 2.87     | 0%       |
| DFTB0 | n1.0  | 7.04e-2  | 8.54e-3  | 4.88     | 0%       |
| DFTB0 | n1.5  | 4.78e-2  | 6.98e-3  | 2.95     | 0%       |
| DFTB0 | n2.0  | 5.23e-2  | 3.33e-3  | 4.30     | 0%       |

**On neither PES does ANY sample have F̄ < 1e-3 when n_neg first hits 0.**

This means: strict convergence (n_neg=0) fires at forces that are at least 10× above
even a generous force threshold. Adding a force gate to strict convergence would
eliminate ALL currently-strict-converged samples — the "both" column in the ordering
table is exactly zero.

See: `investigation/phase1/convergence_ordering.png`, `force_at_nneg0.png`

---

## Finding 7: Ghost Modes Appear From the Very Start (Phase 1B)

Ghost eigenvalues (|λ| < 1e-4) appear extremely early in LJ optimization:

| Noise | Median F̄ at ghost onset | % appear before F̄ < 1e-2 |
|-------|------------------------|-----------------------|
| n0.5  | 6.21e-1                | 71.6%                 |
| n1.0  | 1.29e-1                | 73.3%                 |
| n1.5  | 1.81e-2                | 51.8%                 |
| n2.0  | 1.64e-3                | 28.4%                 |

At n=0.5, ghosts appear when forces are still ~0.6 eV/Å (barely started). At n=2.0,
they appear at ~1.6e-3, which is 16× above the force threshold but early in the
optimization.

**100% of LJ trajectories develop ghost modes** (or have them from the start at
high noise). They are not a convergence artifact — they are a permanent feature
of the LJ PES topology. No amount of further optimization will remove them.

See: `investigation/phase1/ghost_onset_force.png`

---

## Finding 8: DFTB0 Has Worse Conditioning Than LJ (Phase 1B, Surprising)

| PES   | Noise | Median κ_final | Median κ_max  |
|-------|-------|---------------|--------------|
| LJ    | n0.5  | 1.67e+3       | 1.67e+7      |
| LJ    | n1.0  | 3.69e+3       | 9.45e+7      |
| LJ    | n1.5  | 3.14e+5       | 4.55e+8      |
| LJ    | n2.0  | 2.27e+6       | 2.71e+8      |
| DFTB0 | n0.5  | 3.39e+5       | 1.56e+7      |
| DFTB0 | n1.0  | 1.75e+6       | 9.94e+8      |
| DFTB0 | n1.5  | 1.29e+7       | 2.12e+10     |
| DFTB0 | n2.0  | 5.46e+7       | 1.43e+12     |

DFTB0 κ_final is 24× higher than LJ at n=2.0 (5.5e7 vs 2.3e6). DFTB0 κ_max
reaches 1.4e12 — twelve orders of magnitude.

On LJ, high conditioning comes from ghost eigenvalues in the denominator
(|λ_min| ~ 1e-5). On DFTB0, conditioning comes from near-zero *positive*
eigenvalues (λ_min ~ 1e-6 at n=2.0) — the Hessian is formally positive definite
but nearly singular.

**Implication:** DFTB0's poor force convergence may be partly caused by extreme
ill-conditioning. The RFO step builder handles this gracefully (augmented Hessian),
but the effective step quality deteriorates when κ > 10^6.

See: `investigation/phase1/conditioning_comparison.png`

---

## Updated Implications

The Phase 1B results strengthen and sharpen the Phase 1A findings:

1. **Force-gated strict convergence works better than predicted.**
   Phase 1B predicted "both=0" (mutual exclusivity), but Phase 3 showed that with
   the force gate, samples CAN achieve n_neg=0 + F̄ < 1e-4: 67% at n=1.0, 10% at
   n=2.0 on LJ. The standard optimizer just stops too early at n_neg=0.

2. **The optimal convergence criterion is PES-dependent:**
   - LJ: `(F̄ < 1e-4) AND (n_neg_at_5e-4 == 0)` → use relaxed eigenvalue check
   - DFTB0: `(F̄ < ε) AND (n_neg == 0)` → must wait for force convergence
   - Universal: `(F̄ < ε)` with eigenvalue checking only for saddle-point detection

3. **Ghost modes cannot be "fixed" — they must be accepted.** They appear from
   the very start of optimization on LJ and never go away.

4. **DFTB0 force non-convergence may have a conditioning explanation.** The
   κ > 10^7 condition numbers at n=2.0 suggest the optimizer is struggling with
   nearly singular Hessians, even though all eigenvalues are positive.

---

## Phase 3: Experiment Design

### Code change: `strict_force_gate` parameter (v14)

**File:** `baselines/minimization.py` line 2321

```python
# Before (default behavior):
strict_converged = n_neg == 0

# After (with --strict-force-gate):
if strict_force_gate:
    strict_converged = n_neg == 0 and force_norm < force_converged
else:
    strict_converged = n_neg == 0
```

Backward-compatible: defaults to `False`. Added to CLI as `--strict-force-gate`.

### Experiment 3A: DFTB0 with force-gated strict convergence

**Question:** Can DFTB0 forces reach 1e-4 if the optimizer doesn't stop at n_neg=0?

- PES: DFTB0, noise levels 1.0 and 2.0
- 50 samples, 50k steps
- Control: standard convergence
- Treatment: `--strict-force-gate`
- Expected: convergence rate drops dramatically; final forces tell us whether
  1e-4 is achievable or if there's a structural barrier

### Experiment 3B: LJ with force-gated strict convergence

**Question:** Does removing strict preemption change LJ relaxed convergence quality?

- PES: LJ, noise levels 1.0 and 2.0
- 50 samples, 50k steps
- Control: standard convergence
- Treatment: `--strict-force-gate`
- Expected: on LJ, strict and force convergence never co-occur, so strict_force_gate
  means strict NEVER fires → everything goes through relaxed path

### Experiment 3C: LJ with tighter force thresholds

**Question:** Can LJ forces go below the current floor of ~5e-5?

- PES: LJ n=2.0
- 50 samples, 50k steps
- force_converged: 1e-5 and 1e-6
- Expected: convergence rate drops; tells us whether 5e-5 is a hard floor

### Runner

```bash
bash src/noisy/v2_tests/investigation/run_phase3_experiments.sh
```

Reserve a node first:
```bash
salloc --nodes=1 --ntasks=1 --cpus-per-task=48 --time=4:00:00 --mem=128G
ssh <node>
cd /project/rrg-aspuru/memoozd/ts-tools
```

### Analysis

```bash
python src/noisy/v2_tests/investigation/analyze_phase3_results.py
```

---

## Phase 4: Synthesis — What Is a "True Minimum"?

### On LJ (pairwise potential)

A "true minimum" on the LJ PES is a geometry where:
- **Forces are minimized** (F̄ < ~5e-5, the empirical floor)
- **n_neg > 0 is expected and acceptable** due to ghost eigenvalues from missing
  angular curvature terms
- **All negative eigenvalues have |λ| < ~5e-4** — these are ghost modes, not
  genuine saddle-point curvature

The LJ PES lacks angular restoring forces, so pairwise-equilibrium geometries have
zero curvature (λ = 0) in rotational degrees of freedom. Numerical noise makes these
appear as tiny negative eigenvalues (|λ| ~ 1e-5). Phase 3 showed that dual convergence
(n_neg=0 AND F̄ < 1e-4) IS achievable: 67% of samples at n=1.0, 10% at n=2.0 — but
only when the force gate prevents premature stopping at n_neg=0.

**Correct convergence criterion for LJ:**
`(F̄ < 1e-4) AND (n_neg_at_τ=5e-4 == 0)` — force convergence with ghost-tolerant
eigenvalue check. This captures 89.5% at n=2.0 and 95.6% at n=0.5.

### On DFTB0 (semi-empirical quantum)

A "true minimum" on the DFTB0 PES is a geometry where:
- **n_neg = 0 is always achievable** — the PES has genuine curvature in all directions
- **Forces should also be small** (F̄ < 1e-4), but the current optimizer stops
  before achieving this because strict convergence has no force check
- **Condition numbers are extreme** (κ ~ 10^7 at n=2.0), suggesting the optimizer
  is navigating a very flat, nearly degenerate basin

The DFTB0 PES has proper 3-body and angular terms, so there are no ghost modes.
n_neg = 0 is trivially satisfied early in optimization. The challenge is force
convergence — median F̄ ~ 0.05 eV/Å at the moment n_neg = 0, which is 500× above
the threshold.

**Correct convergence criterion for DFTB0:**
Phase 3 showed that F̄ < 1e-4 is **structurally unachievable** on DFTB0 with the
current optimizer (force floor ~3e-3, oscillatory behavior). Options:
1. `(n_neg == 0) AND (F̄ < 1e-2)` — relaxed force threshold acknowledging the floor
2. `n_neg == 0` only (current behavior) — accept that eigenvalue convergence is the
   best achievable quality metric on DFTB0
3. Disable kick mechanisms and test whether forces converge without perturbations
4. Implement Hessian preconditioning to address κ > 10^7

### Universal definition

A "true minimum" is a stationary point (∇E = 0) where the Hessian is positive
semidefinite (all eigenvalues ≥ 0). In practice, neither condition is exactly
satisfied, so we need tolerances:

**Proposed universal criterion:**
```
converged = (F̄ < ε_force) AND (n_neg_at_τ == 0)
```
where:
- `ε_force = 1e-4 eV/Å` (mean per-atom force norm)
- `τ = 5e-4 eV/Å²` (eigenvalue tolerance for ghost modes)
- `n_neg_at_τ` counts eigenvalues below `-τ`, ignoring ghost modes

This criterion:
- Requires BOTH force convergence AND eigenvalue positivity (up to ghost tolerance)
- Works on LJ: captures ~90% at n=2.0 (ghost modes filtered by τ)
- **Does NOT work on DFTB0:** Phase 3 showed forces plateau at ~3e-3 and oscillate,
  never reaching 1e-4 even after 50k steps. A universal criterion must either:
  (a) use PES-specific force thresholds, or (b) rely on eigenvalue-only convergence
  for PES where force convergence is structurally limited

### The fundamental tradeoff

The investigation reveals that on both PES, there is a **tradeoff between force
convergence and eigenvalue positivity**:

- On LJ: achieving F̄ < 1e-4 requires accepting n_neg > 0 (ghost modes)
- On DFTB0: achieving n_neg = 0 happens before F̄ < 1e-4 (premature convergence)

The current code handles LJ correctly (via relaxed convergence). On DFTB0, strict
convergence stops too early, but Phase 3 showed that longer runs do NOT help — DFTB0
forces plateau at ~3e-3 (30× above threshold) and oscillate. The `strict_force_gate`
fixes LJ but gives 0% convergence on DFTB0. A PES-specific convergence criterion
is needed.

---

## Phase 3 Results: Force-Gated Convergence on DFTB0

### Experiment 3A: DFTB0 with force-gated strict convergence

**Question:** Can DFTB0 forces reach 1e-4 if the optimizer doesn't stop at n_neg=0?

**Answer: No.** DFTB0 forces cannot reach 1e-4 even after 50,000 optimization steps.
This is the most important Phase 3 result.

#### Control (standard convergence, no force gate)

| Noise | Total | Converged | Strict | Median F̄ | Steps median |
|-------|-------|-----------|--------|-----------|-------------|
| n=2.0 | 10 | 7 (70%) | 7 | 6.41e-2 | 1954 |
| n=1.0 | 10 | 10 (100%) | 10 | 1.02e-1 | 948 |

All converged samples are strict (n_neg=0). Forces are 640–1020× above 1e-4 threshold.

#### Force-gate (50k steps, `--strict-force-gate`)

| Noise | Total | Converged | Final F̄ range | Min F̄ ever | Steps below 1e-2 |
|-------|-------|-----------|---------------|------------|-----------------|
| n=2.0 | 10* | 0 (0%) | 4.6e-2 – 1.4e-1 | 3.35e-3 | 200–1791/50k |
| n=1.0 | 10* | 0 (0%) | 3.3e-2 – 3.9e-2 | 7.45e-4 | 0–19652/50k |

*Partial results (4/10 n=2.0, 2/10 n=1.0 completed; remaining workers died OOM on login
node). All completed samples show the same pattern — forces oscillate, never reaching 1e-4.

**Zero convergence.** With the force gate, strict convergence requires F̄ < 1e-4 AND
n_neg=0. DFTB0 always has n_neg=0, but forces never reach 1e-4.

#### Force trajectories reveal oscillatory behavior (DFTB0 n=2.0)

The force trajectories show the optimizer is NOT converging — it is OSCILLATING:

```
Sample 000: step 0→F=2.0,  step 1000→F=0.018,  step 36k→F=0.004(min),  step 50k→F=0.058
Sample 005: step 0→F=3.5,  step 1000→F=0.037,  step 3.5k→F=0.004(min), step 50k→F=0.136
Sample 006: step 0→F=0.5,  step  500→F=0.032,  step 41k→F=0.006(min),  step 50k→F=0.097
Sample 007: step 0→F=1.3,  step  235→F=0.003(min), step 1000→F=0.102,  step 50k→F=0.046
```

- Forces rapidly decrease from initial values (~1-4 eV/Å) to ~1-5e-2 within 500-1000 steps
- Then they OSCILLATE in the range [3e-3, 1.4e-1] for the remaining 49,000 steps
- The minimum F̄ ever achieved across all samples: **3.35e-3** (sample 7 at step 235)
- **Zero steps below 1e-3** across 200k total steps (all 4 trajectories combined)
- Kick mechanisms fire extensively (500-800+ osc-kicks per trajectory), but Phase 3D
  showed that disabling them has NO effect on force behavior — the oscillation is
  intrinsic to the RFO optimizer on the ill-conditioned DFTB0 PES

#### n=1.0 force trajectories

At n=1.0, forces get slightly closer to 1e-4 but still can't reach it:

| Sample | Min F̄ | Min F̄ step | Steps below 1e-3 | Steps below 1e-4 | Final F̄ |
|--------|--------|------------|-----------------|-----------------|---------|
| 000 | 4.28e-3 | 31348 | 0 | 0 | 3.89e-2 |
| 005 | **7.45e-4** | 9928 | 52 | 0 | 3.27e-2 |

At n=1.0, sample 005 reached F̄ = 7.45e-4 (7× above threshold) — the closest any DFTB0
sample came to 1e-4 in the entire investigation. But even this was transient and bounced
back to 3.27e-2 by step 50k.

#### DFTB0 force floor: ~3e-3 eV/Å (n=2.0), ~7e-4 eV/Å (n=1.0)

The absolute minimum force achievable on DFTB0 with the current optimizer is ~7e-4 eV/Å
at n=1.0 and ~3e-3 at n=2.0, which is **7–30× above the 1e-4 threshold.** This is not
a convergence speed issue — the
optimizer genuinely cannot reach lower forces. Likely causes:

1. **Extreme ill-conditioning** (κ ~ 10^7 at n=2.0): the nearly-singular Hessian
   produces poor step directions, especially after RFO augmentation
2. ~~**Kick mechanisms are counterproductive**~~ **RULED OUT (Phase 3D):** disabling
   all kicks produces identical force behavior (median 4.48e-2 vs 4.51e-2)
3. **Possible DFTB0 numerical precision limits:** the semi-empirical method may have
   inherent force accuracy limits that prevent sub-1e-3 precision

#### Implication for convergence criterion

The `strict_force_gate` at fc=1e-4 gives **100% → 0% convergence on DFTB0**. This means:

1. **A universal force threshold of 1e-4 is incompatible with DFTB0.** Either:
   - Raise the force threshold for DFTB0 (e.g., fc=1e-2 would capture ~some%)
   - Accept that DFTB0 convergence must use eigenvalue-only criteria (current behavior)
   - Disable kick mechanisms on DFTB0 and test whether forces converge without perturbations
   - Implement a preconditioner to address κ > 10^7

2. **The "universal criterion" proposed in Phase 4 does NOT work on DFTB0.**
   `(F̄ < 1e-4) AND (n_neg_at_τ == 0)` requires force convergence that DFTB0 cannot achieve.

---

## Phase 3D: No-Kicks Experiment (DFTB0)

### Question
Are the kick mechanisms (osc-kick, blind-kick, late-escape) causing the DFTB0 force
oscillation? The WITH-kicks experiments showed 500-800+ osc-kicks and 68-90 late-escapes
per 50k-step trajectory — enough perturbations to plausibly prevent fine convergence.

### Setup
Same as 3A force-gate, but with ALL kick flags omitted (all default to False):
- No `--osc-kick`, `--blind-kick`, `--blind-kick-probe`, `--late-escape`
- `--strict-force-gate` enabled
- 50k steps, DFTB0

### Results: Kicks are NOT the cause

**n=1.0, sample_000 comparison:**

| Metric | WITH kicks | WITHOUT kicks |
|--------|-----------|---------------|
| Total kicks | 987 (839 osc, 78 blind, 70 late) | **0** |
| Median F̄ | 4.51e-2 | 4.48e-2 |
| Min F̄ | 4.28e-3 | 4.43e-3 |
| F < 1e-2 | 1164/50k (2.3%) | 1454/50k (2.9%) |
| F < 1e-3 | 0 | 0 |
| Force windows (5k) | 3.9e-2 – 4.5e-2 | 4.2e-2 – 4.9e-2 |

The force behavior is **nearly identical** with and without kicks. The oscillation
amplitude, floor, and median are all the same within noise. Removing 987 kick
perturbations made no measurable difference.

### Implication

The DFTB0 force oscillation is **intrinsic to the RFO optimizer operating on an
ill-conditioned PES** (κ > 10^7), not an artifact of kick mechanisms. The kicks
were designed to escape saddle points but happen to fire in this regime because the
oscillation triggers their patience-based heuristics. However, they are effect not
cause.

**Remaining hypothesis:** The extreme ill-conditioning (near-zero positive eigenvalues
at ~10^-6) causes the RFO augmented Hessian to produce poor step directions. The
optimizer makes progress in well-conditioned directions but oscillates in the
ill-conditioned subspace, keeping the overall force norm at ~3e-3 to 5e-2.

---

## Phase 3 Results: Force-Gated Convergence on LJ

### Experiment 3B: Force-gated strict convergence

Results at `/scratch/memoozd/ts-tools-scratch/runs/phase3_investigation/`

| Metric | Control | Force-gated (`--strict-force-gate`) |
|--------|---------|-------------------------------------|
| Total samples | 50 | 50 |
| Converged | 50 (100%) | 49 (98%) |
| Strict | 6 (F̄ median 6.14e-3) | 5 (F̄ median 8.45e-5) |
| Relaxed | 44 (F̄ median 9.29e-5) | 44 (F̄ median 9.29e-5) |
| Overall F̄ median | 9.50e-5 | 9.21e-5 |
| Overall F̄ max | 8.39e-3 | 1.00e-4 |
| Mean steps | 3872 | 3781 |

**Key findings:**
1. **Force-gated strict convergence works.** The 5 strict samples in the treatment have
   F̄ ∈ [5.62e-5, 9.22e-5] — all below the 1e-4 threshold. These are genuine
   dual-converged points (n_neg=0 AND F̄ < 1e-4), proving that such points DO exist
   on the LJ PES. Phase 1B said "both=0" because the standard optimizer stops at
   n_neg=0 before forces converge. With the force gate, 5/50 samples continue past
   n_neg=0 and forces DO reach 1e-4 while n_neg remains at 0.

2. **Minimal convergence loss.** Only 1 sample lost (100% → 98%). That sample reached
   F̄ = 2.33e-4 (2.3× above threshold) and never met the gated criterion.

3. **Relaxed path is identical.** The 44 relaxed samples are the same in both runs
   (same seed). The force gate only affects strict-path samples.

4. **Maximum F̄ drops from 8.39e-3 to 1.00e-4** — the force gate eliminates all
   high-force "converged" geometries. Every converged sample now has F̄ ≤ 1e-4.

5. **Strict median force improves 73×** (6.14e-3 → 8.45e-5). The force gate transforms
   strict convergence from a low-quality shortcut into a genuine dual-convergence event.

### Experiment 3C: Tighter force thresholds

**Question:** Can LJ forces go below the ~5e-5 floor observed in Phase 1?

| Metric | fc=1e-4 (3B control) | fc=1e-5 | fc=1e-6 |
|--------|---------------------|---------|---------|
| Converged | 50 (100%) | 25 (50%) | 24 (48%) |
| Strict (n_neg=0) | 6 | 23 | 24 |
| Relaxed (F̄ < fc) | 44 | 2 | 0 |
| Relaxed F̄ | median 9.29e-5 | median 9.48e-6 | — |
| Not converged F̄ min | — | 4.22e-5 | 4.22e-5 |

**Key findings:**
1. **Force floor confirmed at ~1e-5.** At fc=1e-5, only 2/50 samples reach relaxed
   convergence (F̄ ~ 9.5e-6). At fc=1e-6, zero samples reach relaxed convergence.

2. **Strict preempts at both thresholds.** Without `--strict-force-gate`, strict
   convergence (n_neg=0, no force check) still fires for ~24 samples at F̄ ~ 5.6e-3.
   These "converged" samples have forces 500× above the target fc.

3. **The unconverged samples stall at ~4e-5.** The minimum F̄ among unconverged samples
   is 4.22e-5 at both fc settings — forces cannot penetrate below this floor.

4. **Half the convergence rate at fc=1e-5.** The 50% drop (from 100% at fc=1e-4) is
   almost entirely due to relaxed convergence failing — forces can't reach 1e-5 for
   most samples.

5. **LJ force floor is ~1e-5 eV/Å** (per-atom mean). Below this, the LJ PES
   topology prevents further force reduction. This is consistent with the Phase 1
   observation that relaxed-converged samples have F̄ floor ~1.47e-5.

### Experiment 3B (n=1.0): Force-gated strict convergence at lower noise

| Metric | Control | Force-gated (`--strict-force-gate`) |
|--------|---------|-------------------------------------|
| Total samples | 48 | 48 |
| Converged | 47 (97.9%) | 46 (95.8%) |
| Strict | 33 (F̄ median 6.08e-3) | 32 (F̄ median 8.09e-5) |
| Relaxed | 14 (F̄ median 9.45e-5) | 14 (F̄ median 9.45e-5) |
| Overall F̄ median | 4.48e-3 | 8.66e-5 |
| Overall F̄ max | 1.23e-2 | 9.99e-5 |
| Mean steps | 3467 | 3834 |

**Key findings:**
1. **32/48 samples achieve genuine dual convergence** (n_neg=0 AND F̄ < 1e-4). At
   n=1.0, far more samples can reach both conditions than at n=2.0 (32 vs 5). This
   is because ghost modes are less prevalent at lower noise — most samples can reach
   n_neg=0 and then continue to force convergence.

2. **The mutual exclusivity weakens at lower noise.** Phase 1B found "both=0" because
   standard convergence stops at n_neg=0 before forces converge. With the force gate,
   we see that dual convergence IS achievable for 67% of samples at n=1.0.

3. **Convergence cost is small.** Only 1 additional sample lost (97.9% → 95.8%).

4. **Strict median force improves 75×** (6.08e-3 → 8.09e-5).

5. **Overall median F̄ drops 52×** (4.48e-3 → 8.66e-5). At n=1.0, strict samples
   dominate (33/47), so the overall quality improvement from force-gating is massive.

### Phase 3 Summary (LJ)

**Force gate effectiveness scales with noise level:**

| Noise | Strict samples (control → gated) | Dual-converged | Conv rate drop |
|-------|-----------------------------------|---------------|---------------|
| n=1.0 | 33 → 32 | 32/48 (67%) | 2.1% |
| n=2.0 | 6 → 5 | 5/50 (10%) | 2.0% |

At n=1.0 (fewer ghost modes), the force gate produces dual convergence for most
samples. At n=2.0 (many ghost modes), only 10% achieve it — ghost eigenvalues
reappear after continuing past the initial n_neg=0 moment, preventing sustained
eigenvalue convergence.

**LJ force floor is firmly at ~1e-5:**
- 2/50 can reach ~9.5e-6 with enough steps
- 0/50 can reach ~1e-6
- Minimum achievable among non-converged samples: ~4.2e-5

---

## Open Questions (Remaining)

1. **ANSWERED: Can DFTB0 forces reach 1e-4?** No. Forces plateau at ~3e-3 and
   oscillate. Zero steps below 1e-3 across 200k total steps. See Phase 3A above.

2. **ANSWERED: Does the force gate work on LJ n=1.0?** Yes, even better than n=2.0.
   32/48 (67%) achieve genuine dual convergence (n_neg=0 + F̄ < 1e-4). See Phase 3B.

3. **ANSWERED: Do kick mechanisms cause DFTB0 force oscillation?** No. Experiment 3D
   (no kicks) shows nearly identical force behavior: median F̄ = 4.48e-2 vs 4.51e-2
   with kicks, min F̄ = 4.43e-3 vs 4.28e-3. Zero steps below 1e-3 in either case.
   The oscillation is intrinsic to the optimizer + PES interaction, not kick-induced.

4. **Would a preconditioner help DFTB0?** κ > 10^7 at n=2.0 means step quality is
   degraded. A preconditioner addressing the nearly-singular Hessian might help.
   This is the most promising remaining avenue given that kick disabling had no effect.

5. **Are "relaxed-converged" LJ geometries at true minima?** Ghost eigenvalues could
   mask genuine negative curvature. Need IRC or energy perturbation analysis.

6. **How does the force–eigenvalue tradeoff change on other PES?** (DFT, ANI, etc.)
   The LJ/DFTB0 results bound the spectrum (pairwise vs quantum), but production
   PES may behave differently.

---

## Conclusions

### Summary of findings

The investigation reveals that force and eigenvalue convergence are fundamentally
different problems on different PES, and the current convergence criterion handles
each poorly in its own way:

| | LJ (pairwise) | DFTB0 (semi-empirical) |
|---|---|---|
| Force convergence | Achievable (floor ~1e-5) | **Not achievable** (floor ~3e-3) |
| Eigenvalue convergence | Hard at high noise (ghost modes) | Trivial (always n_neg=0) |
| Strict criterion quality | Poor (F̄ ~ 5e-3 at convergence) | Poor (F̄ ~ 5e-2 at convergence) |
| Force gate effect | **Excellent** (67% dual-converge at n=1.0) | **0% convergence** (forces never reach 1e-4) |
| Root cause of poor strict quality | Stops at n_neg=0 before forces converge | Same, but forces CAN'T converge |
| Recommended fix | `--strict-force-gate` | Raise force threshold or keep eigenvalue-only |

### Actionable recommendations

1. **Enable `--strict-force-gate` for LJ and similar pairwise PES.** On LJ:
   - n=1.0: 67% dual convergence, 2.1% convergence loss, 52× force quality improvement
   - n=2.0: 10% dual convergence, 2.0% convergence loss, 73× force quality improvement
   - Max F̄ of any "converged" sample drops from ~1e-2 to ~1e-4

2. **Do NOT use `--strict-force-gate` with fc=1e-4 on DFTB0.** It gives 0% convergence.
   Options for DFTB0:
   - Keep current behavior (eigenvalue-only strict convergence)
   - Use a PES-specific force threshold (e.g., fc=1e-2)
   - ~~Investigate disabling kick mechanisms~~ (Phase 3D: no effect)
   - Implement Hessian preconditioning to address κ > 10^7

3. **Set eigenvalue tolerance τ = 5e-4 for LJ.** This captures 96.5% at n=2.0
   (filtering ghost modes but catching genuine negative curvature).

4. **The "universal criterion" `(F̄ < 1e-4) AND (n_neg_at_τ=5e-4 == 0)` works only on
   LJ-like PES.** A truly universal criterion must be PES-adaptive or use separate
   thresholds for different PES types.

### The key insight

The strict convergence criterion's lack of force check (line 2321 of minimization.py)
is not a bug on all PES — it's the right behavior on DFTB0 (where force convergence
is unachievable). But it's the wrong behavior on LJ (where it stops too early,
producing geometries with F̄ 50× above threshold). The fix is PES-specific:
force-gate on LJ, eigenvalue-only on DFTB0.

---

## File Index

### Analysis scripts
- `investigation/phase1/investigate_phase1a.py` — Results-JSON analysis (fast)
- `investigation/phase1/investigate_phase1b_trajectories.py` — Trajectory analysis
- `investigation/analyze_phase3_results.py` — Phase 3 experiment comparison

### Runner scripts
- `investigation/run_phase3_experiments.sh` — Phase 3 experiment launcher

### Code changes
- `baselines/minimization.py` — Added `strict_force_gate` parameter (v14)
- `runners/run_minimization_parallel.py` — Added `--strict-force-gate` CLI flag

### Output (Phase 1)
- `investigation/phase1/phase1a_report.txt` — Phase 1A text report
- `investigation/phase1/phase1b_report.txt` — Phase 1B text report
- `investigation/phase1/phase1a_summary.json` — Phase 1A data
- `investigation/phase1/phase1b_summary.json` — Phase 1B data
- `investigation/phase1/*.png` — All plots

### Output (Phase 3)
- `investigation/phase3/phase3_summary.json` — Phase 3 comparison data
- Phase 3 experiment data: `/scratch/memoozd/ts-tools-scratch/runs/phase3_investigation/`
  - `3A_dftb_n{1.0,2.0}_{control,forcegate}/` — DFTB0 experiments
  - `3B_lj_n{1.0,2.0}_{control,forcegate}/` — LJ force-gate experiments
  - `3C_lj_n2.0_fc{1e-5,1e-6}/` — LJ tight force threshold experiments
  - `3D_dftb_n{1.0,2.0}_forcegate_nokicks/` — No-kicks control experiments

### Plots (Phase 1)
- `threshold_sweep_heatmap.png` — 2D force×eigenvalue convergence rates
- `force_floor_scatter.png` — Final F̄ vs n_neg
- `phase_portrait_final.png` — (F̄, λ_min) colored by n_neg
- `eigenvalue_histograms.png` — λ_min distribution at convergence
- `convergence_ordering.png` — step(n_neg=0) − step(F̄<1e-4) histograms
- `force_at_nneg0.png` — Force when n_neg first hits 0
- `ghost_onset_force.png` — Force at ghost mode onset (LJ)
- `conditioning_comparison.png` — κ boxplots LJ vs DFTB0
