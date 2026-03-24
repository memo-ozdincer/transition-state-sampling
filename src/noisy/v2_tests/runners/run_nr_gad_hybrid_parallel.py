#!/usr/bin/env python
"""Hybrid NR→GAD parallel runner: minimize first, then climb to TS.

Inspired by iHiSD (Su et al.): gradient flow first brings the geometry into
the basin of attraction of a saddle point, then saddle dynamics (GAD) locates
the transition state precisely.

Phase 1 — Newton-Raphson minimization (v13 with all bells and whistles)
  Converges to a local minimum: n_neg == 0 AND force < threshold.

Phase 2 — GAD saddle search from the converged minimum
  Climbs from the minimum to an index-1 saddle point.
  Convergence: n_neg == 1 (Morse index 1) on Eckart-projected Hessian.
  No tr_threshold filtering — only Eckart projection removes TR modes.

Two GAD variants:
  --gad-variant path     Path-based trust-region GAD (all bells and whistles)
  --gad-variant nopath   State-based adaptive-dt GAD (no trajectory history)

The switching criterion is state-based: GAD starts when NR converges (or
when --gad-on-nr-failure is set and NR exhausts its step budget).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.dependencies.common_utils import Transition1xDataset, UsePos, parse_starting_geometry
from src.noisy.multi_mode_eckartmw import _atomic_nums_to_symbols
from src.noisy.v2_tests.baselines.minimization import run_newton_raphson
from src.parallel.scine_parallel import ParallelSCINEProcessor
from src.parallel.utils import run_batch_parallel

# Import both GAD variants — they have different signatures
from src.noisy.v2_tests.runners.run_gad_baselines_parallel import (
    run_gad_baseline as _run_gad_path,
)
from src.noisy.v2_tests.runners.run_gad_baselines_parallel_nopath import (
    run_gad_baseline as _run_gad_nopath,
)

# ---------------------------------------------------------------------------
# Cascade evaluation (shared with minimization runner)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------
def create_dataloader(h5_path: str, split: str, max_samples: int):
    dataset = Transition1xDataset(
        h5_path=h5_path,
        split=split,
        max_samples=max_samples,
        transform=UsePos("pos_transition"),
    )
    if len(dataset) == 0:
        raise RuntimeError("No Transition1x samples loaded. Check h5 path and split.")
    return DataLoader(dataset, batch_size=1, shuffle=False)


# ---------------------------------------------------------------------------
# Core: run NR then GAD for a single sample
# ---------------------------------------------------------------------------
def run_single_sample(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    nr_params: Dict[str, Any],
    gad_params: Dict[str, Any],
    nr_n_steps: int,
    gad_n_steps: int,
    *,
    gad_variant: str,
    gad_on_nr_failure: bool,
    sample_id: str,
    formula: str,
    known_ts_coords: Optional[torch.Tensor] = None,
    known_reactant_coords: Optional[torch.Tensor] = None,
    known_product_coords: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    all_atomsymbols = _atomic_nums_to_symbols(atomic_nums)
    t0 = time.time()

    # =====================================================================
    # Phase 1: Newton-Raphson minimization
    # =====================================================================
    nr_result: Dict[str, Any] = {}
    nr_converged = False
    nr_final_coords = coords  # fallback

    try:
        nr_raw, _nr_trajectory = run_newton_raphson(
            predict_fn,
            coords,
            atomic_nums,
            all_atomsymbols,
            n_steps=nr_n_steps,
            max_atom_disp=nr_params["max_atom_disp"],
            force_converged=nr_params["force_converged"],
            min_interatomic_dist=nr_params["min_interatomic_dist"],
            nr_threshold=nr_params.get("nr_threshold", 8e-3),
            project_gradient_and_v=nr_params.get("project_gradient_and_v", False),
            purify_hessian=nr_params.get("purify_hessian", False),
            known_ts_coords=known_ts_coords,
            known_reactant_coords=known_reactant_coords,
            known_product_coords=known_product_coords,
            # v2
            lm_mu=nr_params.get("lm_mu", 0.0),
            anneal_force_threshold=nr_params.get("anneal_force_threshold", 0.0),
            cleanup_nr_threshold=nr_params.get("cleanup_nr_threshold", 0.0),
            cleanup_max_steps=nr_params.get("cleanup_max_steps", 50),
            log_spectrum_k=nr_params.get("log_spectrum_k", 10),
            # v3
            shift_epsilon=nr_params.get("shift_epsilon", 0.0),
            stagnation_window=nr_params.get("stagnation_window", 0),
            escape_alpha=nr_params.get("escape_alpha", 0.1),
            lm_mu_anneal_factor=nr_params.get("lm_mu_anneal_factor", 0.0),
            lm_mu_anneal_n_neg_leq=nr_params.get("lm_mu_anneal_n_neg_leq", 2),
            lm_mu_anneal_eval_leq=nr_params.get("lm_mu_anneal_eval_leq", 5e-3),
            neg_mode_line_search=nr_params.get("neg_mode_line_search", False),
            trust_radius_floor=nr_params.get("trust_radius_floor", 0.01),
            # v4
            neg_trust_floor=nr_params.get("neg_trust_floor", 0.0),
            blind_mode_threshold=nr_params.get("blind_mode_threshold", 0.0),
            blind_correction_alpha=nr_params.get("blind_correction_alpha", 0.02),
            aggressive_trust_recovery=nr_params.get("aggressive_trust_recovery", False),
            escape_bidirectional=nr_params.get("escape_bidirectional", False),
            mode_follow_eval_threshold=nr_params.get("mode_follow_eval_threshold", 0.0),
            mode_follow_alpha=nr_params.get("mode_follow_alpha", 0.15),
            mode_follow_after_steps=nr_params.get("mode_follow_after_steps", 2000),
            # v5 SPDN
            optimizer_mode=nr_params.get("optimizer_mode", ""),
            spdn_tau_hard=nr_params.get("spdn_tau_hard", 0.01),
            spdn_tau_soft=nr_params.get("spdn_tau_soft", 1e-4),
            spdn_diis_size=nr_params.get("spdn_diis_size", 8),
            spdn_diis_every=nr_params.get("spdn_diis_every", 5),
            spdn_momentum=nr_params.get("spdn_momentum", 0.0),
            # v7
            step_control=nr_params.get("step_control", "trust_region"),
            max_nr_weight=nr_params.get("max_nr_weight", 0.0),
            # v8 crossover
            crossover_mu_max=nr_params.get("crossover_mu_max", 0.0),
            crossover_n_neg_ref=nr_params.get("crossover_n_neg_ref", 3.0),
            crossover_force_ref=nr_params.get("crossover_force_ref", 0.1),
            # v9 relaxed convergence
            relaxed_eval_threshold=nr_params.get("relaxed_eval_threshold", 0.0),
            accept_relaxed=nr_params.get("accept_relaxed", False),
            # v10 ARC
            arc_sigma_init=nr_params.get("arc_sigma_init", 1.0),
            arc_sigma_min=nr_params.get("arc_sigma_min", 1e-4),
            arc_sigma_max=nr_params.get("arc_sigma_max", 1e4),
            arc_eta1=nr_params.get("arc_eta1", 0.1),
            arc_eta2=nr_params.get("arc_eta2", 0.9),
            arc_gamma1=nr_params.get("arc_gamma1", 2.0),
            arc_gamma2=nr_params.get("arc_gamma2", 0.5),
            gdiis_buffer_size=nr_params.get("gdiis_buffer_size", 0),
            gdiis_every=nr_params.get("gdiis_every", 5),
            gdiis_late_force_threshold=nr_params.get("gdiis_late_force_threshold", 0.0),
            schlegel_trust_update=nr_params.get("schlegel_trust_update", False),
            polynomial_linesearch=nr_params.get("polynomial_linesearch", False),
            # v12 kicks
            osc_kick=nr_params.get("osc_kick", False),
            osc_kick_scale=nr_params.get("osc_kick_scale", 0.1),
            osc_kick_patience=nr_params.get("osc_kick_patience", 3),
            osc_kick_cooldown=nr_params.get("osc_kick_cooldown", 50),
            blind_kick=nr_params.get("blind_kick", False),
            blind_kick_scale=nr_params.get("blind_kick_scale", 0.5),
            blind_kick_overlap_thresh=nr_params.get("blind_kick_overlap_thresh", 0.1),
            blind_kick_force_thresh=nr_params.get("blind_kick_force_thresh", 0.1),
            blind_kick_patience=nr_params.get("blind_kick_patience", 100),
            kick_eigvec_index=nr_params.get("kick_eigvec_index", 0),
            # v12b
            adaptive_kick_scale=nr_params.get("adaptive_kick_scale", False),
            adaptive_kick_C=nr_params.get("adaptive_kick_C", 0.1),
            blind_kick_probe=nr_params.get("blind_kick_probe", False),
            late_escape=nr_params.get("late_escape", False),
            late_escape_after=nr_params.get("late_escape_after", 15000),
            late_escape_alpha=nr_params.get("late_escape_alpha", 0.1),
            late_escape_cooldown=nr_params.get("late_escape_cooldown", 500),
        )

        nr_converged = bool(nr_raw.get("converged"))
        fc = nr_raw.get("final_coords")
        if isinstance(fc, torch.Tensor):
            nr_final_coords = fc.detach().cpu()

        nr_result = {
            "nr_converged": nr_converged,
            "nr_converged_step": nr_raw.get("converged_step"),
            "nr_final_energy": nr_raw.get("final_energy"),
            "nr_final_force_norm": nr_raw.get("final_force_norm"),
            "nr_total_steps": nr_raw.get("total_steps", nr_n_steps),
            "nr_final_n_neg_evals": nr_raw.get("final_n_neg_evals"),
            "nr_final_min_vib_eval": nr_raw.get("final_min_vib_eval"),
            "nr_total_osc_kicks": nr_raw.get("total_osc_kicks", 0),
            "nr_total_blind_kicks": nr_raw.get("total_blind_kicks", 0),
            "nr_total_late_escapes": nr_raw.get("total_late_escapes", 0),
            "nr_error": None,
        }
    except Exception as e:
        nr_result = {
            "nr_converged": False,
            "nr_converged_step": None,
            "nr_final_energy": None,
            "nr_final_force_norm": None,
            "nr_total_steps": 0,
            "nr_final_n_neg_evals": None,
            "nr_final_min_vib_eval": None,
            "nr_total_osc_kicks": 0,
            "nr_total_blind_kicks": 0,
            "nr_total_late_escapes": 0,
            "nr_error": str(e),
        }

    # =====================================================================
    # Switching decision
    # =====================================================================
    run_gad = nr_converged or gad_on_nr_failure

    # =====================================================================
    # Phase 2: GAD saddle search
    # =====================================================================
    gad_result: Dict[str, Any] = {}

    if run_gad:
        try:
            if gad_variant == "path":
                gad_raw = _run_gad_path(
                    predict_fn,
                    nr_final_coords,
                    atomic_nums,
                    n_steps=gad_n_steps,
                    dt=gad_params["dt"],
                    dt_control=gad_params["dt_control"],
                    dt_min=gad_params["dt_min"],
                    dt_max=gad_params["dt_max"],
                    max_atom_disp=gad_params["max_atom_disp"],
                    ts_eps=gad_params["ts_eps"],
                    stop_at_ts=gad_params.get("stop_at_ts", True),
                    min_interatomic_dist=gad_params["min_interatomic_dist"],
                    tr_threshold=gad_params.get("tr_threshold", 8e-3),
                    track_mode=gad_params.get("track_mode", True),
                    project_gradient_and_v=gad_params.get("project_gradient_and_v", True),
                    log_dir=gad_params.get("log_dir"),
                    sample_id=sample_id,
                    formula=formula,
                    projection_mode=gad_params.get("projection_mode", "eckart_full"),
                    purify_hessian=gad_params.get("purify_hessian", False),
                    frame_tracking=gad_params.get("frame_tracking", False),
                    log_spectrum_k=gad_params.get("log_spectrum_k", 10),
                    tr_filter_eig=gad_params.get("tr_filter_eig", False),
                    step_mode=gad_params.get("step_mode", "first_order"),
                    step_filter_threshold=gad_params.get("step_filter_threshold", 8e-3),
                    anti_overshoot=gad_params.get("anti_overshoot", False),
                    cleanup_steps=gad_params.get("cleanup_steps", 0),
                )
            elif gad_variant == "nopath":
                gad_raw = _run_gad_nopath(
                    predict_fn,
                    nr_final_coords,
                    atomic_nums,
                    n_steps=gad_n_steps,
                    dt=gad_params["dt"],
                    dt_adaptation=gad_params["dt_adaptation"],
                    dt_min=gad_params["dt_min"],
                    dt_max=gad_params["dt_max"],
                    dt_scale_factor=gad_params.get("dt_scale_factor", 1.0),
                    max_atom_disp=gad_params["max_atom_disp"],
                    ts_eps=gad_params["ts_eps"],
                    stop_at_ts=gad_params.get("stop_at_ts", True),
                    min_interatomic_dist=gad_params["min_interatomic_dist"],
                    tr_threshold=gad_params.get("tr_threshold", 8e-3),
                    track_mode=gad_params.get("track_mode", True),
                    project_gradient_and_v=gad_params.get("project_gradient_and_v", True),
                    hessian_projection=gad_params.get("hessian_projection", "eckart_mw"),
                    log_dir=gad_params.get("log_dir"),
                    sample_id=sample_id,
                    formula=formula,
                )
            else:
                raise ValueError(f"Unknown gad_variant: {gad_variant}")

            gad_converged = bool(gad_raw.get("converged"))

            # Extract cascade + eigenvalue info from GAD result
            cascade_fields = {k: v for k, v in gad_raw.items() if k.startswith("n_neg_at_")}
            neg_info = {
                k: gad_raw.get(k)
                for k in ["lambda_0", "lambda_1", "abs_lambda_0", "lambda_gap_ratio"]
            }

            gad_result = {
                "gad_converged": gad_converged,
                "gad_converged_step": gad_raw.get("converged_step"),
                "gad_final_morse_index": gad_raw.get("final_morse_index"),
                "gad_total_steps": gad_raw.get("total_steps", gad_n_steps),
                "gad_cascade": cascade_fields,
                "gad_neg_eval_info": neg_info,
                "gad_bottom_spectrum": gad_raw.get("bottom_spectrum_at_convergence", []),
                "gad_eig_product": gad_raw.get("eig_product_at_convergence", float("nan")),
                "gad_error": None,
            }
        except Exception as e:
            gad_result = {
                "gad_converged": False,
                "gad_converged_step": None,
                "gad_final_morse_index": None,
                "gad_total_steps": 0,
                "gad_cascade": {},
                "gad_neg_eval_info": {},
                "gad_bottom_spectrum": [],
                "gad_eig_product": float("nan"),
                "gad_error": str(e),
            }
    else:
        gad_result = {
            "gad_converged": False,
            "gad_converged_step": None,
            "gad_final_morse_index": None,
            "gad_total_steps": 0,
            "gad_cascade": {},
            "gad_neg_eval_info": {},
            "gad_bottom_spectrum": [],
            "gad_eig_product": float("nan"),
            "gad_error": "skipped_nr_failed" if not nr_converged else None,
        }

    wall_time = time.time() - t0

    # Overall success: GAD found a TS
    ts_found = bool(gad_result.get("gad_converged"))

    return {
        "ts_found": ts_found,
        "wall_time": wall_time,
        # Phase 1
        **nr_result,
        # Phase 2
        **gad_result,
        # Combined metrics
        "total_steps_both": nr_result.get("nr_total_steps", 0) + gad_result.get("gad_total_steps", 0),
        "gad_was_run": run_gad,
    }


# ---------------------------------------------------------------------------
# Worker interface (plugs into ParallelSCINEProcessor)
# ---------------------------------------------------------------------------

# Module-level state set by main() before processor.start()
_NR_PARAMS: Dict[str, Any] = {}
_GAD_PARAMS: Dict[str, Any] = {}
_NR_N_STEPS: int = 50000
_GAD_N_STEPS: int = 20000
_GAD_VARIANT: str = "path"
_GAD_ON_NR_FAILURE: bool = False


def scine_worker_sample(predict_fn, payload) -> Dict[str, Any]:
    (sample_idx, batch, start_from, noise_seed,
     nr_params, gad_params, nr_n_steps, gad_n_steps,
     gad_variant, gad_on_nr_failure) = payload
    batch = batch.to("cpu")
    atomic_nums = batch.z.detach().to("cpu")
    start_coords = parse_starting_geometry(
        start_from,
        batch,
        noise_seed=noise_seed,
        sample_index=sample_idx,
    ).detach().to("cpu")

    known_ts_coords = getattr(batch, "pos_transition", None)
    if known_ts_coords is None:
        known_ts_coords = getattr(batch, "pos", None)
    if known_ts_coords is not None:
        known_ts_coords = known_ts_coords.detach().to("cpu")

    known_reactant_coords = getattr(batch, "pos_reactant", None)
    if known_reactant_coords is not None:
        known_reactant_coords = known_reactant_coords.detach().to("cpu")

    known_product_coords = None
    has_product = getattr(batch, "has_product", None)
    if has_product is not None and bool(has_product.item()):
        known_product_coords = getattr(batch, "pos_product", None)
        if known_product_coords is not None:
            known_product_coords = known_product_coords.detach().to("cpu")

    formula = getattr(batch, "formula", "")

    result = run_single_sample(
        predict_fn,
        start_coords,
        atomic_nums,
        nr_params,
        gad_params,
        nr_n_steps,
        gad_n_steps,
        gad_variant=gad_variant,
        gad_on_nr_failure=gad_on_nr_failure,
        sample_id=f"sample_{sample_idx:03d}",
        formula=str(formula),
        known_ts_coords=known_ts_coords,
        known_reactant_coords=known_reactant_coords,
        known_product_coords=known_product_coords,
    )
    result["sample_idx"] = sample_idx
    result["formula"] = str(formula)
    return result


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------
def run_batch(
    processor: ParallelSCINEProcessor,
    dataloader,
    max_samples: int,
    start_from: str,
    noise_seed: Optional[int],
    nr_params: Dict[str, Any],
    gad_params: Dict[str, Any],
    nr_n_steps: int,
    gad_n_steps: int,
    gad_variant: str,
    gad_on_nr_failure: bool,
) -> Dict[str, Any]:
    samples = []
    for i, batch in enumerate(dataloader):
        if i >= max_samples:
            break
        payload = (i, batch, start_from, noise_seed,
                   nr_params, gad_params, nr_n_steps, gad_n_steps,
                   gad_variant, gad_on_nr_failure)
        samples.append((i, payload))

    results = run_batch_parallel(samples, processor)

    n_samples = len(results)
    n_ts_found = sum(1 for r in results if r.get("ts_found"))
    n_nr_converged = sum(1 for r in results if r.get("nr_converged"))
    n_gad_run = sum(1 for r in results if r.get("gad_was_run"))
    n_errors = sum(1 for r in results if r.get("nr_error") or r.get("gad_error"))

    nr_steps_list = [r["nr_total_steps"] for r in results if r.get("nr_converged")]
    gad_steps_list = [r["gad_converged_step"] for r in results if r.get("gad_converged_step") is not None]
    wall_times = [r["wall_time"] for r in results]

    # GAD success breakdown: of those where GAD ran, how many found TS?
    n_ts_from_nr_converged = sum(
        1 for r in results if r.get("nr_converged") and r.get("ts_found")
    )
    n_ts_from_nr_failed = sum(
        1 for r in results if not r.get("nr_converged") and r.get("ts_found")
    )

    return {
        "n_samples": n_samples,
        "n_ts_found": n_ts_found,
        "ts_found_rate": n_ts_found / max(n_samples, 1),
        "n_nr_converged": n_nr_converged,
        "nr_convergence_rate": n_nr_converged / max(n_samples, 1),
        "n_gad_run": n_gad_run,
        "n_ts_from_nr_converged": n_ts_from_nr_converged,
        "n_ts_from_nr_failed": n_ts_from_nr_failed,
        "ts_rate_given_nr_converged": n_ts_from_nr_converged / max(n_nr_converged, 1),
        "n_errors": n_errors,
        "mean_nr_steps_when_converged": float(np.mean(nr_steps_list)) if nr_steps_list else float("nan"),
        "mean_gad_steps_when_ts_found": float(np.mean(gad_steps_list)) if gad_steps_list else float("nan"),
        "mean_wall_time": float(np.mean(wall_times)) if wall_times else float("nan"),
        "total_wall_time": float(sum(wall_times)),
        "results": results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    global _NR_PARAMS, _GAD_PARAMS, _NR_N_STEPS, _GAD_N_STEPS, _GAD_VARIANT, _GAD_ON_NR_FAILURE

    parser = argparse.ArgumentParser(
        description="Hybrid NR→GAD: minimize to local minimum, then climb to TS (parallel, SCINE)"
    )

    # ── Common ──────────────────────────────────────────────────────────
    parser.add_argument("--h5-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--scine-functional", type=str, default="DFTB0")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--start-from", type=str, default="midpoint_rt_noise1.0A")
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--threads-per-worker", type=int, default=4)

    # ── Switching control ───────────────────────────────────────────────
    parser.add_argument(
        "--gad-variant", type=str, default="path", choices=["path", "nopath"],
        help="GAD variant: 'path' (trust-region, all features) or 'nopath' (state-based dt only)",
    )
    parser.add_argument(
        "--gad-on-nr-failure", action="store_true", default=False,
        help="Run GAD even if NR does not converge (use final NR geometry)",
    )

    # ── Phase 1: NR parameters ─────────────────────────────────────────
    nr = parser.add_argument_group("Phase 1: Newton-Raphson minimization")
    nr.add_argument("--nr-n-steps", type=int, default=50000)
    nr.add_argument("--nr-max-atom-disp", type=float, default=1.3)
    nr.add_argument("--nr-force-converged", type=float, default=1e-4)
    nr.add_argument("--nr-min-interatomic-dist", type=float, default=0.5)
    nr.add_argument("--nr-threshold", type=float, default=8e-3)
    nr.add_argument("--nr-optimizer-mode", type=str, default="rfo", choices=["", "arc", "rfo", "spdn"])
    nr.add_argument("--nr-polynomial-linesearch", action="store_true", default=False)
    nr.add_argument("--nr-trust-radius-floor", type=float, default=0.01)
    nr.add_argument("--nr-project-gradient-and-v", action="store_true", default=False)
    nr.add_argument("--nr-log-spectrum-k", type=int, default=10)
    nr.add_argument("--nr-relaxed-eval-threshold", type=float, default=0.01)
    nr.add_argument("--nr-accept-relaxed", action="store_true", default=False)
    # v12 kicks
    nr.add_argument("--nr-osc-kick", action="store_true", default=False)
    nr.add_argument("--nr-osc-kick-patience", type=int, default=3)
    nr.add_argument("--nr-osc-kick-cooldown", type=int, default=50)
    nr.add_argument("--nr-blind-kick", action="store_true", default=False)
    nr.add_argument("--nr-blind-kick-overlap-thresh", type=float, default=0.1)
    nr.add_argument("--nr-blind-kick-force-thresh", type=float, default=0.1)
    nr.add_argument("--nr-blind-kick-patience", type=int, default=100)
    # v12b
    nr.add_argument("--nr-adaptive-kick-scale", action="store_true", default=False)
    nr.add_argument("--nr-adaptive-kick-C", type=float, default=0.1)
    nr.add_argument("--nr-blind-kick-probe", action="store_true", default=False)
    nr.add_argument("--nr-late-escape", action="store_true", default=False)
    nr.add_argument("--nr-late-escape-after", type=int, default=15000)
    nr.add_argument("--nr-late-escape-alpha", type=float, default=0.1)
    nr.add_argument("--nr-late-escape-cooldown", type=int, default=500)

    # ── Phase 2: GAD parameters ────────────────────────────────────────
    gad = parser.add_argument_group("Phase 2: GAD saddle search")
    gad.add_argument("--gad-n-steps", type=int, default=20000)
    gad.add_argument("--gad-dt", type=float, default=0.02)
    gad.add_argument("--gad-dt-min", type=float, default=1e-6)
    gad.add_argument("--gad-dt-max", type=float, default=0.08)
    gad.add_argument("--gad-max-atom-disp", type=float, default=0.35)
    gad.add_argument("--gad-min-interatomic-dist", type=float, default=0.5)
    gad.add_argument("--gad-ts-eps", type=float, default=1e-5)
    gad.add_argument("--gad-tr-threshold", type=float, default=8e-3)
    gad.add_argument("--gad-project-gradient-and-v", action="store_true", default=False)
    gad.add_argument("--gad-log-spectrum-k", type=int, default=10)
    # Path-specific
    gad.add_argument("--gad-dt-control", type=str, default="adaptive", choices=["adaptive", "fixed"])
    gad.add_argument("--gad-projection-mode", type=str, default="eckart_full",
                      choices=["eckart_full", "reduced_basis"])
    gad.add_argument("--gad-step-mode", type=str, default="first_order",
                      choices=["first_order", "newton_gad"])
    gad.add_argument("--gad-step-filter-threshold", type=float, default=8e-3)
    gad.add_argument("--gad-anti-overshoot", action="store_true", default=False)
    gad.add_argument("--gad-cleanup-steps", type=int, default=0)
    # Nopath-specific
    gad.add_argument("--gad-dt-adaptation", type=str, default="eigenvalue_clamped")
    gad.add_argument("--gad-dt-scale-factor", type=float, default=1e-1)
    gad.add_argument("--gad-hessian-projection", type=str, default="eckart_mw")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    diag_dir = Path(args.out_dir) / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # Build NR params dict
    _NR_N_STEPS = args.nr_n_steps
    _NR_PARAMS = {
        "method": "newton_raphson",
        "max_atom_disp": args.nr_max_atom_disp,
        "force_converged": args.nr_force_converged,
        "min_interatomic_dist": args.nr_min_interatomic_dist,
        "nr_threshold": args.nr_threshold,
        "optimizer_mode": args.nr_optimizer_mode,
        "polynomial_linesearch": args.nr_polynomial_linesearch,
        "trust_radius_floor": args.nr_trust_radius_floor,
        "project_gradient_and_v": args.nr_project_gradient_and_v,
        "log_spectrum_k": args.nr_log_spectrum_k,
        "relaxed_eval_threshold": args.nr_relaxed_eval_threshold,
        "accept_relaxed": args.nr_accept_relaxed,
        # v12 kicks
        "osc_kick": args.nr_osc_kick,
        "osc_kick_patience": args.nr_osc_kick_patience,
        "osc_kick_cooldown": args.nr_osc_kick_cooldown,
        "blind_kick": args.nr_blind_kick,
        "blind_kick_overlap_thresh": args.nr_blind_kick_overlap_thresh,
        "blind_kick_force_thresh": args.nr_blind_kick_force_thresh,
        "blind_kick_patience": args.nr_blind_kick_patience,
        # v12b
        "adaptive_kick_scale": args.nr_adaptive_kick_scale,
        "adaptive_kick_C": args.nr_adaptive_kick_C,
        "blind_kick_probe": args.nr_blind_kick_probe,
        "late_escape": args.nr_late_escape,
        "late_escape_after": args.nr_late_escape_after,
        "late_escape_alpha": args.nr_late_escape_alpha,
        "late_escape_cooldown": args.nr_late_escape_cooldown,
    }

    # Build GAD params dict
    _GAD_N_STEPS = args.gad_n_steps
    _GAD_VARIANT = args.gad_variant
    _GAD_ON_NR_FAILURE = args.gad_on_nr_failure

    if args.gad_variant == "path":
        _GAD_PARAMS = {
            "dt": args.gad_dt,
            "dt_control": args.gad_dt_control,
            "dt_min": args.gad_dt_min,
            "dt_max": args.gad_dt_max,
            "max_atom_disp": args.gad_max_atom_disp,
            "ts_eps": args.gad_ts_eps,
            "stop_at_ts": True,
            "min_interatomic_dist": args.gad_min_interatomic_dist,
            "tr_threshold": args.gad_tr_threshold,
            "track_mode": True,
            "project_gradient_and_v": args.gad_project_gradient_and_v,
            "log_dir": str(diag_dir),
            "projection_mode": args.gad_projection_mode,
            "log_spectrum_k": args.gad_log_spectrum_k,
            "step_mode": args.gad_step_mode,
            "step_filter_threshold": args.gad_step_filter_threshold,
            "anti_overshoot": args.gad_anti_overshoot,
            "cleanup_steps": args.gad_cleanup_steps,
        }
    else:  # nopath
        _GAD_PARAMS = {
            "dt": args.gad_dt,
            "dt_adaptation": args.gad_dt_adaptation,
            "dt_min": args.gad_dt_min,
            "dt_max": args.gad_dt_max,
            "dt_scale_factor": args.gad_dt_scale_factor,
            "max_atom_disp": args.gad_max_atom_disp,
            "ts_eps": args.gad_ts_eps,
            "stop_at_ts": True,
            "min_interatomic_dist": args.gad_min_interatomic_dist,
            "tr_threshold": args.gad_tr_threshold,
            "track_mode": True,
            "project_gradient_and_v": args.gad_project_gradient_and_v,
            "hessian_projection": args.gad_hessian_projection,
            "log_dir": str(diag_dir),
        }

    # Print configuration
    print("=" * 60)
    print("Hybrid NR→GAD Optimizer")
    print("=" * 60)
    print(f"GAD variant:        {args.gad_variant}")
    print(f"GAD on NR failure:  {args.gad_on_nr_failure}")
    print(f"NR steps (max):     {_NR_N_STEPS}")
    print(f"GAD steps (max):    {_GAD_N_STEPS}")
    print(f"NR optimizer:       {args.nr_optimizer_mode}")
    print(f"NR PLS:             {args.nr_polynomial_linesearch}")
    print(f"NR kicks:           osc={args.nr_osc_kick} blind={args.nr_blind_kick} "
          f"probe={args.nr_blind_kick_probe} late={args.nr_late_escape}")
    print(f"Start from:         {args.start_from}")
    print(f"Max samples:        {args.max_samples}")
    print("=" * 60)

    processor = ParallelSCINEProcessor(
        functional=args.scine_functional,
        threads_per_worker=args.threads_per_worker,
        n_workers=args.n_workers,
        worker_fn=scine_worker_sample,
    )
    processor.start()

    try:
        dataloader = create_dataloader(args.h5_path, args.split, args.max_samples)
        metrics = run_batch(
            processor,
            dataloader,
            args.max_samples,
            args.start_from,
            args.noise_seed,
            nr_params=_NR_PARAMS,
            gad_params=_GAD_PARAMS,
            nr_n_steps=_NR_N_STEPS,
            gad_n_steps=_GAD_N_STEPS,
            gad_variant=_GAD_VARIANT,
            gad_on_nr_failure=_GAD_ON_NR_FAILURE,
        )

        # Save results
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        results_path = Path(args.out_dir) / f"nr_gad_hybrid_{args.gad_variant}_{job_id}_results.json"
        with open(results_path, "w") as f:
            json.dump(
                {
                    "job_id": job_id,
                    "gad_variant": args.gad_variant,
                    "nr_params": _NR_PARAMS,
                    "gad_params": _GAD_PARAMS,
                    "config": {
                        "nr_n_steps": _NR_N_STEPS,
                        "gad_n_steps": _GAD_N_STEPS,
                        "gad_on_nr_failure": args.gad_on_nr_failure,
                        "max_samples": args.max_samples,
                        "start_from": args.start_from,
                        "noise_seed": args.noise_seed,
                        "scine_functional": args.scine_functional,
                        "n_workers": args.n_workers,
                        "threads_per_worker": args.threads_per_worker,
                        "split": args.split,
                    },
                    "metrics": metrics,
                },
                f,
                indent=2,
                default=str,
            )
        print(f"\nResults saved to: {results_path}")

        # Summary
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"  Samples:              {metrics['n_samples']}")
        print(f"  NR converged:         {metrics['n_nr_converged']} "
              f"({metrics['nr_convergence_rate']:.1%})")
        print(f"  GAD ran:              {metrics['n_gad_run']}")
        print(f"  TS found (overall):   {metrics['n_ts_found']} "
              f"({metrics['ts_found_rate']:.1%})")
        print(f"  TS from NR-converged: {metrics['n_ts_from_nr_converged']} "
              f"({metrics['ts_rate_given_nr_converged']:.1%} of NR-converged)")
        print(f"  TS from NR-failed:    {metrics['n_ts_from_nr_failed']}")
        print(f"  Errors:               {metrics['n_errors']}")
        print(f"  Mean NR steps:        {metrics['mean_nr_steps_when_converged']:.0f}")
        print(f"  Mean GAD steps to TS: {metrics['mean_gad_steps_when_ts_found']:.0f}")
        print(f"  Mean wall time:       {metrics['mean_wall_time']:.1f}s")
        print(f"  Total wall time:      {metrics['total_wall_time']:.0f}s")
        print("=" * 60)

    finally:
        processor.close()


if __name__ == "__main__":
    main()
