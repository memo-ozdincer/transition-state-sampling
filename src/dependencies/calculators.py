from __future__ import annotations

from typing import Any, Dict

import torch

from src.core_algos.types import PredictFn
from .pyg_batch import coords_to_pyg_batch


# ---------------------------------------------------------------------------
# UFF Lennard-Jones parameters: {atomic_number: (sigma_Å, epsilon_kcal/mol)}
# Source: Rappé et al., J. Am. Chem. Soc. 114, 10024 (1992)
# ---------------------------------------------------------------------------
_UFF_LJ_PARAMS: Dict[int, tuple] = {
    1:  (2.571, 0.044),   # H
    6:  (3.431, 0.105),   # C
    7:  (3.260, 0.069),   # N
    8:  (3.118, 0.060),   # O
    9:  (2.662, 0.050),   # F
    16: (3.595, 0.274),   # S
    17: (3.516, 0.227),   # Cl
    35: (3.732, 0.320),   # Br
}

# kcal/mol -> eV
_KCAL_TO_EV = 0.043364104


def _lj_energy_forces_hessian(
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    sigma_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Compute pairwise LJ energy, forces, and analytical Hessian.

    Uses UFF parameters with Lorentz-Berthelot mixing rules.
    All outputs in eV / Å units to match the SCINE/HIP interface.

    Args:
        coords: (N, 3) atom positions in Angstrom.
        atomic_nums: (N,) integer atomic numbers.
        sigma_scale: multiplicative factor on all sigma values. Use 1/3
            to shrink LJ equilibria to match covalent bond lengths.

    Returns:
        dict with "energy" (scalar), "forces" (N, 3), "hessian" (3N, 3N).
    """
    dtype = coords.dtype
    device = coords.device
    N = coords.shape[0]

    # --- Build per-atom sigma, epsilon arrays ---
    sigma_i = torch.zeros(N, dtype=dtype, device=device)
    eps_i = torch.zeros(N, dtype=dtype, device=device)
    for a in range(N):
        z = int(atomic_nums[a].item())
        s, e = _UFF_LJ_PARAMS.get(z, (3.4, 0.1))  # fallback ~ carbon
        sigma_i[a] = s * sigma_scale
        eps_i[a] = e * _KCAL_TO_EV  # convert to eV

    # Lorentz-Berthelot mixing: sigma_ij = (s_i + s_j)/2, eps_ij = sqrt(e_i * e_j)
    sigma_ij = 0.5 * (sigma_i.unsqueeze(1) + sigma_i.unsqueeze(0))  # (N, N)
    eps_ij = torch.sqrt(eps_i.unsqueeze(1) * eps_i.unsqueeze(0))     # (N, N)

    # --- Pairwise displacements ---
    # r_vec[i,j] = coords[j] - coords[i]  (so force on i from j is along +r_vec[i,j])
    r_vec = coords.unsqueeze(0) - coords.unsqueeze(1)  # (N, N, 3)
    r2 = (r_vec * r_vec).sum(dim=-1)                    # (N, N)

    # Mask diagonal (self-interaction)
    mask = ~torch.eye(N, dtype=torch.bool, device=device)
    r2_safe = r2.clamp(min=1e-12)
    r = torch.sqrt(r2_safe)  # (N, N)

    # --- LJ scalars ---
    # V_ij = 4 * eps * [(sigma/r)^12 - (sigma/r)^6]
    sr = sigma_ij / r  # (N, N)
    sr6 = sr.pow(6)
    sr12 = sr6 * sr6

    pair_energy = 4.0 * eps_ij * (sr12 - sr6) * mask.to(dtype)  # (N, N)
    energy = 0.5 * pair_energy.sum()  # each pair counted twice

    # --- First derivative of V w.r.t. r (scalar) ---
    # dV/dr = 4*eps * [-12*sigma^12/r^13 + 6*sigma^6/r^7]
    #       = (4*eps/r) * [-12*(sigma/r)^12 + 6*(sigma/r)^6]
    dVdr = (4.0 * eps_ij / r) * (-12.0 * sr12 + 6.0 * sr6) * mask.to(dtype)  # (N, N)

    # --- Forces ---
    # F_i = -dE/dx_i = -sum_j dV_ij/dr_ij * r_hat_ij (with r_ij = x_j - x_i)
    # But dE/dx_i = sum_j dV_ij/dr_ij * d(r_ij)/dx_i = sum_j dV_ij/dr_ij * (-(x_j-x_i)/r_ij)
    # So F_i = sum_j dV_ij/dr_ij * (x_j - x_i) / r_ij = sum_j (dVdr/r) * r_vec[i,j]
    dVdr_over_r = (dVdr / r).unsqueeze(-1)  # (N, N, 1)
    forces = (dVdr_over_r * r_vec).sum(dim=1)  # (N, 3)
    # Note: pair_energy is counted with 0.5, but forces use full sum over j for each i,
    # and dVdr already corresponds to one pair; the factor works out because
    # F_i = -d(0.5 * sum_{i,j} V_ij)/dx_i = -sum_{j!=i} dV_ij/dr * dr/dx_i
    # which is exactly what we compute above.

    if N < 2:
        return {
            "energy": energy,
            "forces": forces,
            "hessian": torch.zeros(3 * N, 3 * N, dtype=dtype, device=device),
        }

    # --- Analytical Hessian (3N x 3N) ---
    # d²V/dr² = 4*eps * [12*13*sigma^12/r^14 - 6*7*sigma^6/r^8]
    #         = (4*eps/r²) * [156*(sigma/r)^12 - 42*(sigma/r)^6]
    d2Vdr2 = (4.0 * eps_ij / r2_safe) * (156.0 * sr12 - 42.0 * sr6) * mask.to(dtype)

    # Unit vectors: r_hat[i,j] = r_vec[i,j] / r[i,j]
    r_hat = r_vec / r.unsqueeze(-1).clamp(min=1e-12)  # (N, N, 3)

    # Off-diagonal 3x3 blocks H_{ij} (i != j):
    # H_{ij,ab} = d²E/(dx_{ia} dx_{jb})
    #           = [d²V/dr² - (1/r)*dV/dr] * r_hat_a * r_hat_b + (1/r)*dV/dr * delta_ab
    # But since r_ij = |x_j - x_i| and dr/dx_{ia} = -(x_ja - x_ia)/r = -r_hat_a,
    # dr/dx_{jb} = +r_hat_b:
    # H_{ij,ab} = (d²V/dr² - dV/(r*dr)) * r_hat_a * r_hat_b + dV/(r*dr) * delta_ab
    #
    # For the off-diagonal block d²(0.5*sum V)/dx_i dx_j with i!=j:
    # = (d2Vdr2 - dVdr/r) * rhat_a * rhat_b + (dVdr/r) * delta_ab
    # (sign: dr/dx_{ia} = -(rhat_a), dr/dx_{jb} = +(rhat_b), so the product is -rhat_a*rhat_b,
    # but the double negative from the chain rule gives positive.)

    A_ij = (d2Vdr2 - dVdr / r)  # (N, N)  coefficient of outer product
    B_ij = dVdr / r              # (N, N)  coefficient of identity

    # Build (N, N, 3, 3) block Hessian
    # outer product: rhat_a * rhat_b -> (N, N, 3, 3)
    rhat_outer = torch.einsum("ija,ijb->ijab", r_hat, r_hat)  # (N, N, 3, 3)
    I3 = torch.eye(3, dtype=dtype, device=device)

    H_blocks = A_ij.unsqueeze(-1).unsqueeze(-1) * rhat_outer + \
               B_ij.unsqueeze(-1).unsqueeze(-1) * I3.unsqueeze(0).unsqueeze(0)
    # Zero out diagonal blocks (i==i) — will be filled by negative row sum
    diag_mask = torch.eye(N, dtype=torch.bool, device=device)
    H_blocks[diag_mask] = 0.0

    # Diagonal blocks: H_{ii} = -sum_{j!=i} H_{ij}
    H_diag = -H_blocks.sum(dim=1)  # (N, 3, 3)
    # Write diagonal blocks back
    for i in range(N):
        H_blocks[i, i] = H_diag[i]

    # Reshape to (3N, 3N)
    hessian = H_blocks.permute(0, 2, 1, 3).reshape(3 * N, 3 * N)

    return {
        "energy": energy,
        "forces": forces,
        "hessian": hessian,
    }


def make_lj_predict_fn(
    sigma_scale: float = 1.0 / 3.0,
) -> PredictFn:
    """Create a PredictFn backed by pairwise Lennard-Jones with UFF parameters.

    Args:
        sigma_scale: factor applied to all UFF sigma values. Default 1/3
            shrinks LJ equilibria to match covalent bond lengths (~1-1.5 Å),
            so Transition1x geometries sit in valid LJ wells.

    Returns:
        A predict_fn(coords, atomic_nums, ...) -> {"energy", "forces", "hessian"}
        with the same interface as make_scine_predict_fn / make_hip_predict_fn.
    """

    def _predict(
        coords: torch.Tensor,
        atomic_nums: torch.Tensor,
        *,
        do_hessian: bool = True,
        require_grad: bool = False,
    ) -> Dict[str, Any]:
        if require_grad:
            raise NotImplementedError(
                "LJ backend uses analytical derivatives; use require_grad=False"
            )

        coords_2d = coords.detach().reshape(-1, 3)
        result = _lj_energy_forces_hessian(coords_2d, atomic_nums.detach(), sigma_scale)

        if not do_hessian:
            result.pop("hessian", None)

        return result

    return _predict


def make_hip_predict_fn(calculator) -> PredictFn:
    """Adapter for HIP EquiformerTorchCalculator.

    - `require_grad=False`: uses `calculator.predict(...)` (fast, no autograd).
    - `require_grad=True`: calls `calculator.potential.forward(...)` so downstream
      code can differentiate w.r.t. `coords`.

    This matches patterns already used in your repo.
    """

    model = calculator.potential

    def _predict(
        coords: torch.Tensor,
        atomic_nums: torch.Tensor,
        *,
        do_hessian: bool = True,
        require_grad: bool = False,
    ) -> Dict[str, Any]:
        device = coords.device
        batch = coords_to_pyg_batch(coords, atomic_nums, device=device)

        if require_grad:
            if not do_hessian:
                raise ValueError("HIP differentiable path expects do_hessian=True")
            with torch.enable_grad():
                # Equiformer returns (energy, forces, out) in some checkpoints;
                # we standardize to a dict.
                _, _, out = model.forward(batch, otf_graph=True)
                # Common keys used across your scripts
                energy = out.get("energy")
                forces = out.get("forces")
                hessian = out.get("hessian")
                if energy is None and "energy" in out:
                    energy = out["energy"]
                return {"energy": energy, "forces": forces, "hessian": hessian}

        # Non-differentiable fast path
        with torch.no_grad():
            return calculator.predict(batch, do_hessian=do_hessian)

    return _predict


def make_scine_predict_fn(scine_calculator) -> PredictFn:
    """Adapter for `ScineSparrowCalculator`.

    SCINE is CPU-only and not differentiable w.r.t coords via autograd.

    Note: The returned dict includes a special "_scine_calculator" key
    that allows downstream code to access get_last_elements() for
    SCINE-specific mass-weighting.
    """

    def _predict(
        coords: torch.Tensor,
        atomic_nums: torch.Tensor,
        *,
        do_hessian: bool = True,
        require_grad: bool = False,
    ) -> Dict[str, Any]:
        if require_grad:
            raise NotImplementedError(
                "SCINE backend is not autograd-differentiable; use require_grad=False"
            )

        batch = coords_to_pyg_batch(coords.detach().cpu(), atomic_nums.detach().cpu(), device=torch.device("cpu"))
        with torch.no_grad():
            result = scine_calculator.predict(batch, do_hessian=do_hessian)

        # Attach calculator reference for SCINE-specific mass-weighting
        result["_scine_calculator"] = scine_calculator

        return result

    return _predict
