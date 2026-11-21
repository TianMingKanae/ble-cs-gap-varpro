"""Exact analytical Slepian-Bangs FIM, CRB, and identifiability checks."""

from __future__ import annotations

import numpy as np

from .model import C_LIGHT, DF_HZ


def build_complex_jacobian(
    step_idx: np.ndarray,
    grid_k: np.ndarray,
    ant_idx: np.ndarray,
    taus: np.ndarray,
    alphas: np.ndarray,
    omega_step: float,
    include_omega_step: bool = True,
    include_cfosd: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute noise-free observation vector mu and complex Jacobian d(mu)/d(theta).

    Real parameter order:
        [tau_0..tau_{L-1}, (omega_step if include_omega_step),
         Re(alpha_{0,0}), ..., Re(alpha_{P-1,L-1}),
         Im(alpha_{0,0}), ..., Im(alpha_{P-1,L-1}),
         (tau_cfosd if include_cfosd)]
    """
    taus = np.asarray(taus, dtype=np.float64)
    alphas = np.asarray(alphas, dtype=np.complex128)
    if alphas.ndim == 1:
        alphas = alphas[None, :]
    n_paths, n_mp = alphas.shape
    n_steps = len(step_idx)

    phase_mat = -2.0 * np.pi * DF_HZ * np.outer(grid_k, taus)
    e_mat = np.exp(1j * phase_mat)  # (n_steps, L)
    a_step = alphas[ant_idx, :]  # (n_steps, L)
    h_step = np.sum(a_step * e_mat, axis=1)  # (n_steps,)
    drift = np.exp(1j * omega_step * step_idx)
    mu = drift * (h_step ** 2)

    cols: list[np.ndarray] = []
    # Derivatives wrt tau_l
    coef = 2.0 * drift * h_step * (-1j * 2.0 * np.pi * DF_HZ * grid_k)
    d_taus = coef[:, None] * a_step * e_mat  # (n_steps, L)
    for l_idx in range(n_mp):
        cols.append(d_taus[:, l_idx])

    # Derivative wrt omega_step
    if include_omega_step:
        cols.append(1j * step_idx.astype(np.float64) * mu)

    # Derivatives wrt Re(alpha_{p,l}) and Im(alpha_{p,l})
    re_cols: list[np.ndarray] = []
    im_cols: list[np.ndarray] = []
    base = 2.0 * drift * h_step
    for p_idx in range(n_paths):
        mask_p = (ant_idx == p_idx).astype(np.float64)
        for l_idx in range(n_mp):
            d_re = mask_p * base * e_mat[:, l_idx]
            re_cols.append(d_re)
            im_cols.append(1j * d_re)
    cols.extend(re_cols)
    cols.extend(im_cols)

    if include_cfosd:
        cols.append(np.sum(d_taus, axis=1))

    jacobian = np.column_stack(cols)
    return mu, jacobian


def compute_fim_and_crb(
    step_idx: np.ndarray,
    grid_k: np.ndarray,
    ant_idx: np.ndarray,
    sigma2_step: np.ndarray | float,
    taus: np.ndarray,
    alphas: np.ndarray,
    omega_step: float = 0.0,
    include_omega_step: bool = True,
    include_cfosd: bool = False,
) -> dict:
    """Compute exact Slepian-Bangs FIM J = 2 Re(D^H W D) and range CRB in meters."""
    n_steps = len(step_idx)
    sig2 = np.asarray(sigma2_step, dtype=np.float64)
    if sig2.ndim == 0:
        sig2 = np.full(n_steps, float(sig2), dtype=np.float64)
    inv_sig2 = 1.0 / np.maximum(sig2, 1e-30)

    _, jac = build_complex_jacobian(
        step_idx=step_idx,
        grid_k=grid_k,
        ant_idx=ant_idx,
        taus=taus,
        alphas=alphas,
        omega_step=omega_step,
        include_omega_step=include_omega_step,
        include_cfosd=include_cfosd,
    )

    # Slepian-Bangs FIM: 2 * Re(D^H diag(1/sigma^2) D)
    wd = jac * inv_sig2[:, None]
    fim = 2.0 * np.real(jac.conj().T @ wd)
    fim = 0.5 * (fim + fim.T)

    # Diagonal column equilibration removes 10^16 unit disparity (seconds vs rad)
    diag_scale = np.sqrt(np.maximum(np.diag(fim), 1e-30))
    fim_eq = fim / np.outer(diag_scale, diag_scale)
    fim_eq = 0.5 * (fim_eq + fim_eq.T)

    # Condition number & rank diagnostics on equilibrated correlation FIM
    svals_eq = np.linalg.svd(fim_eq, compute_uv=False)
    s_max = float(svals_eq[0]) if len(svals_eq) else 0.0
    s_min = float(svals_eq[-1]) if len(svals_eq) else 0.0
    cond = (s_max / s_min) if s_min > 1e-15 else float("inf")
    eff_rank = int(np.sum(svals_eq > 1e-10 * s_max))

    crb_tau1_s = float("inf")
    crb_tau1_m = float("inf")
    crb_var_tau1_s2 = float("inf")
    crb_var_d_m2 = float("inf")
    crb_omega_step = float("inf")

    if cond < 1e10 and eff_rank == fim.shape[0]:
        try:
            cov_eq = np.linalg.inv(fim_eq)
            cov = cov_eq / np.outer(diag_scale, diag_scale)
            if cov[0, 0] >= 0:
                crb_var_tau1_s2 = float(cov[0, 0])
                crb_var_d_m2 = float((C_LIGHT ** 2) * crb_var_tau1_s2)
                crb_tau1_s = float(np.sqrt(crb_var_tau1_s2))
                crb_tau1_m = float(C_LIGHT * crb_tau1_s)
            if include_omega_step:
                l_count = len(taus)
                if cov[l_count, l_count] >= 0:
                    crb_omega_step = float(np.sqrt(cov[l_count, l_count]))
        except np.linalg.LinAlgError:
            pass

    svals = svals_eq

    kappa_hop_sd = float("nan")
    kappa_hop_var = float("nan")
    if include_omega_step:
        known_res = compute_fim_and_crb(
            step_idx=step_idx,
            grid_k=grid_k,
            ant_idx=ant_idx,
            sigma2_step=sig2,
            taus=taus,
            alphas=alphas,
            omega_step=omega_step,
            include_omega_step=False,
            include_cfosd=False,
        )
        base_m = known_res["rcrb_d_m"]
        if np.isfinite(crb_tau1_m) and np.isfinite(base_m) and base_m > 0:
            kappa_hop_sd = float(crb_tau1_m / base_m)
            kappa_hop_var = float(kappa_hop_sd ** 2)
        elif np.isfinite(base_m):
            kappa_hop_sd = float("inf")
            kappa_hop_var = float("inf")

    return {
        "fim": fim,
        "svals": svals,
        "cond": cond,
        "rank": eff_rank,
        "dim": fim.shape[0],
        "rcrb_tau1_s": crb_tau1_s,
        "rcrb_d_m": crb_tau1_m,
        "crb_var_tau1_s2": crb_var_tau1_s2,
        "crb_var_d_m2": crb_var_d_m2,
        "crb_tau1_s": crb_tau1_s,
        "crb_tau1_m": crb_tau1_m,
        "crb_omega_step": crb_omega_step,
        "kappa_hop_sd": kappa_hop_sd,
        "kappa_hop_var": kappa_hop_var,
        "kappa_hop": kappa_hop_sd,
    }


def verify_jacobian_fd(
    step_idx: np.ndarray,
    grid_k: np.ndarray,
    ant_idx: np.ndarray,
    taus: np.ndarray,
    alphas: np.ndarray,
    omega_step: float,
) -> dict:
    """Verify analytical Jacobian against truly independent standalone finite differences."""
    taus = np.asarray(taus, dtype=np.float64).copy()
    alphas = np.asarray(alphas, dtype=np.complex128).copy()
    if alphas.ndim == 1:
        alphas = alphas[None, :]
    n_paths, n_mp = alphas.shape

    _, jac_analytic = build_complex_jacobian(
        step_idx, grid_k, ant_idx, taus, alphas, omega_step, True, True
    )
    jac_fd = np.zeros_like(jac_analytic)

    def eval_mu(t: np.ndarray, a: np.ndarray, w: float) -> np.ndarray:
        e_independent = np.exp(-2j * np.pi * DF_HZ * np.outer(grid_k, t))
        h_independent = np.sum(a[ant_idx, :] * e_independent, axis=1)
        return np.exp(1j * w * step_idx) * (h_independent ** 2)

    col = 0
    eps_tau = 1e-12
    for l_idx in range(n_mp):
        tp, tm = taus.copy(), taus.copy()
        tp[l_idx] += eps_tau
        tm[l_idx] -= eps_tau
        jac_fd[:, col] = (eval_mu(tp, alphas, omega_step) - eval_mu(tm, alphas, omega_step)) / (2.0 * eps_tau)
        col += 1

    eps_w = 1e-7
    jac_fd[:, col] = (
        eval_mu(taus, alphas, omega_step + eps_w) - eval_mu(taus, alphas, omega_step - eps_w)
    ) / (2.0 * eps_w)
    col += 1

    eps_a = 1e-7
    for p_idx in range(n_paths):
        for l_idx in range(n_mp):
            ap, am = alphas.copy(), alphas.copy()
            ap[p_idx, l_idx] += eps_a
            am[p_idx, l_idx] -= eps_a
            jac_fd[:, col] = (eval_mu(taus, ap, omega_step) - eval_mu(taus, am, omega_step)) / (2.0 * eps_a)
            col += 1

    for p_idx in range(n_paths):
        for l_idx in range(n_mp):
            ap, am = alphas.copy(), alphas.copy()
            ap[p_idx, l_idx] += 1j * eps_a
            am[p_idx, l_idx] -= 1j * eps_a
            jac_fd[:, col] = (eval_mu(taus, ap, omega_step) - eval_mu(taus, am, omega_step)) / (2.0 * eps_a)
            col += 1

    # tau_cfosd = shift all taus simultaneously
    jac_fd[:, col] = (
        eval_mu(taus + eps_tau, alphas, omega_step) - eval_mu(taus - eps_tau, alphas, omega_step)
    ) / (2.0 * eps_tau)

    rel_errs = []
    for c_idx in range(jac_analytic.shape[1]):
        denom = max(float(np.linalg.norm(jac_fd[:, c_idx])), 1e-12)
        err = float(np.linalg.norm(jac_analytic[:, c_idx] - jac_fd[:, c_idx]) / denom)
        rel_errs.append(err)

    return {
        "max_rel_err": max(rel_errs),
        "per_col_rel_err": rel_errs,
    }
