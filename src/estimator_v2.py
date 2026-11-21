"""v2 Estimator: Drift-Calibrated ANM-Seeded Dipole-Free Two-Way Variable Projection (DC-ANM-VarPro).

Architectural revision addressing two physical failure modes identified on development data:
  1. Quadratic cross-term hallucination (phantom early path with zero physical two-way self-term support).
  2. Dipole collapse of unconstrained closely spaced atoms under slight RF front-end frequency tilt.

This module is strictly separate from the frozen v1 estimator (`src/estimator.py`).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

from .estimator import (
    extract_peeling_seeds,
    stage1_step_drift_and_aggregate,
    stage2_mmv_anm_admm,
)
from .model import C_LIGHT, DF_HZ, N_GRID


def _varpro_solve_beta(
    taus_2way: np.ndarray,
    k_obs: np.ndarray,
    y_obs: np.ndarray,
    w_obs: np.ndarray,
    reg: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Closed-form weighted ridge LS for complex two-way amplitudes given continuous delays."""
    e_mat = np.exp(-1j * 2.0 * np.pi * DF_HZ * np.outer(k_obs, taus_2way))
    m_cnt = len(taus_2way)
    n_paths = y_obs.shape[1]
    betas = np.zeros((m_cnt, n_paths), dtype=np.complex128)
    resid = np.empty_like(y_obs)
    rss = 0.0
    eye_m = reg * np.eye(m_cnt, dtype=np.complex128)
    for p in range(n_paths):
        wp = w_obs[:, p]
        ew = e_mat * wp[:, None]
        w_sum = max(float(np.sum(wp)), 1e-12)
        gram = e_mat.conj().T @ ew + eye_m * w_sum
        rhs = ew.conj().T @ y_obs[:, p]
        bp = np.linalg.solve(gram, rhs)
        betas[:, p] = bp
        rp = y_obs[:, p] - e_mat @ bp
        resid[:, p] = rp
        rss += float(np.sum(wp * (np.abs(rp) ** 2)))
    return betas, resid, rss


def _refine_gap_constrained_varpro(
    init_taus_2way: np.ndarray,
    k_obs: np.ndarray,
    y_obs: np.ndarray,
    w_obs: np.ndarray,
    t2_min: float,
    t2_max: float,
    min_sep_2way: float = 10.0e-9,
    reg: float = 1e-2,
    max_nfev: int = 22,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """Refine sorted 2-way delays parameterized as (u_0, du_1, ..., du_{M-1}) with du_j >= du_min."""
    t_sorted = np.sort(np.clip(init_taus_2way, t2_min, t2_max))
    m_cnt = len(t_sorted)
    for i in range(1, m_cnt):
        if t_sorted[i] < t_sorted[i - 1] + min_sep_2way:
            t_sorted[i] = t_sorted[i - 1] + min_sep_2way
    if t_sorted[-1] > t2_max:
        shift = t_sorted[-1] - t2_max
        t_sorted = np.maximum(t2_min, t_sorted - shift)

    u_min = 2.0 * np.pi * DF_HZ * t2_min
    u_max = 2.0 * np.pi * DF_HZ * t2_max
    du_min = 2.0 * np.pi * DF_HZ * min_sep_2way

    if m_cnt == 1:
        x0 = np.array([np.clip(2.0 * np.pi * DF_HZ * t_sorted[0], u_min, u_max)])
        lb = np.array([u_min])
        ub = np.array([u_max])
    else:
        u_init = 2.0 * np.pi * DF_HZ * t_sorted
        gaps = np.maximum(np.diff(u_init), du_min)
        u0_max = max(u_min, u_max - float(np.sum(gaps)))
        x0 = np.concatenate([[np.clip(u_init[0], u_min, u0_max)], gaps])
        lb = np.concatenate([[u_min], np.full(m_cnt - 1, du_min)])
        ub = np.concatenate([[u_max], np.full(m_cnt - 1, max(du_min + 1e-6, u_max - u_min))])

    n_paths = y_obs.shape[1]
    w_sqrt = np.sqrt(w_obs)
    eye_m = reg * np.eye(m_cnt, dtype=np.complex128)
    w_sums = [max(float(np.sum(w_obs[:, p])), 1e-12) for p in range(n_paths)]

    def _x_to_u(x: np.ndarray) -> np.ndarray:
        return np.cumsum(x) if m_cnt > 1 else x

    def fun(x: np.ndarray) -> np.ndarray:
        u = _x_to_u(x)
        e_mat = np.exp(-1j * np.outer(k_obs, u))
        out_r = []
        for p in range(n_paths):
            wp = w_obs[:, p]
            ew = e_mat * wp[:, None]
            gram = e_mat.conj().T @ ew + eye_m * w_sums[p]
            bp = np.linalg.solve(gram, ew.conj().T @ y_obs[:, p])
            rp = w_sqrt[:, p] * (y_obs[:, p] - e_mat @ bp)
            out_r.append(np.real(rp))
            out_r.append(np.imag(rp))
        if m_cnt > 1:
            out_r.append(np.array([10.0 * max(0.0, float(u[-1] - u_max))]))
        return np.concatenate(out_r)

    sol = least_squares(
        fun, x0, bounds=(lb, ub), method="trf",
        ftol=1e-6, xtol=1e-6, gtol=1e-6, max_nfev=max_nfev,
    )
    t_opt = np.clip(_x_to_u(sol.x) / (2.0 * np.pi * DF_HZ), t2_min, t2_max)
    betas, resid, rss = _varpro_solve_beta(t_opt, k_obs, y_obs, w_obs, reg=reg)
    return t_opt, betas, resid, rss, bool(sol.success)


def estimate_range_dc_anm_varpro(
    step_idx: np.ndarray,
    grid_k: np.ndarray,
    ant_idx: np.ndarray,
    y: np.ndarray,
    q_i: np.ndarray,
    q_r: np.ndarray,
    n_paths: int = 1,
    d_cal_m: float = 0.0,
    d_max_m: float = 45.0,
    m_max: int = 7,
    alpha_fap: float = 0.35,
    min_gain_rel: float = 0.02,
    min_sep_2way: float = 10.0e-9,
    reg_varpro: float = 1e-2,
    enable_stage1_drift: bool = True,
    subevent_idx: np.ndarray | None = None,
) -> dict:
    """Estimate first-path range via Stage-1 Drift + ANM-Seeded Dipole-Free Two-Way VarPro."""
    st1 = stage1_step_drift_and_aggregate(
        step_idx=step_idx, grid_k=grid_k, ant_idx=ant_idx, y=y, q_i=q_i, q_r=q_r,
        n_paths=n_paths, enable_drift=enable_stage1_drift, subevent_idx=subevent_idx,
    )
    obs_mask = st1["obs_mask"]
    n_unique_tones = int(np.sum(np.any(obs_mask, axis=1)))
    if (not st1.get("valid_input", True)) or n_unique_tones < 2:
        in_stat = st1.get("input_status", "insufficient_tones") if not st1.get("valid_input", True) else "insufficient_tones"
        return {
            "valid_estimate": False,
            "input_status": in_stat,
            "n_rows_input": st1.get("n_rows_input", 0),
            "n_rows_admissible": st1.get("n_rows_admissible", 0),
            "n_rows_discarded": st1.get("n_rows_discarded", 0),
            "n_unique_usable_tones": int(st1.get("n_unique_usable_tones", n_unique_tones)),
            "d_hat_m": float("nan"),
            "d_raw_m": float("nan"),
            "M_atoms": 0,
            "varpro_success": False,
            "w_step_hat": st1["w_step_hat"],
        }

    y_full = st1["y_full"]
    w_full = st1["w_obs"]
    k_idx = np.flatnonzero(np.any(obs_mask, axis=1))
    k_gaps = np.diff(k_idx)
    has_big_gap = bool(len(k_gaps) > 0 and int(np.max(k_gaps)) > 5)

    t2_min = 2.0 * max(0.35, d_cal_m + 0.10) / C_LIGHT
    t2_max = 2.0 * d_max_m / C_LIGHT

    # Stage 2: ANM completion strictly when contiguous Wi-Fi/ISM holes (>5 MHz) exist to kill grating sidelobes
    anm_seeds_2way: list[float] = []
    admm_info = {"admm_used": False, "admm_converged": None, "admm_primal_rel": None, "n_admm_iter": 0}
    if has_big_gap:
        st2 = stage2_mmv_anm_admm(y_full, w_full, st1["sigma2_step_hat"])
        admm_info = {
            "admm_used": True,
            "admm_converged": bool(st2["admm_converged"]),
            "admm_primal_rel": float(st2["admm_primal_rel"]),
            "n_admm_iter": int(st2["n_iter"]),
        }
        seeds_info = extract_peeling_seeds(
            x_hat=st2["x_hat"], t_hat=st2["T_hat"], y_full=y_full, obs_mask=obs_mask,
            tau_max_1way=0.5 * t2_max, use_anm=True, tau_min_1way=0.5 * t2_min,
        )
        tt = seeds_info["tilde_taus"]
        pp = seeds_info["pows"]
        for idx in np.argsort(pp)[::-1][:10]:
            tv = float(tt[idx])
            if t2_min <= tv <= t2_max:
                anm_seeds_2way.append(tv)

    y_obs = y_full[k_idx, :]
    w_obs = w_full[k_idx, :]
    mean_w = max(float(np.mean(w_obs[w_obs > 0])), 1e-12)
    w_obs = w_obs / mean_w
    scale = max(math.sqrt(float(np.sum(w_obs * (np.abs(y_obs) ** 2)) / max(float(np.sum(w_obs)), 1.0))), 1e-12)
    y_obs = y_obs / scale
    e_total = float(np.sum(w_obs * (np.abs(y_obs) ** 2)))
    k_float = k_idx.astype(np.float64)

    n_fft = 2048
    min_bin = max(1, int(math.floor(t2_min * DF_HZ * n_fft)))
    max_bin = min(n_fft - 2, int(math.ceil(t2_max * DF_HZ * n_fft)))
    excl_bins = max(2, int(math.ceil(min_sep_2way * DF_HZ * n_fft)))

    taus_cur = np.array([], dtype=np.float64)
    betas_cur = np.zeros((0, n_paths), dtype=np.complex128)
    resid = y_obs.copy()
    rss_prev = e_total
    last_ok = True

    for step_m in range(m_max):
        r_grid = np.zeros((N_GRID, n_paths), dtype=np.complex128)
        r_grid[k_idx, :] = w_obs * resid
        mf = np.sum(np.abs(np.fft.ifft(r_grid, n=n_fft, axis=0)) ** 2, axis=1)
        mf_search = mf[min_bin : max_bin + 1].copy()
        for t_ex in taus_cur:
            ex_bin = int(round(t_ex * DF_HZ * n_fft))
            mf_search[max(0, ex_bin - min_bin - excl_bins) : min(len(mf_search), ex_bin - min_bin + excl_bins + 1)] = -1.0

        cands: list[float] = []
        if np.max(mf_search) > 0:
            pk_bin = min_bin + int(np.argmax(mf_search))
            cands.append(float(pk_bin / (n_fft * DF_HZ)))
        for ts in anm_seeds_2way:
            if all(abs(ts - tx) >= min_sep_2way for tx in taus_cur) and all(abs(ts - c) >= 4.0e-9 for c in cands):
                cands.append(ts)
                if len(cands) >= (4 if step_m < 3 else 2):
                    break

        if not cands:
            break

        best_cand = None
        for tc in cands:
            t_opt, b_opt, r_opt, rss_opt, ok_opt = _refine_gap_constrained_varpro(
                np.append(taus_cur, tc), k_float, y_obs, w_obs, t2_min, t2_max,
                min_sep_2way=min_sep_2way, reg=reg_varpro, max_nfev=22,
            )
            if best_cand is None or rss_opt < best_cand[3]:
                best_cand = (t_opt, b_opt, r_opt, rss_opt, ok_opt)
        assert best_cand is not None
        t_opt, b_opt, r_opt, rss_opt, ok_opt = best_cand
        gain = (rss_prev - rss_opt) / max(e_total, 1e-12)
        if step_m >= 1 and gain < min_gain_rel:
            break
        taus_cur, betas_cur, resid, rss_prev, last_ok = t_opt, b_opt, r_opt, rss_opt, ok_opt
        if rss_prev / max(e_total, 1e-12) < 0.02:
            break

    amps = np.linalg.norm(betas_cur, axis=1)
    non_pin = np.flatnonzero(taus_cur > t2_min + 0.15 / C_LIGHT)
    if len(non_pin) > 0 and len(non_pin) < len(taus_cur):
        taus_cur = taus_cur[non_pin]
        amps = amps[non_pin]

    a_max = float(np.max(amps))
    sel_idx = int(np.argmax(amps))
    for i in range(len(taus_cur)):
        if amps[i] >= alpha_fap * a_max:
            sel_idx = i
            break

    d_raw_m = 0.5 * C_LIGHT * float(taus_cur[sel_idx])
    return {
        "valid_estimate": True,
        "input_status": "valid",
        "n_rows_input": st1.get("n_rows_input", 0),
        "n_rows_admissible": st1.get("n_rows_admissible", 0),
        "n_rows_discarded": st1.get("n_rows_discarded", 0),
        "n_unique_usable_tones": int(st1.get("n_unique_usable_tones", n_unique_tones)),
        "d_hat_m": max(0.0, d_raw_m - d_cal_m),
        "d_raw_m": d_raw_m,
        "M_atoms": int(len(taus_cur)),
        "varpro_success": bool(last_ok),
        "w_step_hat": float(st1["w_step_hat"]),
        "rss_rel": float(rss_prev / max(e_total, 1e-12)),
        **admm_info,
    }
