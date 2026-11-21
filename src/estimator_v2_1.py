"""v2.1 Estimator: Drift-Calibrated ANM-Seeded Exact-Gap Two-Way Variable Projection.

Repairs Finding 1 and cheap-to-check items from C's audit of v2 while keeping frozen v1
(`src/estimator.py`) and original v2 (`src/estimator_v2.py`) byte-for-byte untouched:
  1. Exact radial simplex retraction over nonnegative excess slacks x in [0, S]^M
     guarantees t2_min <= tau_1, tau_m - tau_{m-1} >= min_sep_2way, and tau_M <= t2_max
     at every iterate and final output by algebraic identity, with zero post-hoc clipping
     and zero interior coordinate coupling (g(x) = x whenever sum(x) <= S).
  2. Explicit composed ridge-eliminated data-fit outer objective: closed-form ridge
     amplitudes b_p*(tau) are substituted into the weighted data-fit residual
     sum_p ||D_p^{1/2} (y_p - E(tau) b_p*(tau))||_2^2.
  3. Explicit empty/infeasible range-domain guard returns structured status without raising.
  4. Full fitted two-way delays, amplitudes, minimum pairwise gap, and feasibility check
     are returned for direct independent auditing.
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
    data_rss = 0.0
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
        data_rss += float(np.sum(wp * (np.abs(rp) ** 2)))
    return betas, resid, data_rss


def _project_feasible_u(
    u_raw: np.ndarray,
    u_min: float,
    u_max: float,
    du_min: float,
) -> np.ndarray:
    """Exact L2 projection onto {u_min <= u_0, u_m - u_{m-1} >= du_min, u_{M-1} <= u_max}.

    Uses O(M) Pool-Adjacent-Violators (isotonic regression) on shifted coordinates
    w_j = u_j - j * du_min bounded in [u_min, u_max - (M - 1) * du_min].
    """
    u_sorted = np.sort(np.asarray(u_raw, dtype=np.float64))
    m_cnt = len(u_sorted)
    if m_cnt == 0:
        return u_sorted
    slack = (u_max - u_min) - (m_cnt - 1) * du_min
    if slack < -1e-12:
        raise ValueError("Infeasible polytope: domain smaller than (M-1)*du_min.")
    if slack <= 0.0:
        return u_min + np.arange(m_cnt, dtype=np.float64) * du_min

    offsets = np.arange(m_cnt, dtype=np.float64) * du_min
    w = u_sorted - offsets
    blocks: list[list[float]] = []  # [sum_val, count]
    for val in w:
        blocks.append([float(val), 1.0])
        while len(blocks) >= 2 and (blocks[-2][0] / blocks[-2][1]) > (blocks[-1][0] / blocks[-1][1]):
            b2 = blocks.pop()
            blocks[-1][0] += b2[0]
            blocks[-1][1] += b2[1]
    w_iso = np.empty(m_cnt, dtype=np.float64)
    idx = 0
    for s_val, cnt in blocks:
        n_c = int(round(cnt))
        w_iso[idx : idx + n_c] = s_val / cnt
        idx += n_c
    w_proj = np.clip(w_iso, u_min, u_min + slack)
    return w_proj + offsets


def _x_to_u_simplex(x: np.ndarray, u_min: float, u_max: float, du_min: float) -> np.ndarray:
    """Exact radial simplex retraction from nonnegative slacks x in [0, S]^M to feasible u.

    Let S = max(0, (u_max - u_min) - (M - 1) * du_min).
    Define g(x) = x if sum(x) <= S else (S / sum(x)) * x, where x >= 0.
    Then u_0 = u_min + g_0 and u_m = u_{m-1} + du_min + g_m satisfy:
      u_0 >= u_min,  u_m - u_{m-1} >= du_min,  and  u_{M-1} <= u_max
    by algebraic identity with zero post-hoc clipping.
    """
    m_cnt = len(x)
    slack = max(0.0, (u_max - u_min) - (m_cnt - 1) * du_min)
    x_pos = np.maximum(0.0, np.asarray(x, dtype=np.float64))
    s_sum = float(np.sum(x_pos))
    if s_sum > slack and s_sum > 0.0:
        g = x_pos * (slack / s_sum)
    else:
        g = x_pos
    u = np.empty(m_cnt, dtype=np.float64)
    u[0] = u_min + g[0]
    for i in range(1, m_cnt):
        u[i] = u[i - 1] + du_min + g[i]
    return u


def _u_to_x_simplex(u: np.ndarray, u_min: float, u_max: float, du_min: float) -> np.ndarray:
    """Exact inverse map on the feasible polytope u -> nonnegative excess slacks x in [0, S]^M."""
    m_cnt = len(u)
    slack = max(0.0, (u_max - u_min) - (m_cnt - 1) * du_min)
    x = np.empty(m_cnt, dtype=np.float64)
    x[0] = max(0.0, float(u[0] - u_min))
    for i in range(1, m_cnt):
        x[i] = max(0.0, float((u[i] - u[i - 1]) - du_min))
    return np.clip(x, 0.0, slack)


def _refine_gap_constrained_varpro_v2_1(
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
    """Refine sorted 2-way delays via exact radial-simplex parameterization over [0, S]^M.

    Guarantees t2_min <= t_opt[0], diff(t_opt) >= min_sep_2way, and t_opt[-1] <= t2_max
    at every iterate and final output without post-hoc clipping.
    """
    m_cnt = len(init_taus_2way)
    u_min = 2.0 * np.pi * DF_HZ * t2_min
    u_max = 2.0 * np.pi * DF_HZ * t2_max
    du_min = 2.0 * np.pi * DF_HZ * min_sep_2way
    slack = (u_max - u_min) - (m_cnt - 1) * du_min
    if slack < -1e-12:
        raise ValueError("Requested atom count exceeds domain gap capacity.")

    u_init = _project_feasible_u(
        2.0 * np.pi * DF_HZ * np.asarray(init_taus_2way, dtype=np.float64),
        u_min, u_max, du_min,
    )
    if slack <= 1e-15:
        t_opt = u_init / (2.0 * np.pi * DF_HZ)
        betas, resid, data_rss = _varpro_solve_beta(t_opt, k_obs, y_obs, w_obs, reg=reg)
        return t_opt, betas, resid, data_rss, True

    x0 = _u_to_x_simplex(u_init, u_min, u_max, du_min)
    lb = np.zeros(m_cnt, dtype=np.float64)
    ub = np.full(m_cnt, slack, dtype=np.float64)

    n_paths = y_obs.shape[1]
    w_sqrt = np.sqrt(w_obs)
    eye_m = reg * np.eye(m_cnt, dtype=np.complex128)
    w_sums = [max(float(np.sum(w_obs[:, p])), 1e-12) for p in range(n_paths)]

    def fun(x: np.ndarray) -> np.ndarray:
        u = _x_to_u_simplex(x, u_min, u_max, du_min)
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
        return np.concatenate(out_r)

    sol = least_squares(
        fun, x0, bounds=(lb, ub), method="trf",
        ftol=1e-6, xtol=1e-6, gtol=1e-6, max_nfev=max_nfev,
    )
    u_opt = _x_to_u_simplex(sol.x, u_min, u_max, du_min)
    t_opt = u_opt / (2.0 * np.pi * DF_HZ)

    # Hard algebraic post-check (float64 tolerance 1e-14 s)
    if t_opt[0] < t2_min - 1e-14 or t_opt[-1] > t2_max + 1e-14:
        raise RuntimeError("Simplex retraction upper/lower bound violated.")
    if m_cnt >= 2 and float(np.min(np.diff(t_opt))) < min_sep_2way - 1e-14:
        raise RuntimeError("Simplex retraction minimum gap violated.")

    betas, resid, data_rss = _varpro_solve_beta(t_opt, k_obs, y_obs, w_obs, reg=reg)
    return t_opt, betas, resid, data_rss, bool(sol.success)


def estimate_range_dc_anm_varpro_v2_1(
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
    """Estimate first-path range via Stage-1 Drift + ANM-Seeded Exact-Gap VarPro (v2.1)."""
    st1 = stage1_step_drift_and_aggregate(
        step_idx=step_idx, grid_k=grid_k, ant_idx=ant_idx, y=y, q_i=q_i, q_r=q_r,
        n_paths=n_paths, enable_drift=enable_stage1_drift, subevent_idx=subevent_idx,
    )
    obs_mask = st1["obs_mask"]
    n_unique_tones = int(np.sum(np.any(obs_mask, axis=1)))

    def _invalid_return(status: str) -> dict:
        return {
            "valid_estimate": False,
            "input_status": status,
            "n_rows_input": st1.get("n_rows_input", 0),
            "n_rows_admissible": st1.get("n_rows_admissible", 0),
            "n_rows_discarded": st1.get("n_rows_discarded", 0),
            "n_unique_usable_tones": int(st1.get("n_unique_usable_tones", n_unique_tones)),
            "d_hat_m": float("nan"),
            "d_raw_m": float("nan"),
            "M_atoms": 0,
            "atom_delays_2way_ns": [],
            "atom_amplitudes_l2": [],
            "min_gap_2way_ns": None,
            "bounds_and_gap_feasible": False,
            "varpro_success": False,
            "w_step_hat": st1["w_step_hat"],
            "admm_used": False,
            "admm_converged": None,
            "admm_primal_rel": None,
            "n_admm_iter": 0,
        }

    if (not st1.get("valid_input", True)) or n_unique_tones < 2:
        in_stat = st1.get("input_status", "insufficient_tones") if not st1.get("valid_input", True) else "insufficient_tones"
        return _invalid_return(in_stat)

    if not (math.isfinite(d_cal_m) and math.isfinite(d_max_m) and math.isfinite(min_sep_2way) and min_sep_2way >= 0.0 and reg_varpro >= 0.0):
        return _invalid_return("invalid_range_domain")

    t2_min = 2.0 * max(0.35, d_cal_m + 0.10) / C_LIGHT
    t2_max = 2.0 * d_max_m / C_LIGHT
    n_fft = 2048
    min_bin = max(1, int(math.floor(t2_min * DF_HZ * n_fft)))
    max_bin = min(n_fft - 2, int(math.ceil(t2_max * DF_HZ * n_fft)))
    if not (t2_min < t2_max and min_bin <= max_bin):
        return _invalid_return("invalid_range_domain")

    y_full = st1["y_full"]
    w_full = st1["w_obs"]
    k_idx = np.flatnonzero(np.any(obs_mask, axis=1))
    k_gaps = np.diff(k_idx)
    has_big_gap = bool(len(k_gaps) > 0 and int(np.max(k_gaps)) > 5)

    # Stage 2: ANM candidate seeds only when contiguous Wi-Fi/ISM gaps exceed 5 MHz
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

    excl_bins = max(2, int(math.ceil(min_sep_2way * DF_HZ * n_fft)))
    if min_sep_2way > 0.0:
        m_cap = 1 + int(math.floor((t2_max - t2_min) / min_sep_2way))
    else:
        m_cap = m_max
    m_eff_max = min(int(m_max), max(1, m_cap))

    taus_cur = np.array([], dtype=np.float64)
    betas_cur = np.zeros((0, n_paths), dtype=np.complex128)
    resid = y_obs.copy()
    rss_prev = e_total
    last_ok = True

    for step_m in range(m_eff_max):
        r_grid = np.zeros((N_GRID, n_paths), dtype=np.complex128)
        r_grid[k_idx, :] = w_obs * resid
        mf = np.sum(np.abs(np.fft.ifft(r_grid, n=n_fft, axis=0)) ** 2, axis=1)
        mf_search = mf[min_bin : max_bin + 1].copy()
        for t_ex in taus_cur:
            ex_bin = int(round(t_ex * DF_HZ * n_fft))
            mf_search[max(0, ex_bin - min_bin - excl_bins) : min(len(mf_search), ex_bin - min_bin + excl_bins + 1)] = -1.0

        cands: list[float] = []
        if mf_search.size > 0 and float(np.max(mf_search)) > 0.0:
            pk_bin = min_bin + int(np.argmax(mf_search))
            cands.append(float(np.clip(pk_bin / (n_fft * DF_HZ), t2_min, t2_max)))
        for ts in anm_seeds_2way:
            if all(abs(ts - tx) >= min_sep_2way for tx in taus_cur) and all(abs(ts - c) >= 4.0e-9 for c in cands):
                cands.append(ts)
                if len(cands) >= (4 if step_m < 3 else 2):
                    break

        if not cands:
            break

        best_cand = None
        for tc in cands:
            t_opt, b_opt, r_opt, rss_opt, ok_opt = _refine_gap_constrained_varpro_v2_1(
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

    if len(taus_cur) == 0:
        return _invalid_return("no_candidate_atom")

    # Full fitted atom set verification before lower-bound pin pruning
    full_delays_ns = [float(t * 1e9) for t in taus_cur]
    full_amps_l2 = [float(a) for a in np.linalg.norm(betas_cur, axis=1)]
    min_gap_ns = float(np.min(np.diff(taus_cur)) * 1e9) if len(taus_cur) >= 2 else None
    bounds_ok = bool(
        taus_cur[0] >= t2_min - 1e-14
        and taus_cur[-1] <= t2_max + 1e-14
        and (min_gap_ns is None or min_gap_ns >= min_sep_2way * 1e9 - 1e-5)
    )

    amps = np.linalg.norm(betas_cur, axis=1)
    taus_sel = taus_cur
    non_pin = np.flatnonzero(taus_cur > t2_min + 0.15 / C_LIGHT)
    if len(non_pin) > 0 and len(non_pin) < len(taus_cur):
        taus_sel = taus_cur[non_pin]
        amps = amps[non_pin]

    a_max = float(np.max(amps))
    sel_idx = int(np.argmax(amps))
    for i in range(len(taus_sel)):
        if amps[i] >= alpha_fap * a_max:
            sel_idx = i
            break

    d_raw_m = 0.5 * C_LIGHT * float(taus_sel[sel_idx])
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
        "atom_delays_2way_ns": full_delays_ns,
        "atom_amplitudes_l2": full_amps_l2,
        "min_gap_2way_ns": min_gap_ns,
        "bounds_and_gap_feasible": bounds_ok,
        "varpro_success": bool(last_ok),
        "w_step_hat": float(st1["w_step_hat"]),
        "rss_rel": float(rss_prev / max(e_total, 1e-12)),
        **admm_info,
    }
