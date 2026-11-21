"""O v3: v2.1 fitting pipeline (byte-identical logic up to atom selection) returning the fitted state."""
from __future__ import annotations
import math
import numpy as np
from src.estimator import extract_peeling_seeds, stage1_step_drift_and_aggregate, stage2_mmv_anm_admm
from src.estimator_v2_1 import _refine_gap_constrained_varpro_v2_1, _varpro_solve_beta
from src.model import C_LIGHT, DF_HZ, N_GRID

def fit_v21_state(
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

    return {"valid_estimate": True, "taus": taus_cur.copy(), "betas": betas_cur.copy(), "k_obs": k_float, "y_obs": y_obs,
            "w_obs": w_obs, "resid": resid, "scale": scale, "mean_w": mean_w, "sigma2_raw": float(st1["sigma2_step_hat"]),
            "t2_min": t2_min, "t2_max": t2_max, "e_total": e_total, "rss": rss_prev, "anm_seeds": list(anm_seeds_2way),
            "has_big_gap": has_big_gap, "min_sep_2way": min_sep_2way, "reg_varpro": reg_varpro}
