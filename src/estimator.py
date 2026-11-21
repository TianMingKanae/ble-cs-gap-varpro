"""Proposed 3-stage SDR-ANM-QML estimator for BLE 6.0/6.2 Channel Sounding."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from .model import C_LIGHT, DF_HZ, N_GRID

# Precompute Hermitian Toeplitz index maps & diagonal-averaging indices for N = N_GRID
_IDX_MAT = np.arange(N_GRID)[:, None] - np.arange(N_GRID)[None, :]
_POS_MASK = _IDX_MAT >= 0
_ABS_IDX = np.abs(_IDX_MAT)
_LOWER_R, _LOWER_C = np.tril_indices(N_GRID)
_LOWER_LAGS = _LOWER_R - _LOWER_C
_DIAG_COUNTS = (N_GRID - np.arange(N_GRID)).astype(np.float64)


def _toeplitz_hermitian(u: np.ndarray) -> np.ndarray:
    """Build Hermitian Toeplitz matrix T with T[r, c] = u[r - c] for r >= c."""
    mat = np.where(_POS_MASK, u[_ABS_IDX], np.conj(u[_ABS_IDX]))
    return 0.5 * (mat + mat.conj().T)


def _toeplitz_diag_means(m11: np.ndarray) -> np.ndarray:
    """Vectorized extraction of Hermitian Toeplitz vector u[0..N-1] via np.bincount."""
    vals = m11[_LOWER_R, _LOWER_C]
    u_re = np.bincount(_LOWER_LAGS, weights=np.real(vals), minlength=N_GRID) / _DIAG_COUNTS
    u_im = np.bincount(_LOWER_LAGS, weights=np.imag(vals), minlength=N_GRID) / _DIAG_COUNTS
    u = u_re + 1j * u_im
    u[0] = np.real(u[0])
    return u


def _compute_reliability_weights(q_i: np.ndarray, q_r: np.ndarray) -> np.ndarray:
    """Map BLE CS quality indicators {0,1,2} to reliability weights."""
    q_max = np.maximum(q_i, q_r)
    rho = np.where(q_max == 0, 1.0, np.where(q_max == 1, 0.10, 0.0))
    if np.sum(rho > 0) < max(6, len(q_max) // 12):
        rho = np.where(q_max == 0, 1.0, np.where(q_max == 1, 0.25, 0.05))
    return rho.astype(np.float64)


def _is_integer_valued(arr: np.ndarray) -> bool:
    """Return True iff finite non-bool real array holds exact integer values."""
    if arr.dtype == bool or np.iscomplexobj(arr):
        return False
    if np.issubdtype(arr.dtype, np.integer):
        return True
    if not np.issubdtype(arr.dtype, np.number):
        return False
    return bool(np.all(np.isfinite(arr)) and np.all(arr == np.floor(arr)))


def _filter_and_validate_inputs(
    step_idx: np.ndarray,
    grid_k: np.ndarray,
    ant_idx: np.ndarray,
    y: np.ndarray,
    q_i: np.ndarray,
    q_r: np.ndarray,
    n_paths: int = 1,
    subevent_idx: np.ndarray | None = None,
) -> tuple[bool, str, dict | None]:
    """Validate shapes, finiteness, integer discrete indices, and filter one admissible row set."""
    s_arr = np.asarray(step_idx)
    k_arr = np.asarray(grid_k)
    a_arr = np.asarray(ant_idx)
    y_arr = np.asarray(y)
    qi_arr = np.asarray(q_i)
    qr_arr = np.asarray(q_r)
    arrs = [s_arr, k_arr, a_arr, y_arr, qi_arr, qr_arr]
    sub_arr_in = None if subevent_idx is None else np.asarray(subevent_idx)
    if sub_arr_in is not None:
        arrs.append(sub_arr_in)
    if any(arr.ndim != 1 for arr in arrs) or len(y_arr) == 0 or any(len(arr) != len(y_arr) for arr in arrs):
        return False, "empty_input", None
    if any(not np.all(np.isfinite(arr)) for arr in arrs):
        return False, "non_finite_input", None
    disc_arrs = [s_arr, k_arr, a_arr, qi_arr, qr_arr] + ([sub_arr_in] if sub_arr_in is not None else [])
    if any(not _is_integer_valued(arr) for arr in disc_arrs):
        return False, "non_integer_index", None

    s_i = s_arr.astype(np.int64, copy=False)
    k_i = k_arr.astype(np.int64, copy=False)
    a_i = a_arr.astype(np.int64, copy=False)
    qi_i = qi_arr.astype(np.int64, copy=False)
    qr_i = qr_arr.astype(np.int64, copy=False)
    sub_i = None if sub_arr_in is None else sub_arr_in.astype(np.int64, copy=False)

    adm_mask = (
        (k_i >= 0) & (k_i < N_GRID)
        & (a_i >= 0) & (a_i < n_paths)
        & (qi_i >= 0) & (qi_i <= 2)
        & (qr_i >= 0) & (qr_i <= 2)
    )
    n_in = int(len(y_arr))
    n_adm = int(np.sum(adm_mask))
    if n_adm == 0:
        return False, "no_in_range_tones", {"n_rows_input": n_in, "n_rows_admissible": 0, "n_rows_discarded": n_in}

    y_c = np.asarray(y_arr[adm_mask], dtype=np.complex128)
    k_adm = k_i[adm_mask]
    qi_adm = qi_i[adm_mask]
    qr_adm = qr_i[adm_mask]
    rho = _compute_reliability_weights(qi_adm, qr_adm)
    usable = (rho > 0) & (np.abs(y_c) > 1e-15)
    if (not np.any(usable)) or float(np.sum(np.abs(y_c[usable]) ** 2)) <= 1e-24:
        return False, "zero_signal", {"n_rows_input": n_in, "n_rows_admissible": n_adm, "n_rows_discarded": n_in - n_adm, "n_unique_usable_tones": 0}
    n_unique_usable = int(len(np.unique(k_adm[usable])))
    if n_unique_usable < 2:
        return False, "insufficient_tones", {
            "n_rows_input": n_in,
            "n_rows_admissible": n_adm,
            "n_rows_discarded": n_in - n_adm,
            "n_unique_usable_tones": n_unique_usable,
        }

    return True, "valid", {
        "step_idx": s_i[adm_mask],
        "grid_k": k_adm,
        "ant_idx": a_i[adm_mask],
        "y": y_c,
        "q_i": qi_adm,
        "q_r": qr_adm,
        "subevent_idx": None if sub_i is None else sub_i[adm_mask],
        "n_rows_input": n_in,
        "n_rows_admissible": n_adm,
        "n_rows_discarded": n_in - n_adm,
        "n_unique_usable_tones": n_unique_usable,
    }


def _validate_procedure_inputs(
    step_idx: np.ndarray,
    grid_k: np.ndarray,
    ant_idx: np.ndarray,
    y: np.ndarray,
    q_i: np.ndarray,
    q_r: np.ndarray,
    n_paths: int = 1,
) -> tuple[bool, str]:
    """Validate finiteness, non-emptiness, integer discrete indices, and nonzero usable IQ energy."""
    ok, status, _ = _filter_and_validate_inputs(
        step_idx, grid_k, ant_idx, y, q_i, q_r, n_paths=n_paths
    )
    return ok, status


def stage1_step_drift_and_aggregate(
    step_idx: np.ndarray,
    grid_k: np.ndarray,
    ant_idx: np.ndarray,
    y: np.ndarray,
    q_i: np.ndarray,
    q_r: np.ndarray,
    n_paths: int = 1,
    enable_drift: bool = True,
    subevent_idx: np.ndarray | None = None,
) -> dict:
    """Estimate step-indexed drift w_step via phase-aligned real-part coherence, align subevents, and aggregate."""
    ok_in, in_status, adm = _filter_and_validate_inputs(
        step_idx, grid_k, ant_idx, y, q_i, q_r, n_paths=n_paths, subevent_idx=subevent_idx
    )
    if not ok_in or adm is None or "step_idx" not in adm:
        n_in = len(np.asarray(y)) if np.asarray(y).ndim == 1 else 0
        n_adm = int(adm.get("n_rows_admissible", 0)) if isinstance(adm, dict) else 0
        return {
            "valid_input": False,
            "input_status": in_status,
            "n_rows_input": n_in,
            "n_rows_admissible": n_adm,
            "n_rows_discarded": max(0, n_in - n_adm),
            "n_unique_usable_tones": int(adm.get("n_unique_usable_tones", 0)) if isinstance(adm, dict) else 0,
            "w_step_hat": 0.0,
            "coh_peak": 0.0,
            "n_pairs": 0,
            "lag_gcd": 0,
            "drift_status": in_status,
            "subevent_phase_hats": {0: 0.0},
            "subevent_align_status": "invalid_input",
            "y_full": np.zeros((N_GRID, n_paths), dtype=np.complex128),
            "w_obs": np.zeros((N_GRID, n_paths), dtype=np.float64),
            "obs_mask": np.zeros((N_GRID, n_paths), dtype=bool),
            "sigma2_step_hat": float("nan"),
        }

    step_idx = adm["step_idx"]
    grid_k = adm["grid_k"]
    ant_idx = adm["ant_idx"]
    y = adm["y"]
    q_i = adm["q_i"]
    q_r = adm["q_r"]
    subevent_idx = adm["subevent_idx"]
    n_rows_input = int(adm["n_rows_input"])
    n_rows_admissible = int(adm["n_rows_admissible"])
    n_rows_discarded = int(adm["n_rows_discarded"])
    n_steps = len(step_idx)
    rho = _compute_reliability_weights(q_i, q_r)

    if subevent_idx is None:
        sub_arr = np.zeros(n_steps, dtype=int)
        multi_sub = False
    else:
        sub_arr = np.asarray(subevent_idx, dtype=int)
        multi_sub = n_steps > 0 and bool(np.any(sub_arr != sub_arr[0]))

    if multi_sub:
        unique_subs = np.unique(sub_arr)
        sub_starts: dict[int, int] = {}
        for s_id in unique_subs:
            idxs_s = np.flatnonzero(sub_arr == s_id)
            sub_starts[int(s_id)] = int(step_idx[idxs_s[0]]) if len(idxs_s) else 0
        step_eff = np.asarray([step_idx[i] - sub_starts[int(sub_arr[i])] for i in range(n_steps)], dtype=np.float64)
    else:
        step_eff = step_idx.astype(np.float64)

    w_hat = 0.0
    coh_peak = 0.0
    n_pairs = 0
    lag_gcd = 0
    drift_status = "disabled" if not enable_drift else "no_usable_repeats"

    if enable_drift and n_steps >= 4:
        max_lag = int(np.max(step_idx) - np.min(step_idx))
        if max_lag >= 1:
            lag_phasors = np.zeros(max_lag + 1, dtype=np.complex128)
            lag_weight_sum = 0.0
            by_kps: dict[tuple[int, int, int], list[int]] = defaultdict(list)
            for idx_s in range(n_steps):
                if rho[idx_s] > 0 and abs(y[idx_s]) > 0:
                    by_kps[(int(grid_k[idx_s]), int(ant_idx[idx_s]), int(sub_arr[idx_s]))].append(idx_s)
            for idxs in by_kps.values():
                m_len = len(idxs)
                if m_len < 2:
                    continue
                for a in range(m_len):
                    i1 = idxs[a]
                    for b in range(a + 1, m_len):
                        i2 = idxs[b]
                        dn = int(step_idx[i2] - step_idx[i1])
                        if dn <= 0:
                            continue
                        prod = rho[i1] * rho[i2] * y[i2] * np.conj(y[i1])
                        if abs(prod) <= 0.0:
                            continue
                        lag_phasors[dn] += prod
                        lag_weight_sum += abs(prod)
                        n_pairs += 1

            nz = np.flatnonzero(np.abs(lag_phasors) > 0)
            if len(nz) >= 1 and lag_weight_sum > 0.0:
                lag_gcd = int(math.gcd(*nz.tolist()))
                drift_status = "identifiable_gcd_1" if lag_gcd == 1 else f"aliased_gcd_{lag_gcd}"
                w_bound = min(0.25, math.pi / max(lag_gcd, 1))
                ws_grid = np.linspace(-w_bound, w_bound, 2001)
                c_nz = lag_phasors[nz]
                # Phase-aligned real-part circular coherence (retains zero-lag pair phase)
                scores = np.real(c_nz @ np.exp(-1j * np.outer(nz, ws_grid)))
                best_i = int(np.argmax(scores))
                w0 = float(ws_grid[best_i])
                lo = max(-w_bound, w0 - 0.002)
                hi = min(w_bound, w0 + 0.002)
                opt = minimize_scalar(
                    lambda w: -float(np.real(np.sum(c_nz * np.exp(-1j * nz * w)))),
                    bounds=(lo, hi),
                    method="bounded",
                    options={"xatol": 1e-11},
                )
                w_hat = float(opt.x)
                coh_peak = float(-opt.fun / lag_weight_sum)

    y_cal = y * np.exp(-1j * w_hat * step_eff) if w_hat != 0.0 else y.copy()

    # Estimate and remove inter-subevent phase reset offsets when multiple subevents exist
    sub_phase_hats: dict[int, float] = {int(sub_arr[0]): 0.0} if n_steps > 0 else {0: 0.0}
    align_status = "single_subevent"
    if multi_sub:
        unique_subs = np.unique(sub_arr)
        n_subs = len(unique_subs)
        y_sub_sum = np.zeros((n_subs, N_GRID, n_paths), dtype=np.complex128)
        w_sub_sum = np.zeros((n_subs, N_GRID, n_paths), dtype=np.float64)
        sub_to_row = {int(s_id): idx_m for idx_m, s_id in enumerate(unique_subs)}
        for idx_s in range(n_steps):
            k = int(grid_k[idx_s])
            p = int(ant_idx[idx_s])
            if 0 <= k < N_GRID and 0 <= p < n_paths and rho[idx_s] > 0:
                m_row = sub_to_row[int(sub_arr[idx_s])]
                y_sub_sum[m_row, k, p] += rho[idx_s] * y_cal[idx_s]
                w_sub_sum[m_row, k, p] += rho[idx_s]
        y_sub = np.zeros_like(y_sub_sum)
        nz_sub = w_sub_sum > 0
        y_sub[nz_sub] = y_sub_sum[nz_sub] / w_sub_sum[nz_sub]

        # Dominant two-way delay estimate on reference subevent 0 for disjoint-tone fallback
        cir0 = np.sum(np.abs(np.fft.ifft(y_sub[0], n=512, axis=0)) ** 2, axis=1)
        tau2_dom = float(np.argmax(cir0)) / (512.0 * DF_HZ)
        a0_paths = np.zeros(n_paths, dtype=np.complex128)
        w0_paths = np.sum(w_sub_sum[0], axis=0)
        for p_idx in range(n_paths):
            kp = np.flatnonzero(w_sub_sum[0, :, p_idx] > 0)
            if len(kp):
                wkp = w_sub_sum[0, kp, p_idx]
                a0_paths[p_idx] = np.sum(wkp * y_sub[0, kp, p_idx] * np.exp(1j * 2.0 * np.pi * DF_HZ * kp * tau2_dom)) / max(float(np.sum(wkp)), 1e-12)

        sub_phase_hats = {int(unique_subs[0]): 0.0}
        align_methods: list[str] = []
        for m_row in range(1, n_subs):
            s_id = int(unique_subs[m_row])
            ov = (w_sub_sum[0] > 0) & (w_sub_sum[m_row] > 0) & (np.abs(y_sub[0]) > 1e-15) & (np.abs(y_sub[m_row]) > 1e-15)
            psi_m = 0.0
            if np.any(ov):
                z_ov = np.sum(np.minimum(w_sub_sum[0][ov], w_sub_sum[m_row][ov]) * y_sub[m_row][ov] * np.conj(y_sub[0][ov]))
                if np.abs(z_ov) > 1e-15:
                    psi_m = float(np.angle(z_ov))
                    align_methods.append("overlap")
                else:
                    align_methods.append("zero_ov")
            else:
                z_tpl = 0.0 + 0.0j
                for p_idx in range(n_paths):
                    if w0_paths[p_idx] <= 0 or abs(a0_paths[p_idx]) <= 1e-15:
                        continue
                    km = np.flatnonzero(w_sub_sum[m_row, :, p_idx] > 0)
                    if len(km):
                        wkm = w_sub_sum[m_row, km, p_idx]
                        am_p = np.sum(wkm * y_sub[m_row, km, p_idx] * np.exp(1j * 2.0 * np.pi * DF_HZ * km * tau2_dom)) / max(float(np.sum(wkm)), 1e-12)
                        z_tpl += min(w0_paths[p_idx], float(np.sum(wkm))) * am_p * np.conj(a0_paths[p_idx])
                if abs(z_tpl) > 1e-15:
                    psi_m = float(np.angle(z_tpl))
                    align_methods.append("template")
                else:
                    align_methods.append("zero_tpl")
            sub_phase_hats[s_id] = psi_m
            if psi_m != 0.0:
                y_cal[sub_arr == s_id] *= np.exp(-1j * psi_m)
        if align_methods:
            align_status = "aligned_" + ("overlap" if all(m == "overlap" for m in align_methods) else ("template" if any(m == "template" for m in align_methods) else "mixed"))

    y_sum = np.zeros((N_GRID, n_paths), dtype=np.complex128)
    w_sum = np.zeros((N_GRID, n_paths), dtype=np.float64)
    visit_counts = np.zeros((N_GRID, n_paths), dtype=np.int32)
    for idx_s in range(n_steps):
        k = int(grid_k[idx_s])
        p = int(ant_idx[idx_s])
        if 0 <= k < N_GRID and 0 <= p < n_paths and rho[idx_s] > 0:
            y_sum[k, p] += rho[idx_s] * y_cal[idx_s]
            w_sum[k, p] += rho[idx_s]
            visit_counts[k, p] += 1

    obs_mask = w_sum > 0
    y_full = np.zeros((N_GRID, n_paths), dtype=np.complex128)
    y_full[obs_mask] = y_sum[obs_mask] / w_sum[obs_mask]

    # Estimate per-step variance from repeated-visit residuals after de-rotation
    res_energy = 0.0
    dof_sum = 0.0
    for idx_s in range(n_steps):
        k = int(grid_k[idx_s])
        p = int(ant_idx[idx_s])
        if 0 <= k < N_GRID and 0 <= p < n_paths and rho[idx_s] > 0 and visit_counts[k, p] >= 2:
            diff = y_cal[idx_s] - y_full[k, p]
            res_energy += rho[idx_s] * float(np.real(diff * np.conj(diff)))
            dof_sum += rho[idx_s] * (visit_counts[k, p] - 1.0) / visit_counts[k, p]

    if dof_sum >= 3.0 and res_energy > 0.0:
        sigma2_step_hat = float(res_energy / dof_sum)
    else:
        # Fallback noise floor from high-delay tail of zero-padded IFFT
        cir = np.abs(np.fft.ifft(y_full, n=512, axis=0)) ** 2
        tail = cir[300:460, :]
        med_tail = float(np.median(tail)) * (512.0 / max(np.sum(obs_mask) / n_paths, 1.0))
        mean_y2 = float(np.mean(np.abs(y_full[obs_mask]) ** 2)) if np.any(obs_mask) else 1.0
        sigma2_step_hat = max(med_tail, 1e-8 * mean_y2)

    return {
        "valid_input": True,
        "input_status": "valid",
        "n_rows_input": n_rows_input,
        "n_rows_admissible": n_rows_admissible,
        "n_rows_discarded": n_rows_discarded,
        "n_unique_usable_tones": int(adm["n_unique_usable_tones"]),
        "w_step_hat": w_hat,
        "coh_peak": coh_peak,
        "n_pairs": n_pairs,
        "lag_gcd": lag_gcd,
        "drift_status": drift_status,
        "subevent_phase_hats": sub_phase_hats,
        "subevent_align_status": align_status,
        "y_full": y_full,
        "w_obs": w_sum,
        "obs_mask": obs_mask,
        "sigma2_step_hat": sigma2_step_hat,
    }


def stage2_mmv_anm_admm(
    y_full: np.ndarray,
    w_obs: np.ndarray,
    sigma2_step_hat: float,
    rho_admm: float = 1.0,
    max_iter: int | None = None,
    tol: float = 1.5e-3,
    alpha_relax: float = 1.6,
) -> dict:
    """Pure-NumPy over-relaxed ADMM for quality-weighted MMV Atomic Norm Minimization."""
    n, n_paths = y_full.shape
    mask = w_obs > 0
    m_obs = int(np.sum(mask))
    if (
        m_obs == 0
        or not np.all(np.isfinite(y_full[mask]))
        or float(np.sum(w_obs[mask] * (np.abs(y_full[mask]) ** 2))) <= 1e-24
    ):
        return {
            "x_hat": y_full.copy(),
            "u_hat": np.zeros(n, dtype=np.complex128),
            "T_hat": np.eye(n, dtype=np.complex128),
            "scale": 1.0,
            "n_iter": 0,
            "admm_converged": False,
            "admm_primal_rel": float("nan"),
            "admm_dual_rel": float("nan"),
        }

    n_unique_tones = int(np.sum(np.any(mask, axis=1)))
    if max_iter is None:
        max_iter = 20 if n_unique_tones >= 48 else 40

    mean_w = float(np.mean(w_obs[mask]))
    w_tilde = np.where(mask, w_obs / max(mean_w, 1e-12), 0.0)
    mean_sig = float(np.sum(w_obs[mask] * (np.abs(y_full[mask]) ** 2)) / np.sum(w_obs[mask]))
    scale = max(math.sqrt(mean_sig), 1e-12)
    y_norm = y_full / scale

    sigma_norm2 = sigma2_step_hat / ((scale ** 2) * max(mean_w, 1.0))
    sigma_norm2 = float(np.clip(sigma_norm2, 5e-4, 0.25))
    lam_anm = float(np.clip(0.55 * math.sqrt(sigma_norm2 * m_obs * math.log(n)), 0.15, 2.5))

    dim = n + n_paths
    x_mat = y_norm.copy()
    u_vec = np.zeros(n, dtype=np.complex128)
    u_vec[0] = float(np.sum(np.abs(y_norm[mask]) ** 2) / max(m_obs, 1))
    w_mat = np.eye(n_paths, dtype=np.complex128)
    q_mat = np.zeros((dim, dim), dtype=np.complex128)
    lam_dual = np.zeros((dim, dim), dtype=np.complex128)

    denom_x = w_tilde + 2.0 * rho_admm
    y_weighted = w_tilde * y_norm
    u0_shift = lam_anm / (2.0 * rho_admm * n)
    w_shift = (lam_anm / (2.0 * rho_admm)) * np.eye(n_paths, dtype=np.complex128)

    z_mat = np.zeros((dim, dim), dtype=np.complex128)
    n_iter = 0
    converged = False
    pri_rel = float("inf")
    dual_rel = float("inf")
    for it in range(1, max_iter + 1):
        n_iter = it
        m_mat = q_mat - lam_dual
        m11 = m_mat[:n, :n]
        m12 = m_mat[:n, n:]
        m22 = m_mat[n:, n:]

        # Closed-form (u, X, W) updates
        u_vec = _toeplitz_diag_means(m11)
        u_vec[0] = np.real(u_vec[0]) - u0_shift
        x_mat = (y_weighted + 2.0 * rho_admm * m12) / denom_x
        w_mat = 0.5 * (m22 + m22.conj().T) - w_shift

        t_mat = _toeplitz_hermitian(u_vec)
        z_mat[:n, :n] = t_mat
        z_mat[:n, n:] = x_mat
        z_mat[n:, :n] = x_mat.conj().T
        z_mat[n:, n:] = w_mat

        # Over-relaxed split update
        z_hat = alpha_relax * z_mat + (1.0 - alpha_relax) * q_mat

        # PSD projection Q = P_+(Z_hat + Lambda)
        b_mat = 0.5 * ((z_hat + lam_dual) + (z_hat + lam_dual).conj().T)
        evals, evecs = np.linalg.eigh(b_mat)
        pos = evals > 0.0
        q_prev = q_mat
        if np.any(pos):
            v_pos = evecs[:, pos]
            q_mat = (v_pos * evals[pos][None, :]) @ v_pos.conj().T
            q_mat = 0.5 * (q_mat + q_mat.conj().T)
        else:
            q_mat = np.zeros((dim, dim), dtype=np.complex128)

        lam_dual = lam_dual + (z_hat - q_mat)

        if (it >= 9 and it % 3 == 0) or (it == max_iter):
            norm_pri = float(np.linalg.norm(z_mat - q_mat, "fro"))
            norm_dual = float(rho_admm * np.linalg.norm(q_mat - q_prev, "fro"))
            ref = max(float(np.linalg.norm(z_mat, "fro")), 1.0)
            pri_rel = norm_pri / ref
            dual_rel = norm_dual / ref
            if pri_rel < tol and dual_rel < tol:
                converged = True
                break

    return {
        "x_hat": x_mat * scale,
        "u_hat": u_vec * (scale ** 2),
        "T_hat": _toeplitz_hermitian(u_vec) * (scale ** 2),
        "scale": scale,
        "lam_anm": lam_anm,
        "n_iter": n_iter,
        "admm_converged": converged,
        "admm_primal_rel": pri_rel,
        "admm_dual_rel": dual_rel,
    }


def _esprit_from_toeplitz(t_mat: np.ndarray, rank_r: int, tau_2way_max: float) -> np.ndarray:
    """Extract candidate two-way delays in [0, tau_2way_max] from Toeplitz matrix."""
    evals, evecs = np.linalg.eigh(t_mat)
    idx_desc = np.argsort(evals)[::-1]
    evals = evals[idx_desc]
    evecs = evecs[:, idx_desc]

    valid_r = min(rank_r, int(np.sum(evals > max(1e-5 * evals[0], 1e-12))))
    if valid_r < 1:
        return np.array([], dtype=np.float64)

    u_s = evecs[:, :valid_r]
    u1 = u_s[:-1, :]
    u2 = u_s[1:, :]
    psi, _, _, _ = np.linalg.lstsq(u1, u2, rcond=None)
    mu = np.linalg.eigvals(psi)
    mags = np.abs(mu)
    keep = (mags > 0.45) & (mags < 1.55)
    mu = mu[keep]
    if len(mu) == 0:
        return np.array([], dtype=np.float64)

    angles = -np.angle(mu)  # e^{-j 2*pi*df*tau} convention
    tau_2way = np.mod(angles / (2.0 * np.pi * DF_HZ), 1.0 / DF_HZ)
    # Map near-wrap values slightly below 1/DF_HZ cleanly to 0 if close to 1/DF_HZ
    tau_2way = np.where(tau_2way > 0.95 / DF_HZ, 0.0, tau_2way)
    tau_2way = tau_2way[tau_2way <= tau_2way_max]
    return np.sort(np.unique(np.round(tau_2way, 11)))


def _cluster_delays(taus: np.ndarray, min_sep: float = 1.5e-9) -> np.ndarray:
    """Deduplicate close delay candidates while preserving earliest representatives."""
    if len(taus) == 0:
        return taus
    s = np.sort(taus)
    kept = [float(s[0])]
    for val in s[1:]:
        if val - kept[-1] >= min_sep:
            kept.append(float(val))
    return np.asarray(kept, dtype=np.float64)


def _score_two_way_atoms(
    tilde_taus: np.ndarray,
    k_obs: np.ndarray,
    y_mat_obs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Score extracted two-way delays by energy on observed masked channels."""
    if len(tilde_taus) == 0:
        return tilde_taus, np.array([], dtype=np.float64)
    a_mat = np.exp(-1j * 2.0 * np.pi * DF_HZ * np.outer(k_obs, tilde_taus))
    # Regularized LS to stabilize across large gaps
    gram = a_mat.conj().T @ a_mat + 1e-2 * len(k_obs) * np.eye(len(tilde_taus))
    rhs = a_mat.conj().T @ y_mat_obs
    coeffs = np.linalg.solve(gram, rhs)
    pows = np.sum(np.abs(coeffs) ** 2, axis=1)
    return tilde_taus, pows


def extract_peeling_seeds(
    x_hat: np.ndarray,
    t_hat: np.ndarray,
    y_full: np.ndarray,
    obs_mask: np.ndarray,
    tau_max_1way: float,
    use_anm: bool = True,
    tau_min_1way: float = 0.35 / C_LIGHT,
) -> dict:
    """Generate sorted candidate one-way delay seeds from Stage 2 Toeplitz + CIR."""
    tau_2way_max = 2.0 * tau_max_1way
    cand_2way: list[float] = []

    if use_anm:
        for r in (2, 4, 6, 8):
            t_r = _esprit_from_toeplitz(t_hat, rank_r=r, tau_2way_max=tau_2way_max)
            cand_2way.extend(t_r.tolist())

        # Also extract top local peaks from ADMM-completed CIR
        n_fft = 2048
        cir = np.sum(np.abs(np.fft.ifft(x_hat, n=n_fft, axis=0)) ** 2, axis=1)
        max_idx = min(n_fft - 2, int(math.ceil(tau_2way_max * DF_HZ * n_fft)))
        pks: list[tuple[float, int]] = []
        for i in range(1, max_idx):
            if cir[i] >= cir[i - 1] and cir[i] >= cir[i + 1]:
                pks.append((float(cir[i]), i))
        if cir[0] > cir[1]:
            pks.append((float(cir[0]), 0))
        pks.sort(reverse=True)
        for _, i in pks[:6]:
            cand_2way.append(i / (n_fft * DF_HZ))
    else:
        # Masked CIR fallback when Stage-2 ANM is ablated off
        n_fft = 2048
        cir = np.sum(np.abs(np.fft.ifft(y_full, n=n_fft, axis=0)) ** 2, axis=1)
        max_idx = min(n_fft - 2, int(math.ceil(tau_2way_max * DF_HZ * n_fft)))
        pks = []
        for i in range(1, max_idx):
            if cir[i] >= cir[i - 1] and cir[i] >= cir[i + 1]:
                pks.append((float(cir[i]), i))
        pks.sort(reverse=True)
        for _, i in pks[:6]:
            cand_2way.append(i / (n_fft * DF_HZ))

    tilde_taus = _cluster_delays(np.asarray(cand_2way, dtype=np.float64), min_sep=1.5e-9)
    # Rank by LS power on observed unique frequency mask (across all antennas)
    k_active = np.flatnonzero(np.any(obs_mask, axis=1))
    tilde_taus, pows = _score_two_way_atoms(tilde_taus, k_active, y_full[k_active, :])

    if len(tilde_taus) == 0:
        tilde_taus = np.array([2.0 * 5.0 / C_LIGHT], dtype=np.float64)
        pows = np.array([1.0], dtype=np.float64)

    order_pow = np.argsort(pows)[::-1]
    strong_mask = pows >= 0.04 * np.max(pows)
    strong_2way = np.sort(tilde_taus[strong_mask])

    # Build L=1 seeds: top-power self-terms + earliest strong self-terms + residual fallback
    seeds_l1: list[float] = []
    for idx in order_pow[:4]:
        seeds_l1.append(float(np.clip(0.5 * tilde_taus[idx], tau_min_1way, tau_max_1way)))
    for t2 in strong_2way[:3]:
        seeds_l1.append(float(np.clip(0.5 * t2, tau_min_1way, tau_max_1way)))

    # Add a coarse 1-path masked correlator peak for split-band robustness
    n_corr_grid = max(180, int(math.ceil(4.0 * tau_max_1way * C_LIGHT)))
    tau_grid = np.linspace(tau_min_1way, tau_max_1way, n_corr_grid)
    steered = np.exp(1j * 4.0 * np.pi * DF_HZ * np.outer(tau_grid, k_active))
    corr_scores = np.sum(np.abs(steered @ y_full[k_active, :]) ** 2, axis=1)
    top_corr_idxs = []
    for i in np.argsort(corr_scores)[::-1]:
        t_val = float(tau_grid[i])
        if all(abs(t_val - prev) > 2.5e-9 for prev in top_corr_idxs):
            top_corr_idxs.append(t_val)
            if len(top_corr_idxs) >= 2:
                break
    seeds_l1.extend(top_corr_idxs)
    seeds_l1_arr = _cluster_delays(np.asarray(seeds_l1, dtype=np.float64), min_sep=1.8e-9)
    # _cluster_delays sorts ascending, so index 0 guarantees earliest retained candidate; keep up to 4
    seeds_l1_kept = seeds_l1_arr[:4]

    return {
        "tilde_taus": tilde_taus,
        "pows": pows,
        "strong_2way": strong_2way,
        "seeds_l1": seeds_l1_kept,
    }


def _detect_split_band_grating_tau(k_unique: np.ndarray) -> float | None:
    """Detect large interior gap >= 12 MHz and return two-way grating period 1/(2*df_gap)."""
    if len(k_unique) < 2:
        return None
    k_sorted = np.sort(k_unique)
    diffs = np.diff(k_sorted)
    if np.max(diffs) < 12:
        return None
    idx_gap = int(np.argmax(diffs))
    k_left = k_sorted[: idx_gap + 1]
    k_right = k_sorted[idx_gap + 1 :]
    df_gap_hz = (float(np.mean(k_right)) - float(np.mean(k_left))) * DF_HZ
    if df_gap_hz <= 0.0:
        return None
    return 1.0 / (2.0 * df_gap_hz)


def _init_amplitudes_for_taus(
    taus: np.ndarray,
    k_all: np.ndarray,
    p_all: np.ndarray,
    y_norm_vec: np.ndarray,
    w_vec: np.ndarray,
    n_paths: int,
) -> np.ndarray:
    """Solve quadratic-term LS + relative sign selection to initialize alphas (P x L)."""
    l_cnt = len(taus)
    u_vec = 2.0 * np.pi * DF_HZ * taus
    pair_sums: list[float] = []
    pair_ij: list[tuple[int, int]] = []
    for i in range(l_cnt):
        pair_sums.append(2.0 * u_vec[i])
        pair_ij.append((i, i))
    for i in range(l_cnt):
        for j in range(i + 1, l_cnt):
            pair_sums.append(u_vec[i] + u_vec[j])
            pair_ij.append((i, j))

    alphas = np.zeros((n_paths, l_cnt), dtype=np.complex128)
    w_sqrt = np.sqrt(w_vec)

    for p_idx in range(n_paths):
        sel = p_all == p_idx
        if not np.any(sel):
            alphas[p_idx, :] = 1e-3
            continue
        k_p = k_all[sel]
        y_p = y_norm_vec[sel]
        ws_p = w_sqrt[sel]

        design = np.exp(-1j * np.outer(k_p, np.asarray(pair_sums, dtype=np.float64)))
        wd = design * ws_p[:, None]
        wy = y_p * ws_p
        # Mild ridge stabilizes close delays across large holes
        reg = 1e-2 * np.sum(w_vec[sel]) * np.eye(len(pair_sums))
        beta = np.linalg.solve(wd.conj().T @ wd + reg, wd.conj().T @ wy)

        # Self terms beta[0..L-1] = alpha_{p,l}^2
        base_a = np.sqrt(beta[:l_cnt])
        if l_cnt == 1:
            alphas[p_idx, 0] = base_a[0]
            continue

        # Choose sign bits s_l in {+1, -1} (fix dominant path sign = +1)
        dom = int(np.argmax(np.abs(base_a)))
        other_idxs = [idx for idx in range(l_cnt) if idx != dom]
        e_1way = np.exp(-1j * np.outer(k_p, u_vec))

        best_rss = float("inf")
        best_a = base_a.copy()
        n_combs = 1 << len(other_idxs)
        for mask_bits in range(n_combs):
            a_cand = base_a.copy()
            for bit_i, l_idx in enumerate(other_idxs):
                if (mask_bits >> bit_i) & 1:
                    a_cand[l_idx] = -a_cand[l_idx]
            h_cand = e_1way @ a_cand
            rss = float(np.sum(w_vec[sel] * np.abs(y_p - (h_cand ** 2)) ** 2))
            if rss < best_rss:
                best_rss = rss
                best_a = a_cand
        alphas[p_idx, :] = best_a
    return alphas


def _fit_fixed_order_qml(
    seed_taus: np.ndarray,
    k_all: np.ndarray,
    p_all: np.ndarray,
    y_norm_vec: np.ndarray,
    w_vec: np.ndarray,
    n_paths: int,
    tau_max_1way: float,
    tau_min_1way: float = 0.35 / C_LIGHT,
    reg_amp: float = 1e-3,
    max_nfev: int = 50,
) -> dict:
    """Exact masked Quadratic-CFR ML fit with analytical Jacobian."""
    l_cnt = len(seed_taus)
    m_obs = len(y_norm_vec)
    seed_taus = np.sort(np.clip(seed_taus, tau_min_1way, tau_max_1way))
    alphas0 = _init_amplitudes_for_taus(seed_taus, k_all, p_all, y_norm_vec, w_vec, n_paths)

    u_min = 2.0 * np.pi * DF_HZ * tau_min_1way
    u_max = 2.0 * np.pi * DF_HZ * tau_max_1way
    u0 = np.clip(2.0 * np.pi * DF_HZ * seed_taus, u_min, u_max)
    theta0 = np.concatenate([u0, np.real(alphas0).ravel(), np.imag(alphas0).ravel()])

    lb = np.full_like(theta0, -15.0)
    ub = np.full_like(theta0, 15.0)
    lb[:l_cnt] = u_min
    ub[:l_cnt] = u_max

    w_sqrt = np.sqrt(w_vec)
    pl_cnt = n_paths * l_cnt
    k_float = k_all.astype(np.float64)
    p_rows = [np.flatnonzero(p_all == p_idx) for p_idx in range(n_paths)]
    amp_scale_reg = math.sqrt(reg_amp * max(float(np.sum(w_vec)), 1.0) / max(2 * pl_cnt, 1))
    eye_reg = amp_scale_reg * np.eye(2 * pl_cnt)

    # Cache last evaluation so paired fun(theta) + jac(theta) calls avoid recomputing exp/h
    cache: dict[str, object] = {"theta": None, "e_mat": None, "a_step": None, "h": None}

    def _eval_forward(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        prev_th = cache["theta"]
        if prev_th is not None and np.array_equal(theta, prev_th):
            return cache["e_mat"], cache["a_step"], cache["h"]  # type: ignore[return-value]
        u = theta[:l_cnt]
        a_re = theta[l_cnt : l_cnt + pl_cnt].reshape(n_paths, l_cnt)
        a_im = theta[l_cnt + pl_cnt :].reshape(n_paths, l_cnt)
        alphas = a_re + 1j * a_im

        e_mat = np.exp(-1j * np.outer(k_float, u))  # (m_obs, L)
        a_step = alphas[p_all, :]  # (m_obs, L)
        h = np.sum(a_step * e_mat, axis=1)  # (m_obs,)
        cache["theta"] = theta.copy()
        cache["e_mat"] = e_mat
        cache["a_step"] = a_step
        cache["h"] = h
        return e_mat, a_step, h

    def fun(theta: np.ndarray) -> np.ndarray:
        _, _, h = _eval_forward(theta)
        mu = h ** 2
        res_c = w_sqrt * (mu - y_norm_vec)
        return np.concatenate([np.real(res_c), np.imag(res_c), amp_scale_reg * theta[l_cnt:]])

    def jac(theta: np.ndarray) -> np.ndarray:
        e_mat, a_step, h = _eval_forward(theta)
        n_theta = len(theta)
        j_mat = np.zeros((2 * m_obs + 2 * pl_cnt, n_theta), dtype=np.float64)

        # Derivatives wrt u_l: d(mu)/d(u_l) = -j * 2 * k * h * alpha_{p,l} * e_{k,l}
        d_u = (-2j * k_float * w_sqrt * h)[:, None] * a_step * e_mat  # (m_obs, L)
        j_mat[:m_obs, :l_cnt] = np.real(d_u)
        j_mat[m_obs : 2 * m_obs, :l_cnt] = np.imag(d_u)

        # Derivatives wrt Re/Im(alpha_{p,l}) indexed directly by active antenna rows
        base = (2.0 * w_sqrt * h)[:, None] * e_mat  # (m_obs, L)
        for p_idx in range(n_paths):
            rows = p_rows[p_idx]
            if len(rows) == 0:
                continue
            blk = base[rows, :]
            col_re = l_cnt + p_idx * l_cnt
            col_im = l_cnt + pl_cnt + p_idx * l_cnt
            j_mat[rows, col_re : col_re + l_cnt] = np.real(blk)
            j_mat[m_obs + rows, col_re : col_re + l_cnt] = np.imag(blk)
            j_mat[rows, col_im : col_im + l_cnt] = -np.imag(blk)
            j_mat[m_obs + rows, col_im : col_im + l_cnt] = np.real(blk)

        j_mat[2 * m_obs :, l_cnt:] = eye_reg
        return j_mat

    sol = least_squares(
        fun,
        theta0,
        jac=jac,
        bounds=(lb, ub),
        method="trf",
        ftol=1e-7,
        xtol=1e-7,
        gtol=1e-7,
        max_nfev=max_nfev,
    )
    th = sol.x
    taus_fit = th[:l_cnt] / (2.0 * np.pi * DF_HZ)
    alphas_fit = (th[l_cnt : l_cnt + pl_cnt] + 1j * th[l_cnt + pl_cnt :]).reshape(n_paths, l_cnt)

    # Sort fitted paths ascending
    order = np.argsort(taus_fit)
    taus_fit = taus_fit[order]
    alphas_fit = alphas_fit[:, order]

    e_mat = np.exp(-1j * 2.0 * np.pi * DF_HZ * np.outer(k_float, taus_fit))
    h_fit = np.sum(alphas_fit[p_all, :] * e_mat, axis=1)
    mu_fit = h_fit ** 2
    rss_unreg = float(np.sum(w_vec * (np.abs(mu_fit - y_norm_vec) ** 2)))

    return {
        "taus": taus_fit,
        "alphas": alphas_fit,
        "mu_norm": mu_fit,
        "rss_norm": rss_unreg,
        "L": l_cnt,
        "qml_success": bool(sol.success),
        "qml_status": int(sol.status),
        "qml_optimality": float(sol.optimality),
        "qml_nfev": int(sol.nfev),
    }


def _select_fap_from_fit(
    fit: dict,
    k_all: np.ndarray,
    p_all: np.ndarray,
    y_norm_vec: np.ndarray,
    w_vec: np.ndarray,
    eta_los: float = 0.12,
    min_sep_ns: float = 3.3e-9,
    tau_min_1way: float = 0.35 / C_LIGHT,
) -> float:
    """Merge sub-Rayleigh splits, drop boundary pins, and pick earliest significant path."""
    taus = fit["taus"].copy()
    alphas = fit["alphas"].copy()
    l_cnt = len(taus)
    if l_cnt == 1:
        return float(C_LIGHT * taus[0])

    # Coherently merge sub-Rayleigh split fragments via phase-aligned LS projection
    n_paths = alphas.shape[0]
    merged_t: list[float] = []
    merged_a: list[np.ndarray] = []
    used = np.zeros(l_cnt, dtype=bool)
    for i in range(l_cnt):
        if used[i]:
            continue
        grp = [i]
        for j in range(i + 1, l_cnt):
            if not used[j] and abs(taus[j] - taus[i]) < min_sep_ns:
                grp.append(j)
        for j in grp:
            used[j] = True
        p_grp = np.sum(np.abs(alphas[:, grp]) ** 2, axis=0)
        if np.sum(p_grp) > 0:
            t_avg = float(np.sum(taus[grp] * p_grp) / np.sum(p_grp))
        else:
            t_avg = float(taus[i])
        if len(grp) == 1:
            a_sum = alphas[:, grp[0]].copy()
        else:
            a_sum = np.empty(n_paths, dtype=np.complex128)
            for p_idx in range(n_paths):
                sel = p_all == p_idx
                if np.any(sel):
                    k_p = k_all[sel].astype(np.float64)
                    w_p = w_vec[sel]
                    h_grp = np.sum(
                        alphas[p_idx, grp][None, :]
                        * np.exp(-1j * 2.0 * np.pi * DF_HZ * np.outer(k_p, taus[grp])),
                        axis=1,
                    )
                    a_sum[p_idx] = np.sum(
                        w_p * h_grp * np.exp(1j * 2.0 * np.pi * DF_HZ * k_p * t_avg)
                    ) / max(float(np.sum(w_p)), 1e-12)
                else:
                    a_sum[p_idx] = np.sum(alphas[p_idx, grp])
        merged_t.append(t_avg)
        merged_a.append(a_sum)

    taus_m = np.asarray(merged_t, dtype=np.float64)
    alphas_m = np.column_stack(merged_a) if len(merged_a) else alphas
    l_m = len(taus_m)

    # Filter paths pinned at lower bound unless sole component
    pin_tol = tau_min_1way + 0.05 / C_LIGHT
    non_pinned = [idx for idx in range(l_m) if taus_m[idx] > pin_tol]
    if non_pinned and len(non_pinned) < l_m:
        taus_m = taus_m[non_pinned]
        alphas_m = alphas_m[:, non_pinned]
        l_m = len(taus_m)

    if l_m == 1:
        return float(C_LIGHT * taus_m[0])

    pows = np.sum(np.abs(alphas_m) ** 2, axis=0)
    max_pow = float(np.max(pows)) if len(pows) else 1.0

    # Compute leave-one-out net RSS contribution to reject canceling dipole noise paths
    e_mat = np.exp(-1j * 2.0 * np.pi * DF_HZ * np.outer(k_all.astype(np.float64), taus_m))
    h_full = np.sum(alphas_m[p_all, :] * e_mat, axis=1)
    rss_full = float(np.sum(w_vec * (np.abs(h_full ** 2 - y_norm_vec) ** 2)))
    delta_rss = np.zeros(l_m, dtype=np.float64)
    for l_idx in range(l_m):
        term_l = alphas_m[p_all, l_idx] * e_mat[:, l_idx]
        h_minus = h_full - term_l
        rss_minus = float(np.sum(w_vec * (np.abs(h_minus ** 2 - y_norm_vec) ** 2)))
        delta_rss[l_idx] = max(0.0, rss_minus - rss_full)
    max_drss = max(float(np.max(delta_rss)), 1e-12)

    for l_idx in range(l_m):
        if (pows[l_idx] >= eta_los * max_pow) and (delta_rss[l_idx] >= 0.04 * max_drss):
            return float(C_LIGHT * taus_m[l_idx])

    dom = int(np.argmax(delta_rss))
    return float(C_LIGHT * taus_m[dom])


def estimate_range_sdr_anm_qml(
    step_idx: np.ndarray,
    grid_k: np.ndarray,
    ant_idx: np.ndarray,
    y: np.ndarray,
    q_i: np.ndarray,
    q_r: np.ndarray,
    n_paths: int = 1,
    d_cal_m: float = 0.0,
    d_max_m: float = 45.0,
    l_max: int = 4,
    eta_los: float = 0.12,
    bic_scale: float = 1.5,
    enable_stage1_drift: bool = True,
    enable_stage2_anm: bool = True,
    enable_stage3_qml: bool = True,
    subevent_idx: np.ndarray | None = None,
) -> dict:
    """Estimate first-path range using the 3-stage SDR-ANM-QML algorithm."""
    tau_min_1way = max(0.35, d_cal_m + 0.10) / C_LIGHT
    tau_max_1way = d_max_m / C_LIGHT

    # Stage 1: Drift calibration + quality-weighted aggregation
    st1 = stage1_step_drift_and_aggregate(
        step_idx=step_idx,
        grid_k=grid_k,
        ant_idx=ant_idx,
        y=y,
        q_i=q_i,
        q_r=q_r,
        n_paths=n_paths,
        enable_drift=enable_stage1_drift,
        subevent_idx=subevent_idx,
    )
    y_full = st1["y_full"]
    w_obs = st1["w_obs"]
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
            "taus_hat": np.array([], dtype=np.float64),
            "alphas_hat": np.zeros((n_paths, 0), dtype=np.complex128),
            "L_star": 0,
            "w_step_hat": st1["w_step_hat"],
            "coh_peak": st1["coh_peak"],
            "lag_gcd": st1["lag_gcd"],
            "drift_status": st1["drift_status"],
            "subevent_phase_hats": st1.get("subevent_phase_hats", {0: 0.0}),
            "subevent_align_status": st1.get("subevent_align_status", "invalid_input"),
            "n_admm_iter": 0,
            "admm_converged": False,
            "admm_primal_rel": float("nan"),
            "admm_dual_rel": float("nan"),
            "qml_success": False,
            "qml_status": -2,
            "qml_optimality": float("nan"),
            "qml_nfev": 0,
            "rss_norm": float("nan"),
        }

    # Stage 2: MMV-ANM SDP via pure-NumPy ADMM
    if enable_stage2_anm:
        st2 = stage2_mmv_anm_admm(y_full, w_obs, st1["sigma2_step_hat"])
    else:
        mean_sig = (
            float(np.sum(w_obs[obs_mask] * (np.abs(y_full[obs_mask]) ** 2)) / np.sum(w_obs[obs_mask]))
            if np.any(obs_mask)
            else 1.0
        )
        st2 = {
            "x_hat": y_full.copy(),
            "u_hat": np.zeros(N_GRID, dtype=np.complex128),
            "T_hat": np.eye(N_GRID, dtype=np.complex128),
            "scale": max(math.sqrt(mean_sig), 1e-12),
            "n_iter": 0,
            "admm_converged": True,
            "admm_primal_rel": 0.0,
            "admm_dual_rel": 0.0,
        }

    seeds_info = extract_peeling_seeds(
        x_hat=st2["x_hat"],
        t_hat=st2["T_hat"],
        y_full=y_full,
        obs_mask=obs_mask,
        tau_max_1way=tau_max_1way,
        use_anm=enable_stage2_anm,
        tau_min_1way=tau_min_1way,
    )

    if not enable_stage3_qml:
        # Stage-2 ablation readout: first strong two-way / 2 from completed CIR
        t2_strong = seeds_info["strong_2way"]
        if len(t2_strong):
            tau_est = 0.5 * float(t2_strong[0])
        else:
            tau_est = float(seeds_info["seeds_l1"][0])
        return {
            "valid_estimate": True,
            "input_status": "valid",
            "d_hat_m": max(0.0, float(C_LIGHT * tau_est) - d_cal_m),
            "d_raw_m": float(C_LIGHT * tau_est),
            "taus_hat": np.array([tau_est]),
            "alphas_hat": np.zeros((n_paths, 1), dtype=np.complex128),
            "L_star": 1,
            "w_step_hat": st1["w_step_hat"],
            "coh_peak": st1["coh_peak"],
            "lag_gcd": st1["lag_gcd"],
            "drift_status": st1["drift_status"],
            "subevent_phase_hats": st1.get("subevent_phase_hats", {0: 0.0}),
            "subevent_align_status": st1.get("subevent_align_status", "single_subevent"),
            "n_admm_iter": st2["n_iter"],
            "admm_converged": st2["admm_converged"],
            "admm_primal_rel": st2["admm_primal_rel"],
            "admm_dual_rel": st2["admm_dual_rel"],
            "qml_success": False,
            "qml_status": -1,
            "qml_optimality": float("nan"),
            "qml_nfev": 0,
        }

    # Stage 3: Exact Masked Quadratic-CFR ML + Peeling Expansion + BIC
    k_all, p_all = np.nonzero(obs_mask)
    if len(k_all) == 0:
        return {
            "valid_estimate": False,
            "input_status": "empty_mask",
            "d_hat_m": float("nan"),
            "d_raw_m": float("nan"),
            "taus_hat": np.array([], dtype=np.float64),
            "alphas_hat": np.zeros((n_paths, 0), dtype=np.complex128),
            "L_star": 0,
            "w_step_hat": st1["w_step_hat"],
            "coh_peak": st1["coh_peak"],
            "lag_gcd": st1["lag_gcd"],
            "drift_status": st1["drift_status"],
            "subevent_phase_hats": st1.get("subevent_phase_hats", {0: 0.0}),
            "subevent_align_status": st1.get("subevent_align_status", "invalid_input"),
            "n_admm_iter": st2["n_iter"],
            "admm_converged": st2["admm_converged"],
            "admm_primal_rel": st2["admm_primal_rel"],
            "admm_dual_rel": st2["admm_dual_rel"],
            "qml_success": False,
            "qml_status": -2,
            "qml_optimality": float("nan"),
            "qml_nfev": 0,
            "rss_norm": float("nan"),
        }

    scale = st2["scale"]
    y_norm_vec = y_full[k_all, p_all] / scale
    mean_w = float(np.mean(w_obs[k_all, p_all]))
    w_vec = w_obs[k_all, p_all] / max(mean_w, 1e-12)
    m_eff = len(y_norm_vec)
    n_real = 2.0 * m_eff
    sigma_norm2 = st1["sigma2_step_hat"] / ((scale ** 2) * max(mean_w, 1.0))
    rss_floor = max(0.25 * sigma_norm2, 0.005) * m_eff

    # Cap max model order conservatively on severely thinned masks
    eff_l_max = l_max if n_unique_tones >= 28 else min(l_max, 3)

    k_unique = np.unique(k_all)
    k_float_all = k_all.astype(np.float64)
    delta_tau_g = _detect_split_band_grating_tau(k_unique)

    def _escape_grating_trap(best_fit: dict) -> dict:
        if delta_tau_g is None:
            return best_fit
        cur_best = best_fit
        base_taus = best_fit["taus"]
        shifts = (-2, -1, 1, 2) if n_unique_tones <= 45 else (-1, 1)
        for m_shift in shifts:
            shifted = base_taus + m_shift * delta_tau_g
            if np.all((shifted >= tau_min_1way) & (shifted <= tau_max_1way)):
                cand = _fit_fixed_order_qml(
                    shifted,
                    k_all,
                    p_all,
                    y_norm_vec,
                    w_vec,
                    n_paths,
                    tau_max_1way,
                    tau_min_1way=tau_min_1way,
                    max_nfev=18,
                )
                if cand["rss_norm"] < cur_best["rss_norm"]:
                    cur_best = cand
        return cur_best

    # Order L=1: multi-start over Stage-2 L=1 seeds + grating-lobe lattice escape
    best_by_l: dict[int, dict] = {}
    cand_fits_l1 = [
        _fit_fixed_order_qml(
            np.array([t1]),
            k_all,
            p_all,
            y_norm_vec,
            w_vec,
            n_paths,
            tau_max_1way,
            tau_min_1way=tau_min_1way,
            max_nfev=22,
        )
        for t1 in seeds_info["seeds_l1"]
    ]
    best_by_l[1] = _escape_grating_trap(min(cand_fits_l1, key=lambda d: d["rss_norm"]))

    tilde_taus = seeds_info["tilde_taus"]
    pows_2way = seeds_info["pows"]
    top_2way_idx = np.argsort(pows_2way)[::-1]

    # Greedy peeling + residual CIR expansion for L = 2..eff_l_max
    n_fft_res = 1024
    for l_cur in range(2, eff_l_max + 1):
        prev_fit = best_by_l[l_cur - 1]
        prev_taus = prev_fit["taus"]

        # Compute residual spectrum on observed grid points
        e_prev = np.exp(-1j * 2.0 * np.pi * DF_HZ * np.outer(k_float_all, prev_taus))
        h_prev = np.sum(prev_fit["alphas"][p_all, :] * e_prev, axis=1)
        res_grid = np.zeros((N_GRID, n_paths), dtype=np.complex128)
        res_grid[k_all, p_all] = y_norm_vec - (h_prev ** 2)
        cir_res = np.sum(np.abs(np.fft.ifft(res_grid, n=n_fft_res, axis=0)) ** 2, axis=1)
        max_bin = min(n_fft_res - 2, int(math.ceil(2.0 * tau_max_1way * DF_HZ * n_fft_res)))
        res_pks: list[tuple[float, int]] = []
        for i in range(1, max_bin):
            if cir_res[i] >= cir_res[i - 1] and cir_res[i] >= cir_res[i + 1]:
                res_pks.append((float(cir_res[i]), i))
        res_pks.sort(reverse=True)

        # Gather diverse candidate delays for new path:
        new_cands: list[float] = []
        # Test earliest retained Stage-2 seeds at L=2 alongside residual & peeling terms
        if l_cur == 2:
            early_added = 0
            for t_early in seeds_info["seeds_l1"]:
                if tau_min_1way <= t_early <= float(prev_taus[0]) - 1.5e-9:
                    new_cands.append(float(t_early))
                    early_added += 1
                    if early_added >= 2:
                        break
        # (a) Self & peeled cross-terms from residual CIR peaks
        for _, bin_i in res_pks[:2]:
            t2_r = bin_i / (n_fft_res * DF_HZ)
            new_cands.append(0.5 * t2_r)
            diff_1 = t2_r - float(prev_taus[0])
            if 1.5e-9 <= diff_1 <= tau_max_1way:
                new_cands.append(diff_1)
        # (b) Self & peeled cross-terms from Stage-2 ESPRIT/CIR atoms
        for idx in top_2way_idx[:6]:
            t2_a = float(tilde_taus[idx])
            new_cands.append(0.5 * t2_a)
            for tp in prev_taus[:2]:
                diff_p = t2_a - float(tp)
                if 1.5e-9 <= diff_p <= tau_max_1way:
                    new_cands.append(diff_p)

        # Also test coarse residual 1-way scan for suppressed early LOS path
        n_res_scan = max(120, int(math.ceil(120.0 * (d_max_m / 45.0))))
        tau_scan = np.linspace(tau_min_1way, tau_max_1way, n_res_scan)
        steered_res = np.exp(1j * 4.0 * np.pi * DF_HZ * np.outer(tau_scan, k_unique))
        res_u = np.sum(np.abs(steered_res @ res_grid[k_unique, :]) ** 2, axis=1)
        new_cands.append(float(tau_scan[int(np.argmax(res_u))]))

        # Keep priority order up to 4 candidates, guaranteeing earliest valid candidate in slot <=4
        valid_cands = [
            float(np.clip(tc, tau_min_1way, tau_max_1way))
            for tc in new_cands
            if all(abs(float(np.clip(tc, tau_min_1way, tau_max_1way)) - tp) >= 1.5e-9 for tp in prev_taus)
        ]
        filtered_cands: list[float] = []
        for tc_c in valid_cands:
            if all(abs(tc_c - prev_c) >= 1.2e-9 for prev_c in filtered_cands):
                filtered_cands.append(tc_c)
                if len(filtered_cands) >= 3:
                    break
        if valid_cands:
            earliest_tc = min(valid_cands)
            if all(abs(earliest_tc - prev_c) >= 1.2e-9 for prev_c in filtered_cands):
                filtered_cands.append(earliest_tc)

        if not filtered_cands:
            break

        trial_fits = [
            _fit_fixed_order_qml(
                np.append(prev_taus, tc),
                k_all,
                p_all,
                y_norm_vec,
                w_vec,
                n_paths,
                tau_max_1way,
                tau_min_1way=tau_min_1way,
                max_nfev=18,
            )
            for tc in filtered_cands
        ]
        # For L=2, also test pairing earliest two distinct L=1 runner-up minima if not already covered
        if l_cur == 2 and len(cand_fits_l1) >= 2:
            l1_sorted = sorted(cand_fits_l1, key=lambda d: d["rss_norm"])
            t_a = float(l1_sorted[0]["taus"][0])
            for alt in l1_sorted[1:3]:
                t_b = float(alt["taus"][0])
                if abs(t_a - t_b) >= 1.5e-9:
                    trial_fits.append(
                        _fit_fixed_order_qml(
                            np.array([t_a, t_b]),
                            k_all,
                            p_all,
                            y_norm_vec,
                            w_vec,
                            n_paths,
                            tau_max_1way,
                            tau_min_1way=tau_min_1way,
                            max_nfev=20,
                        )
                    )
                    break

        best_cur = min(trial_fits, key=lambda d: d["rss_norm"])
        if l_cur == 2:
            best_cur = _escape_grating_trap(best_cur)
        best_by_l[l_cur] = best_cur

    # BIC Model Order Selection
    best_bic = float("inf")
    l_star = 1
    for l_val, fit in best_by_l.items():
        dof = l_val + 2 * n_paths * l_val
        mean_rss = (fit["rss_norm"] + rss_floor + 1e-10) / m_eff
        bic = n_real * math.log(mean_rss) + bic_scale * dof * math.log(max(n_real, 4.0))
        if bic < best_bic:
            best_bic = bic
            l_star = l_val

    fit_star = best_by_l[l_star]
    if l_star >= 3:
        fit_star = _escape_grating_trap(fit_star)
    polished = _fit_fixed_order_qml(
        fit_star["taus"],
        k_all,
        p_all,
        y_norm_vec,
        w_vec,
        n_paths,
        tau_max_1way,
        tau_min_1way=tau_min_1way,
        max_nfev=40,
    )
    if polished["rss_norm"] < fit_star["rss_norm"]:
        fit_star = polished

    d_raw_m = _select_fap_from_fit(
        fit_star,
        k_all,
        p_all,
        y_norm_vec,
        w_vec,
        eta_los=eta_los,
        tau_min_1way=tau_min_1way,
    )
    return {
        "valid_estimate": True,
        "input_status": "valid",
        "n_rows_input": st1.get("n_rows_input", 0),
        "n_rows_admissible": st1.get("n_rows_admissible", 0),
        "n_rows_discarded": st1.get("n_rows_discarded", 0),
        "n_unique_usable_tones": int(st1.get("n_unique_usable_tones", n_unique_tones)),
        "d_hat_m": max(0.0, d_raw_m - d_cal_m),
        "d_raw_m": d_raw_m,
        "taus_hat": fit_star["taus"],
        "alphas_hat": fit_star["alphas"] * math.sqrt(scale),
        "L_star": l_star,
        "w_step_hat": st1["w_step_hat"],
        "coh_peak": st1["coh_peak"],
        "lag_gcd": st1["lag_gcd"],
        "drift_status": st1["drift_status"],
        "subevent_phase_hats": st1.get("subevent_phase_hats", {0: 0.0}),
        "subevent_align_status": st1.get("subevent_align_status", "single_subevent"),
        "n_admm_iter": st2["n_iter"],
        "admm_converged": st2["admm_converged"],
        "admm_primal_rel": st2["admm_primal_rel"],
        "admm_dual_rel": st2["admm_dual_rel"],
        "qml_success": fit_star["qml_success"],
        "qml_status": fit_star["qml_status"],
        "qml_optimality": fit_star["qml_optimality"],
        "qml_nfev": fit_star["qml_nfev"],
        "rss_norm": fit_star["rss_norm"],
    }
