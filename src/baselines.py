"""Literature baseline estimators on identical BLE CS procedure inputs."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .estimator import (
    _filter_and_validate_inputs,
    _validate_procedure_inputs,
    stage1_step_drift_and_aggregate,
    stage2_mmv_anm_admm,
)
from .model import C_LIGHT, DF_HZ, N_GRID


def _aggregate_quality_grid(
    grid_k: np.ndarray,
    ant_idx: np.ndarray,
    y: np.ndarray,
    q_i: np.ndarray,
    q_r: np.ndarray,
    n_paths: int = 1,
    first_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate observed tones onto (N_GRID, P) complex array."""
    q_max = np.maximum(q_i, q_r)
    good = q_max == 0
    if np.sum(good) < max(4, len(y) // 10):
        good = q_max <= 1
    if not np.any(good):
        good = np.ones(len(y), dtype=bool)

    y_grid = np.zeros((N_GRID, n_paths), dtype=np.complex128)
    counts = np.zeros((N_GRID, n_paths), dtype=np.float64)
    for idx_s in np.flatnonzero(good):
        k = int(grid_k[idx_s])
        p = int(ant_idx[idx_s])
        if 0 <= k < N_GRID and 0 <= p < n_paths:
            if first_only and counts[k, p] > 0:
                continue
            y_grid[k, p] += y[idx_s]
            counts[k, p] += 1.0
    mask = counts > 0
    y_grid[mask] /= counts[mask]
    return y_grid, mask


def _prepare_grid_with_meta(
    grid_k: np.ndarray,
    ant_idx: np.ndarray,
    y: np.ndarray,
    q_i: np.ndarray,
    q_r: np.ndarray,
    n_paths: int = 1,
    first_only: bool = False,
    step_idx: np.ndarray | None = None,
    apply_stage1_drift: bool = False,
    subevent_idx: np.ndarray | None = None,
) -> tuple[bool, str, dict, np.ndarray, np.ndarray, np.ndarray, float]:
    """Validate/filter one admissible row set and return grid, mask, weights, sigma2, and row counts."""
    y_arr = np.asarray(y)
    s_chk = np.arange(len(y_arr), dtype=int) if step_idx is None else step_idx
    ok_in, in_stat, adm = _filter_and_validate_inputs(
        s_chk, grid_k, ant_idx, y_arr, q_i, q_r, n_paths=n_paths, subevent_idx=subevent_idx
    )
    meta = {
        "n_rows_input": int(adm["n_rows_input"]) if isinstance(adm, dict) and "n_rows_input" in adm else (len(y_arr) if y_arr.ndim == 1 else 0),
        "n_rows_admissible": int(adm["n_rows_admissible"]) if isinstance(adm, dict) and "n_rows_admissible" in adm else 0,
        "n_rows_discarded": int(adm["n_rows_discarded"]) if isinstance(adm, dict) and "n_rows_discarded" in adm else 0,
        "n_unique_usable_tones": int(adm.get("n_unique_usable_tones", 0)) if isinstance(adm, dict) else 0,
    }
    if not ok_in or adm is None or "step_idx" not in adm:
        return (
            False,
            in_stat,
            meta,
            np.zeros((N_GRID, n_paths), dtype=np.complex128),
            np.zeros((N_GRID, n_paths), dtype=bool),
            np.zeros((N_GRID, n_paths), dtype=np.float64),
            float("nan"),
        )

    s_adm = adm["step_idx"]
    k_adm = adm["grid_k"]
    a_adm = adm["ant_idx"]
    y_adm = adm["y"]
    qi_adm = adm["q_i"]
    qr_adm = adm["q_r"]
    sub_adm = adm["subevent_idx"]

    if apply_stage1_drift and step_idx is not None:
        st1 = stage1_step_drift_and_aggregate(
            step_idx=s_adm,
            grid_k=k_adm,
            ant_idx=a_adm,
            y=y_adm,
            q_i=qi_adm,
            q_r=qr_adm,
            n_paths=n_paths,
            enable_drift=True,
            subevent_idx=sub_adm,
        )
        n_obs = len(s_adm)
        psi_map = st1.get("subevent_phase_hats", {0: 0.0})
        if sub_adm is not None:
            multi_sub = n_obs > 0 and bool(np.any(sub_adm != sub_adm[0]))
            if multi_sub:
                sub_starts = {
                    int(s_id): int(s_adm[np.flatnonzero(sub_adm == s_id)[0]])
                    for s_id in np.unique(sub_adm)
                }
                step_eff = np.asarray(
                    [s_adm[i] - sub_starts[int(sub_adm[i])] for i in range(n_obs)],
                    dtype=np.float64,
                )
                psi_vec = np.asarray([psi_map.get(int(sub_adm[i]), 0.0) for i in range(n_obs)], dtype=np.float64)
            else:
                step_eff = s_adm.astype(np.float64)
                psi_vec = np.zeros(n_obs, dtype=np.float64)
        else:
            step_eff = s_adm.astype(np.float64)
            psi_vec = np.zeros(n_obs, dtype=np.float64)
        y_in = y_adm * np.exp(-1j * (st1["w_step_hat"] * step_eff + psi_vec))
    else:
        y_in = y_adm

    y_g, m_g = _aggregate_quality_grid(
        k_adm, a_adm, y_in, qi_adm, qr_adm, n_paths=n_paths, first_only=first_only
    )
    if (not np.any(m_g)) or float(np.max(np.abs(y_g))) <= 1e-15:
        return False, "zero_signal", meta, y_g, m_g, m_g.astype(np.float64), float("nan")
    # Apply the identical noise-floor rule across both raw and Stage-1-compensated arms
    cir_n = np.abs(np.fft.ifft(y_g, n=512, axis=0)) ** 2
    tail_n = cir_n[300:460, :]
    med_tail = float(np.median(tail_n)) * (512.0 / max(np.sum(m_g) / n_paths, 1.0))
    mean_y2 = float(np.mean(np.abs(y_g[m_g]) ** 2)) if np.any(m_g) else 1.0
    sigma2_hat = max(med_tail, 1e-8 * mean_y2)
    return True, "valid", meta, y_g, m_g, m_g.astype(np.float64), sigma2_hat


def _prepare_grid_with_optional_drift(
    grid_k: np.ndarray,
    ant_idx: np.ndarray,
    y: np.ndarray,
    q_i: np.ndarray,
    q_r: np.ndarray,
    n_paths: int = 1,
    first_only: bool = False,
    step_idx: np.ndarray | None = None,
    apply_stage1_drift: bool = False,
    subevent_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Prepare (N_GRID, P) CFR grid with strictly phase-only Stage-1 drift de-rotation control."""
    _, _, _, y_g, m_g, w_g, sigma2_hat = _prepare_grid_with_meta(
        grid_k, ant_idx, y, q_i, q_r,
        n_paths=n_paths, first_only=first_only, step_idx=step_idx,
        apply_stage1_drift=apply_stage1_drift, subevent_idx=subevent_idx,
    )
    return y_g, m_g, w_g, sigma2_hat


def _wrap_baseline_result(
    d_out: float,
    valid: bool,
    status: str,
    meta: dict,
    return_info: bool,
    extra: dict | None = None,
) -> float | dict:
    if not return_info:
        return float(d_out)
    out = {
        "d_hat_m": float(d_out),
        "valid_estimate": bool(valid and math.isfinite(d_out)),
        "input_status": status,
        "n_rows_input": meta.get("n_rows_input", 0),
        "n_rows_admissible": meta.get("n_rows_admissible", 0),
        "n_rows_discarded": meta.get("n_rows_discarded", 0),
        "n_unique_usable_tones": meta.get("n_unique_usable_tones", 0),
    }
    if extra:
        out.update(extra)
    return out


def baseline_ifft_argmax(
    grid_k: np.ndarray,
    ant_idx: np.ndarray,
    y: np.ndarray,
    q_i: np.ndarray,
    q_r: np.ndarray,
    n_paths: int = 1,
    d_cal_m: float = 0.0,
    n_fft: int = 4096,
    d_max_m: float = 45.0,
    step_idx: np.ndarray | None = None,
    apply_stage1_drift: bool = False,
    subevent_idx: np.ndarray | None = None,
    return_info: bool = False,
) -> float | dict:
    """Coherent mean per channel -> zero-padded IFFT peak argmax."""
    ok, stat, meta, y_grid, mask, _, _ = _prepare_grid_with_meta(
        grid_k, ant_idx, y, q_i, q_r,
        n_paths=n_paths, step_idx=step_idx, apply_stage1_drift=apply_stage1_drift, subevent_idx=subevent_idx,
    )
    if not ok or (not np.any(mask)):
        return _wrap_baseline_result(float("nan"), False, stat, meta, return_info)
    cir = np.sum(np.abs(np.fft.ifft(y_grid, n=n_fft, axis=0)) ** 2, axis=1)
    max_bin = min(n_fft - 2, int(math.ceil(2.0 * d_max_m * DF_HZ * n_fft / C_LIGHT)))
    pk = int(np.argmax(cir[: max_bin + 1]))
    d_raw = pk * C_LIGHT / (2.0 * n_fft * DF_HZ)
    return _wrap_baseline_result(max(0.0, float(d_raw - d_cal_m)), True, "valid", meta, return_info)


def baseline_ifft_leading_edge(
    grid_k: np.ndarray,
    ant_idx: np.ndarray,
    y: np.ndarray,
    q_i: np.ndarray,
    q_r: np.ndarray,
    n_paths: int = 1,
    d_cal_m: float = 0.0,
    n_fft: int = 4096,
    alpha_thresh: float = 0.40,
    d_max_m: float = 45.0,
    step_idx: np.ndarray | None = None,
    apply_stage1_drift: bool = False,
    subevent_idx: np.ndarray | None = None,
    return_info: bool = False,
) -> float | dict:
    """Coherent mean per channel -> IFFT leading-edge first peak above alpha * max."""
    ok, stat, meta, y_grid, mask, _, _ = _prepare_grid_with_meta(
        grid_k, ant_idx, y, q_i, q_r,
        n_paths=n_paths, step_idx=step_idx, apply_stage1_drift=apply_stage1_drift, subevent_idx=subevent_idx,
    )
    if not ok or (not np.any(mask)):
        return _wrap_baseline_result(float("nan"), False, stat, meta, return_info)
    cir = np.sqrt(np.sum(np.abs(np.fft.ifft(y_grid, n=n_fft, axis=0)) ** 2, axis=1))
    max_bin = min(n_fft - 2, int(math.ceil(2.0 * d_max_m * DF_HZ * n_fft / C_LIGHT)))
    thresh = alpha_thresh * float(np.max(cir[: max_bin + 1]))
    for i in range(max_bin):
        left_ok = (i == 0) or (cir[i] >= cir[i - 1])
        right_ok = cir[i] >= cir[i + 1]
        if left_ok and right_ok and cir[i] >= thresh:
            d_raw = i * C_LIGHT / (2.0 * n_fft * DF_HZ)
            return _wrap_baseline_result(max(0.0, float(d_raw - d_cal_m)), True, "valid", meta, return_info)
    pk = int(np.argmax(cir[: max_bin + 1]))
    return _wrap_baseline_result(max(0.0, float(pk * C_LIGHT / (2.0 * n_fft * DF_HZ) - d_cal_m)), True, "valid", meta, return_info)


def baseline_kim_k15(
    grid_k: np.ndarray,
    ant_idx: np.ndarray,
    y: np.ndarray,
    q_i: np.ndarray,
    q_r: np.ndarray,
    n_paths: int = 1,
    d_cal_m: float = 0.0,
    k_target: int = 15,
    d_max_m: float = 45.0,
    p_radius: float = 0.5,
    alpha_thresh: float = 0.40,
    n_fft: int = 4096,
    step_idx: np.ndarray | None = None,
    apply_stage1_drift: bool = False,
    subevent_idx: np.ndarray | None = None,
    return_info: bool = False,
) -> float | dict:
    """Adapted Kim & Tripathi (arXiv:2611.23471v1) Alg. 1 K=15 tone selection on observed mask."""
    ok, stat, meta, y_grid, mask, _, _ = _prepare_grid_with_meta(
        grid_k, ant_idx, y, q_i, q_r,
        n_paths=n_paths, step_idx=step_idx, apply_stage1_drift=apply_stage1_drift, subevent_idx=subevent_idx,
    )
    if not ok or (not np.any(mask)):
        return _wrap_baseline_result(float("nan"), False, stat, meta, return_info)
    l0 = sorted(int(k + 2) for k in np.flatnonzero(np.any(mask, axis=1)))
    if len(l0) < 2:
        return _wrap_baseline_result(float("nan"), False, "insufficient_tones", meta, return_info)

    k_sel = min(k_target, len(l0))
    g_star = max(1, int(math.floor(149.896 / d_max_m)) - 1)

    # Step 1: Partition free channels L0 into maximal groups with gaps <= g_star
    groups: list[list[int]] = [[l0[0]]]
    for ch in l0[1:]:
        if ch - groups[-1][-1] <= g_star:
            groups[-1].append(ch)
        else:
            groups.append([ch])

    big_groups = [g for g in groups if len(g) >= k_sel]
    if big_groups:
        a_set = set(max(big_groups, key=lambda g: g[-1] - g[0]))
        low, high = min(a_set), max(a_set)
    else:
        a_set = set(l0)
        low, high = l0[0], l0[-1]

    # Step 2-3: Uniform candidate targets + p*sigma free-tone selection
    if k_sel == 1:
        selected = [l0[0]]
    else:
        sigma = (high - low) / (k_sel - 1)
        selected_set: set[int] = set()
        l0_set = set(l0)
        for t_idx in range(k_sel):
            tau_pos = low + t_idx * sigma
            c_avail = a_set - selected_set
            if not c_avail:
                c_avail = l0_set - selected_set
            c0 = [a for a in c_avail if a in l0_set and abs(a - tau_pos) <= p_radius * sigma]
            if c0:
                s = min(c0, key=lambda a: (abs(a - tau_pos), a))
            else:
                s = min(c_avail, key=lambda a: (abs(a - tau_pos), a))
            selected_set.add(s)
        selected = sorted(selected_set)

    y_sub = np.zeros_like(y_grid)
    sel_grid = np.asarray([ch - 2 for ch in selected], dtype=int)
    y_sub[sel_grid, :] = y_grid[sel_grid, :]

    cir = np.sqrt(np.sum(np.abs(np.fft.ifft(y_sub, n=n_fft, axis=0)) ** 2, axis=1))
    max_bin = min(n_fft - 2, int(math.ceil(2.0 * d_max_m * DF_HZ * n_fft / C_LIGHT)))
    thresh = alpha_thresh * float(np.max(cir[: max_bin + 1]))
    for i in range(max_bin):
        left_ok = (i == 0) or (cir[i] >= cir[i - 1])
        right_ok = cir[i] >= cir[i + 1]
        if left_ok and right_ok and cir[i] >= thresh:
            return _wrap_baseline_result(max(0.0, float(i * C_LIGHT / (2.0 * n_fft * DF_HZ) - d_cal_m)), True, "valid", meta, return_info)
    pk = int(np.argmax(cir[: max_bin + 1]))
    return _wrap_baseline_result(max(0.0, float(pk * C_LIGHT / (2.0 * n_fft * DF_HZ) - d_cal_m)), True, "valid", meta, return_info)


def _unwrap_one_way_continuity(y_vals: np.ndarray) -> np.ndarray:
    """Recover one-way sqrt(y) along ordered sequence by continuity phase tracking."""
    roots = np.sqrt(y_vals.astype(np.complex128))
    if len(roots) <= 1:
        return roots
    out = roots.copy()
    for i in range(1, len(out)):
        if abs(out[i] - out[i - 1]) > abs(-out[i] - out[i - 1]):
            out[i] = -out[i]
    return out


def _hankel_fbss_signal_subspace(h_vec: np.ndarray, l_sub: int) -> tuple[np.ndarray, int]:
    """Form Hankel matrix -> Forward-Backward covariance -> eigenvalue-ratio signal subspace."""
    m = len(h_vec)
    n_col = m - l_sub + 1
    h_mat = np.empty((l_sub, n_col), dtype=np.complex128)
    for r in range(l_sub):
        h_mat[r, :] = h_vec[r : r + n_col]
    r_f = (h_mat @ h_mat.conj().T) / max(n_col, 1)
    r_fb = 0.5 * (r_f + np.conj(r_f[::-1, ::-1]))
    r_fb = 0.5 * (r_fb + r_fb.conj().T)

    evals, evecs = np.linalg.eigh(r_fb)
    idx_desc = np.argsort(evals)[::-1]
    evals = np.maximum(evals[idx_desc], 0.0)
    evecs = evecs[:, idx_desc]

    # Eigenvalue-ratio threshold (with minimum rank 1 and max rank l_sub // 2)
    r_sig = 1
    if evals[0] > 0:
        r_sig = int(np.sum(evals > 0.02 * evals[0]))
    r_sig = max(1, min(r_sig, max(1, l_sub // 2 - 1)))
    return evecs[:, :r_sig], r_sig


def _pick_first_music_peak(taus_grid: np.ndarray, spec: np.ndarray, rel_thresh: float = 0.20) -> float:
    """Select first significant local maximum of MUSIC pseudospectrum."""
    s_max = float(np.max(spec))
    thresh = rel_thresh * s_max
    for i in range(1, len(spec) - 1):
        if spec[i] > spec[i - 1] and spec[i] >= spec[i + 1] and spec[i] >= thresh:
            return float(C_LIGHT * taus_grid[i])
    return float(C_LIGHT * taus_grid[int(np.argmax(spec))])


def _max_eigenvalue_ratio_rank(evals: np.ndarray, max_r: int) -> int:
    """Select signal subspace rank via maximum consecutive eigenvalue ratio (Wieme IPIN 2026)."""
    upper = min(max_r, len(evals) - 1)
    if upper < 1 or evals[0] <= 0.0:
        return 1
    ratios = evals[:upper] / np.maximum(evals[1 : upper + 1], 1e-10 * evals[0])
    # Only consider eigenvalues above noise floor gate 1e-4 * evals[0]
    ratios = np.where(evals[:upper] >= 1e-4 * evals[0], ratios, 0.0)
    return max(1, int(np.argmax(ratios)) + 1)


def baseline_fbss_music(
    grid_k: np.ndarray,
    ant_idx: np.ndarray,
    y: np.ndarray,
    q_i: np.ndarray,
    q_r: np.ndarray,
    n_paths: int = 1,
    d_cal_m: float = 0.0,
    d_max_m: float = 45.0,
    first_only: bool = True,
    step_idx: np.ndarray | None = None,
    apply_stage1_drift: bool = False,
    subevent_idx: np.ndarray | None = None,
    return_info: bool = False,
) -> float | dict:
    """Adapted Wieme et al. (IPIN 2026) two-way FBSS-MUSIC (L=36, max-eig-ratio, d=c*tau_2w/2)."""
    ok, stat, meta, y_grid, mask, _, _ = _prepare_grid_with_meta(
        grid_k, ant_idx, y, q_i, q_r,
        n_paths=n_paths, first_only=first_only, step_idx=step_idx,
        apply_stage1_drift=apply_stage1_drift, subevent_idx=subevent_idx,
    )
    if not ok or (not np.any(mask)):
        return _wrap_baseline_result(float("nan"), False, stat, meta, return_info)
    l_h = 36
    n_col = N_GRID - l_h + 1
    r_avg = np.zeros((l_h, l_h), dtype=np.complex128)
    valid_p = 0

    for p in range(n_paths):
        obs_k = np.flatnonzero(mask[:, p])
        if len(obs_k) < 4:
            continue
        # Direct two-way CFR sequence x_k on N_GRID (missing holes zero-filled when no NN bank)
        x_2way = np.zeros(N_GRID, dtype=np.complex128)
        x_2way[obs_k] = y_grid[obs_k, p]
        h_mat = np.empty((l_h, n_col), dtype=np.complex128)
        for r in range(l_h):
            h_mat[r, :] = x_2way[r : r + n_col]
        r_f = (h_mat @ h_mat.conj().T) / max(n_col, 1)
        r_fb = 0.5 * (r_f + np.conj(r_f[::-1, ::-1]))
        r_avg += r_fb
        valid_p += 1

    if valid_p == 0:
        return _wrap_baseline_result(float("nan"), False, "insufficient_tones", meta, return_info)
    r_avg = 0.5 * (r_avg + r_avg.conj().T) / valid_p
    evals, evecs = np.linalg.eigh(r_avg)
    idx_desc = np.argsort(evals)[::-1]
    evals = np.maximum(evals[idx_desc], 0.0)
    evecs = evecs[:, idx_desc]

    r_sig = _max_eigenvalue_ratio_rank(evals, max_r=min(8, max(1, l_h // 2 - 1)))
    u_s = evecs[:, :r_sig]

    # Two-way delay tau_2way = 2*d/c in [0, 2*d_max_m/c]; d = c * tau_2way / 2
    n_grid_pts = max(1500, int(math.ceil(1500.0 * d_max_m / 45.0)))
    taus_2way_grid = np.linspace(0.0, 2.0 * d_max_m / C_LIGHT, n_grid_pts)
    k_sub = np.arange(l_h, dtype=np.float64)[:, None]
    e_steer = np.exp(-1j * 2.0 * np.pi * DF_HZ * k_sub * taus_2way_grid[None, :])
    proj = u_s.conj().T @ e_steer
    denom = np.maximum(l_h - np.sum(np.abs(proj) ** 2, axis=0), 1e-10)
    spec = 1.0 / denom
    d_2way = _pick_first_music_peak(taus_2way_grid, spec, rel_thresh=0.20)
    d_raw = 0.5 * d_2way
    return _wrap_baseline_result(max(0.0, d_raw - d_cal_m), True, "valid", meta, return_info)


def baseline_sheikh_wps(
    grid_k: np.ndarray,
    ant_idx: np.ndarray,
    y: np.ndarray,
    q_i: np.ndarray,
    q_r: np.ndarray,
    n_paths: int = 1,
    d_cal_m: float = 0.0,
    d_max_m: float = 45.0,
    step_idx: np.ndarray | None = None,
    apply_stage1_drift: bool = False,
    subevent_idx: np.ndarray | None = None,
    return_info: bool = False,
) -> float | dict:
    """Sheikh et al. (IEEE IoT J. 2023) Eq. (10) L_j-Weighted-Average Pseudospectrum (WPS)."""
    ok, stat, meta, y_grid, mask, _, _ = _prepare_grid_with_meta(
        grid_k, ant_idx, y, q_i, q_r,
        n_paths=n_paths, step_idx=step_idx, apply_stage1_drift=apply_stage1_drift, subevent_idx=subevent_idx,
    )
    if not ok or (not np.any(mask)):
        return _wrap_baseline_result(float("nan"), False, stat, meta, return_info)
    n_grid_pts = max(1500, int(math.ceil(1500.0 * d_max_m / 45.0)))
    taus_grid = np.linspace(0.0, d_max_m / C_LIGHT, n_grid_pts)

    total_denom = np.zeros_like(taus_grid)
    total_bands = 0

    for p in range(n_paths):
        obs_k = np.flatnonzero(mask[:, p])
        if len(obs_k) < 4:
            continue
        # Partition into maximal contiguous available subbands B_j
        bands: list[list[int]] = [[int(obs_k[0])]]
        for k in obs_k[1:]:
            if int(k) == bands[-1][-1] + 1:
                bands[-1].append(int(k))
            else:
                bands.append([int(k)])

        for band in bands:
            if len(band) < 4:
                continue
            b_arr = np.asarray(band, dtype=int)
            h_band = _unwrap_one_way_continuity(y_grid[b_arr, p])
            l_j = len(b_arr) // 2 + 1
            u_sj, _ = _hankel_fbss_signal_subspace(h_band, l_j)
            k_sub = np.arange(l_j, dtype=np.float64)[:, None]
            e_steer = np.exp(-1j * 2.0 * np.pi * DF_HZ * k_sub * taus_grid[None, :])
            proj = u_sj.conj().T @ e_steer
            orth = np.maximum(l_j - np.sum(np.abs(proj) ** 2, axis=0), 1e-10)
            # Exact Sheikh Eq. (10) smoothing-factor weight L_j
            total_denom += float(l_j) * orth
            total_bands += 1

    if total_bands == 0:
        # Prespecified native WPS failure when no contiguous >=4-tone subband exists (no silent IFFT fallback)
        return _wrap_baseline_result(float("nan"), False, "no_contiguous_subband", meta, return_info)

    spec = 1.0 / np.maximum(total_denom, 1e-10)
    d_raw = _pick_first_music_peak(taus_grid, spec, rel_thresh=0.20)
    return _wrap_baseline_result(max(0.0, d_raw - d_cal_m), True, "valid", meta, return_info)


def baseline_sheikh_anm_music(
    grid_k: np.ndarray,
    ant_idx: np.ndarray,
    y: np.ndarray,
    q_i: np.ndarray,
    q_r: np.ndarray,
    n_paths: int = 1,
    d_cal_m: float = 0.0,
    d_max_m: float = 45.0,
    step_idx: np.ndarray | None = None,
    apply_stage1_drift: bool = False,
    subevent_idx: np.ndarray | None = None,
    admm_max_iter: int | None = None,
    admm_tol: float = 1.5e-3,
    return_info: bool = False,
) -> float | dict:
    """Adapted Sheikh (IoTJ 2023 Sec. III-C) h^2-ANM completion + sqrt-unwrap + FBSS-MUSIC."""
    ok, stat, meta, y_grid, mask, w_obs, sigma2_hat = _prepare_grid_with_meta(
        grid_k, ant_idx, y, q_i, q_r,
        n_paths=n_paths, step_idx=step_idx, apply_stage1_drift=apply_stage1_drift, subevent_idx=subevent_idx,
    )
    if (not ok) or (not np.any(mask)) or (not np.any(w_obs > 0)):
        return _wrap_baseline_result(
            float("nan"), False, stat, meta, return_info,
            {"admm_converged": False, "admm_primal_rel": float("nan"), "admm_dual_rel": float("nan"), "n_admm_iter": 0},
        )
    st2 = stage2_mmv_anm_admm(y_grid, w_obs, sigma2_hat, max_iter=admm_max_iter, tol=admm_tol)
    x_completed = st2["x_hat"]

    l_h = N_GRID // 2 + 1
    n_col = N_GRID - l_h + 1
    r_avg = np.zeros((l_h, l_h), dtype=np.complex128)
    valid_p = 0
    for p in range(n_paths):
        h_1way = _unwrap_one_way_continuity(x_completed[:, p])
        h_mat = np.empty((l_h, n_col), dtype=np.complex128)
        for r in range(l_h):
            h_mat[r, :] = h_1way[r : r + n_col]
        r_f = (h_mat @ h_mat.conj().T) / max(n_col, 1)
        r_fb = 0.5 * (r_f + np.conj(r_f[::-1, ::-1]))
        r_avg += r_fb
        valid_p += 1

    if valid_p == 0:
        return _wrap_baseline_result(
            float("nan"), False, "no_valid_paths", meta, return_info,
            {
                "admm_converged": bool(st2["admm_converged"]),
                "admm_primal_rel": float(st2["admm_primal_rel"]),
                "admm_dual_rel": float(st2["admm_dual_rel"]),
                "n_admm_iter": int(st2["n_iter"]),
            },
        )
    r_avg = 0.5 * (r_avg + r_avg.conj().T) / valid_p
    evals, evecs = np.linalg.eigh(r_avg)
    idx_desc = np.argsort(evals)[::-1]
    evals = np.maximum(evals[idx_desc], 0.0)
    evecs = evecs[:, idx_desc]

    r_sig = max(1, min(int(np.sum(evals > 0.02 * max(evals[0], 1e-12))), 8))
    u_s = evecs[:, :r_sig]

    n_grid_pts = max(1500, int(math.ceil(1500.0 * d_max_m / 45.0)))
    taus_grid = np.linspace(0.0, d_max_m / C_LIGHT, n_grid_pts)
    k_sub = np.arange(l_h, dtype=np.float64)[:, None]
    e_steer = np.exp(-1j * 2.0 * np.pi * DF_HZ * k_sub * taus_grid[None, :])
    proj = u_s.conj().T @ e_steer
    denom = np.maximum(l_h - np.sum(np.abs(proj) ** 2, axis=0), 1e-10)
    spec = 1.0 / denom
    d_raw = _pick_first_music_peak(taus_grid, spec, rel_thresh=0.20)
    d_out = max(0.0, d_raw - d_cal_m)
    if return_info:
        return _wrap_baseline_result(
            d_out, True, "valid", meta, True,
            {
                "admm_converged": bool(st2["admm_converged"]),
                "admm_primal_rel": float(st2["admm_primal_rel"]),
                "admm_dual_rel": float(st2["admm_dual_rel"]),
                "n_admm_iter": int(st2["n_iter"]),
            },
        )
    return d_out
