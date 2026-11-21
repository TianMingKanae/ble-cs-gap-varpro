"""Prospective confirmatory statistical analysis & failure accounting for BLE CS ranging."""

from __future__ import annotations

import math

import numpy as np


_INVALID_LABEL_TOKENS = frozenset({"nan", "none", "nat", "null", "inf", "+inf", "-inf"})


def _validate_and_partition_clusters(cluster_ids: object, n_total: int) -> tuple[int, list[np.ndarray]]:
    """Validate cluster labels (preserving Python element types before NumPy coercion) and exact partition."""
    if isinstance(cluster_ids, np.ndarray):
        if cluster_ids.ndim != 1 or len(cluster_ids) != n_total:
            raise ValueError(f"cluster_ids must be 1D with length {n_total}")
        if cluster_ids.dtype == bool:
            raise ValueError("cluster_ids cannot be boolean array")
        raw_items = cluster_ids.tolist()
    else:
        try:
            raw_items = list(cluster_ids)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("cluster_ids must be a 1D sequence") from exc
        if len(raw_items) != n_total:
            raise ValueError(f"cluster_ids must have length {n_total}, got {len(raw_items)}")

    by_label: dict[tuple[str, object], list[int]] = {}
    for row_idx, raw_item in enumerate(raw_items):
        if raw_item is None or isinstance(raw_item, (bool, np.bool_)):
            raise ValueError(f"Invalid cluster label at row {row_idx}: {raw_item!r}")
        if isinstance(raw_item, (float, np.floating)):
            val = float(raw_item)
            if not math.isfinite(val):
                raise ValueError(f"Non-finite cluster label at row {row_idx}: {raw_item!r}")
            key = ("num", int(val) if val.is_integer() else val)
        elif isinstance(raw_item, (int, np.integer)):
            key = ("num", int(raw_item))
        elif isinstance(raw_item, (complex, np.complexfloating)):
            raise ValueError(f"Complex cluster label not permitted at row {row_idx}: {raw_item!r}")
        elif isinstance(raw_item, (bytes, bytearray)):
            s = bytes(raw_item).decode("utf-8", errors="strict").strip()
            if (not s) or (s.lower() in _INVALID_LABEL_TOKENS):
                raise ValueError(f"Missing/blank cluster label at row {row_idx}: {raw_item!r}")
            key = ("str", s)
        elif isinstance(raw_item, (str, np.str_)):
            s = str(raw_item).strip()
            if (not s) or (s.lower() in _INVALID_LABEL_TOKENS):
                raise ValueError(f"Missing/blank cluster label at row {row_idx}: {raw_item!r}")
            key = ("str", s)
        else:
            raise ValueError(f"Unsupported cluster label type at row {row_idx}: {type(raw_item).__name__}")
        by_label.setdefault(key, []).append(row_idx)

    if not by_label:
        raise ValueError("cluster_ids produced zero clusters")

    sorted_keys = sorted(by_label.keys(), key=lambda kv: (kv[0], str(kv[1])) if kv[0] != "num" else ("num", kv[1]))
    cluster_rows = [np.asarray(by_label[k], dtype=int) for k in sorted_keys]
    all_idx = np.concatenate(cluster_rows)
    if len(all_idx) != n_total or not np.array_equal(np.sort(all_idx), np.arange(n_total)):
        raise ValueError("cluster_ids does not form an exact partition of input rows")
    return len(cluster_rows), cluster_rows


def holm_bonferroni(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Exact Holm step-down adjusted p-values for a family of k tests."""
    p = np.asarray(p_values, dtype=np.float64)
    k = len(p)
    order = np.argsort(p)
    adj_sorted = np.empty(k, dtype=np.float64)
    running_max = 0.0
    for r, idx in enumerate(order):
        val = (k - r) * float(p[idx])
        running_max = max(running_max, val)
        adj_sorted[r] = min(1.0, running_max)
    adj = np.empty(k, dtype=np.float64)
    adj[order] = adj_sorted
    return adj


def _apply_failure_penalty(
    abs_err: np.ndarray,
    valid_mask: np.ndarray,
    penalty_m: np.ndarray,
) -> np.ndarray:
    """Impute penalty_m (cohort d_max_m) where valid_mask is False or abs_err is non-finite."""
    ok = np.asarray(valid_mask, dtype=bool) & np.isfinite(abs_err)
    return np.where(ok, abs_err, penalty_m).astype(np.float64)


def run_confirmatory_comparison(
    sdr_abs_err: np.ndarray,
    sdr_valid: np.ndarray,
    comp_abs_errs: dict[str, np.ndarray],
    comp_valids: dict[str, np.ndarray],
    cluster_ids: np.ndarray,
    penalty_m: np.ndarray | float = 45.0,
    comparators: tuple[str, ...] = ("IFFT-LE", "Sheikh-WPS"),
    n_boot: int = 2000,
    seed: int = 20261120,
    alpha: float = 0.05,
) -> dict:
    """Evaluate co-primary MAE & marginal-P90 contrasts via cluster bootstrap + Holm correction.

    Metrics:
      - Primary analysis: failure-penalized paired sample across all procedures in cohort
        (invalid/non-finite/exception/timeout replaced by cohort d_max_m).
      - Secondary analysis: common-pair valid-only comparisons on jointly valid procedures.
      - Contrast 1: Delta_MAE = mean(|e_SDR|) - mean(|e_comp|) (< 0 favors SDR-ANM-QML).
      - Contrast 2: Delta_P90 = Q_0.90(|e_SDR|) - Q_0.90(|e_comp|) (< 0 favors SDR-ANM-QML),
        using numpy.quantile(..., method='linear') on the marginal distributions.
      - Claim rule per comparator: superiority claimed iff BOTH Delta_MAE and Delta_P90 are < 0
        with two-sided Holm-adjusted bootstrap p < alpha on the primary failure-penalized analysis.
    """
    sdr_err = np.asarray(sdr_abs_err, dtype=np.float64)
    sdr_ok_raw = np.asarray(sdr_valid, dtype=bool)
    if sdr_err.ndim != 1 or len(sdr_err) == 0:
        raise ValueError("sdr_abs_err must be a non-empty 1D array")
    n_total = len(sdr_err)
    if sdr_ok_raw.ndim != 1 or len(sdr_ok_raw) != n_total:
        raise ValueError(f"sdr_valid must be 1D with length {n_total}")
    n_clusters, cluster_rows = _validate_and_partition_clusters(cluster_ids, n_total)
    sdr_ok = sdr_ok_raw & np.isfinite(sdr_err)
    if np.ndim(penalty_m) == 0:
        pen_scalar = float(penalty_m)
        if not np.isfinite(pen_scalar) or pen_scalar <= 0.0:
            raise ValueError("penalty_m must be finite and > 0")
        pen_arr = np.full(n_total, pen_scalar, dtype=np.float64)
    else:
        pen_arr = np.asarray(penalty_m, dtype=np.float64)
        if pen_arr.ndim != 1 or len(pen_arr) != n_total or not np.all(np.isfinite(pen_arr)) or np.any(pen_arr <= 0.0):
            raise ValueError(f"penalty_m must be 1D finite > 0 with length {n_total}")
    for cname in comparators:
        if cname not in comp_abs_errs or cname not in comp_valids:
            raise ValueError(f"Missing comparator arrays for {cname!r}")
        c_err = np.asarray(comp_abs_errs[cname], dtype=np.float64)
        c_val = np.asarray(comp_valids[cname], dtype=bool)
        if c_err.ndim != 1 or len(c_err) != n_total:
            raise ValueError(f"comp_abs_errs[{cname!r}] must be 1D with length {n_total}")
        if c_val.ndim != 1 or len(c_val) != n_total:
            raise ValueError(f"comp_valids[{cname!r}] must be 1D with length {n_total}")

    sdr_pen = _apply_failure_penalty(sdr_err, sdr_ok, pen_arr)
    comp_pen = {
        cname: _apply_failure_penalty(
            np.asarray(comp_abs_errs[cname], dtype=np.float64),
            np.asarray(comp_valids[cname], dtype=bool),
            pen_arr,
        )
        for cname in comparators
    }

    # Pre-index per-cluster arrays so resampling B=2000 clusters is vectorized/fast
    sdr_c = [sdr_pen[idx] for idx in cluster_rows]
    comp_c = {cname: [comp_pen[cname][idx] for idx in cluster_rows] for cname in comparators}

    rng = np.random.default_rng(seed)
    boot_deltas: dict[tuple[str, str], np.ndarray] = {
        (cname, ep): np.empty(n_boot, dtype=np.float64)
        for cname in comparators
        for ep in ("MAE", "P90")
    }
    for b in range(n_boot):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        sdr_b = np.concatenate([sdr_c[i] for i in draw])
        sdr_mae_b = float(np.mean(sdr_b))
        sdr_p90_b = float(np.quantile(sdr_b, 0.90, method="linear"))
        for cname in comparators:
            cb = np.concatenate([comp_c[cname][i] for i in draw])
            boot_deltas[(cname, "MAE")][b] = sdr_mae_b - float(np.mean(cb))
            boot_deltas[(cname, "P90")][b] = sdr_p90_b - float(np.quantile(cb, 0.90, method="linear"))

    n_tests = 2 * len(comparators)
    q_lo = 0.5 * (alpha / max(n_tests, 1))
    q_hi = 1.0 - q_lo

    tests: list[dict] = []
    raw_pvals: list[float] = []
    for cname in comparators:
        cp = comp_pen[cname]
        point_diffs = {
            "MAE": float(np.mean(sdr_pen)) - float(np.mean(cp)),
            "P90": float(np.quantile(sdr_pen, 0.90, method="linear"))
            - float(np.quantile(cp, 0.90, method="linear")),
        }
        # Equal-cluster-weighted sensitivity MAE (sensitivity check alongside procedure-weighted primary)
        cl_diff_mae = float(np.mean([np.mean(sdr_c[i]) - np.mean(comp_c[cname][i]) for i in range(n_clusters)]))
        for ep in ("MAE", "P90"):
            bd = boot_deltas[(cname, ep)]
            n_le = int(np.sum(bd <= 0.0))
            n_ge = int(np.sum(bd >= 0.0))
            p_raw = min(1.0, float(1 + 2 * min(n_le, n_ge)) / float(n_boot + 1))
            ci = (
                float(np.quantile(bd, q_lo, method="linear")),
                float(np.quantile(bd, q_hi, method="linear")),
            )
            raw_pvals.append(p_raw)
            tests.append(
                {
                    "comparator": cname,
                    "endpoint": ep,
                    "delta_point_m": point_diffs[ep],
                    "delta_cluster_equal_mae_m": cl_diff_mae if ep == "MAE" else None,
                    "p_raw": p_raw,
                    "simultaneous_ci_m": ci,
                    "simultaneous_ci_level": 1.0 - alpha / max(n_tests, 1),
                }
            )

    p_holm = holm_bonferroni(raw_pvals)
    for t_entry, ph in zip(tests, p_holm):
        t_entry["p_holm"] = float(ph)
        t_entry["pass_holm"] = bool(ph < alpha and t_entry["delta_point_m"] < 0.0)

    claims: dict[str, bool] = {}
    secondary_valid_only: dict[str, dict] = {}
    for cname in comparators:
        c_tests = [t for t in tests if t["comparator"] == cname]
        claims[cname] = all(t["pass_holm"] for t in c_tests)
        c_ok = np.asarray(comp_valids[cname], dtype=bool) & np.isfinite(comp_abs_errs[cname])
        joint = sdr_ok & c_ok
        if np.any(joint):
            s_j = sdr_err[joint]
            c_j = np.asarray(comp_abs_errs[cname], dtype=np.float64)[joint]
            secondary_valid_only[cname] = {
                "n_common_valid_pairs": int(np.sum(joint)),
                "n_total_pairs": n_total,
                "sdr_valid_pct": float(100.0 * np.mean(sdr_ok)),
                "comp_valid_pct": float(100.0 * np.mean(c_ok)),
                "delta_mae_valid_only_m": float(np.mean(s_j) - np.mean(c_j)),
                "delta_p90_valid_only_m": float(
                    np.quantile(s_j, 0.90, method="linear") - np.quantile(c_j, 0.90, method="linear")
                ),
            }
        else:
            secondary_valid_only[cname] = {
                "n_common_valid_pairs": 0,
                "n_total_pairs": n_total,
                "sdr_valid_pct": float(100.0 * np.mean(sdr_ok)),
                "comp_valid_pct": float(100.0 * np.mean(c_ok)),
                "delta_mae_valid_only_m": float("nan"),
                "delta_p90_valid_only_m": float("nan"),
            }

    return {
        "n_procedures": n_total,
        "n_clusters": n_clusters,
        "n_boot": n_boot,
        "alpha": alpha,
        "sdr_valid_count": int(np.sum(sdr_ok)),
        "sdr_invalid_count": int(n_total - np.sum(sdr_ok)),
        "tests": tests,
        "superiority_claimed": claims,
        "secondary_valid_only": secondary_valid_only,
    }


def summarize_nlos_cohort(
    signed_errs_by_method: dict[str, np.ndarray],
    valid_by_method: dict[str, np.ndarray],
    penalty_m: float | np.ndarray = 45.0,
) -> dict[str, dict[str, float | int]]:
    """Compute contract Sec. 4 NLOS signed excess-delay & outlier metrics per method."""
    if not signed_errs_by_method:
        raise ValueError("signed_errs_by_method must be non-empty")
    first_key = next(iter(signed_errs_by_method))
    n_total = len(np.asarray(signed_errs_by_method[first_key]))
    if np.ndim(penalty_m) == 0:
        pen_scalar = float(penalty_m)
        if not math.isfinite(pen_scalar) or pen_scalar <= 0.0:
            raise ValueError("penalty_m must be finite > 0")
        pen_arr = np.full(n_total, pen_scalar, dtype=np.float64)
    else:
        pen_arr = np.asarray(penalty_m, dtype=np.float64)
        if pen_arr.ndim != 1 or len(pen_arr) != n_total or not np.all(np.isfinite(pen_arr)) or np.any(pen_arr <= 0.0):
            raise ValueError(f"penalty_m must be 1D finite > 0 with length {n_total}")

    out: dict[str, dict[str, float | int]] = {}
    for mname, s_raw in signed_errs_by_method.items():
        s_arr = np.asarray(s_raw, dtype=np.float64)
        v_raw = np.asarray(valid_by_method[mname], dtype=bool)
        if s_arr.ndim != 1 or len(s_arr) != n_total or v_raw.ndim != 1 or len(v_raw) != n_total:
            raise ValueError(f"NLOS arrays for {mname!r} must be 1D with length {n_total}")
        ok = v_raw & np.isfinite(s_arr)
        abs_pen = _apply_failure_penalty(np.abs(s_arr), ok, pen_arr)
        signed_valid = s_arr[ok]
        out[mname] = {
            "n_total": n_total,
            "n_valid": int(np.sum(ok)),
            "valid_pct": float(100.0 * np.mean(ok)),
            "signed_bias_median_m": float(np.median(signed_valid)) if np.any(ok) else float("nan"),
            "signed_bias_mean_m": float(np.mean(signed_valid)) if np.any(ok) else float("nan"),
            "medae_m": float(np.median(abs_pen)),
            "mae_m": float(np.mean(abs_pen)),
            "p90_m": float(np.quantile(abs_pen, 0.90, method="linear")),
            "gross_outlier_gt_5m_pct": float(100.0 * np.mean(abs_pen > 5.0)),
        }
    return out


def _dry_run() -> None:
    """Exercise unequal clusters, missing/NaN failures, and opposite MAE vs P90 directions."""
    rng = np.random.default_rng(12345)
    cids = np.concatenate([np.full(sz, f"G{i:02d}") for i, sz in enumerate([3, 15, 7, 15, 2, 15, 10, 15, 5, 15, 15, 15])])
    n = len(cids)

    # Comparator A: IFFT-LE -> SDR strictly better on BOTH MAE and P90 (even with 2 NaN failures)
    err_le = rng.uniform(0.8, 3.2, size=n)
    err_sdr = rng.uniform(0.05, 0.5, size=n)
    ok_sdr = np.ones(n, dtype=bool)
    ok_le = np.ones(n, dtype=bool)
    err_sdr[[4, 29]] = np.nan
    ok_sdr[[4, 29]] = False  # penalized to 45.0 m in primary analysis

    # Comparator B: Sheikh-WPS -> opposite MAE vs P90 directions (better MAE, worse P90)
    err_wps = np.full(n, 0.9, dtype=np.float64)
    err_wps[:12] = 1.0  # lower tail at P90 than penalized SDR tails (2*45m -> ~1% vs P90 on 132)
    # Force opposite direction: low P90 for WPS (~1.0 m) while SDR has clean MAE<WPS but heavy tail (>3.5 m on 15%)
    err_wps = rng.uniform(0.5, 0.9, size=n)
    err_sdr_b = rng.uniform(0.05, 0.25, size=n)
    tail_idx = rng.choice(n, size=max(1, int(0.18 * n)), replace=False)
    err_sdr_b[tail_idx] = 2.0  # MAE_SDR ~ 0.45 < MAE_WPS ~ 0.70, but P90_SDR = 2.0 > P90_WPS ~ 0.86

    res1 = run_confirmatory_comparison(
        err_sdr, ok_sdr, {"IFFT-LE": err_le, "Sheikh-WPS": err_wps},
        {"IFFT-LE": ok_le, "Sheikh-WPS": np.ones(n, dtype=bool)}, cids, n_boot=500, seed=1,
    )
    res2 = run_confirmatory_comparison(
        err_sdr_b, np.ones(n, dtype=bool), {"IFFT-LE": err_le, "Sheikh-WPS": err_wps},
        {"IFFT-LE": ok_le, "Sheikh-WPS": np.ones(n, dtype=bool)}, cids, n_boot=500, seed=1,
    )
    print("Dry-run 1 (2 penalized SDR failures out of 132):", flush=True)
    for t in res1["tests"]:
        print(" ", t["comparator"], t["endpoint"], "pt=", round(t["delta_point_m"], 3), "p_holm=", round(t["p_holm"], 4), "pass=", t["pass_holm"])
    print("  claims1:", res1["superiority_claimed"], "sec_valid:", res1["secondary_valid_only"]["IFFT-LE"])
    print("Dry-run 2 (opposite MAE vs P90 on Sheikh-WPS):", flush=True)
    for t in res2["tests"]:
        print(" ", t["comparator"], t["endpoint"], "pt=", round(t["delta_point_m"], 3), "p_holm=", round(t["p_holm"], 4), "pass=", t["pass_holm"])
    print("  claims2:", res2["superiority_claimed"])
    assert res1["sdr_invalid_count"] == 2
    assert res1["secondary_valid_only"]["IFFT-LE"]["n_common_valid_pairs"] == n - 2
    assert res1["superiority_claimed"]["Sheikh-WPS"] is False  # penalized 45 m tail defeats raw win
    assert res2["superiority_claimed"]["IFFT-LE"] is True
    assert res2["superiority_claimed"]["Sheikh-WPS"] is False  # rejected when P90 fails despite better MAE!

    # Reject mismatched array lengths / missing / nonfinite / blank / bool cluster IDs (C Findings 2, 4 & Note 2)
    bad_cluster_cases = [
        np.arange(11),
        np.r_[np.arange(11), np.nan],
        ["G0"] * 11 + [""],
        ["G0"] * 11 + ["nan"],
        ["G0"] * 11 + [None],
        list(range(11)) + [False],
    ]
    for bad_ids in bad_cluster_cases:
        rejected = False
        try:
            run_confirmatory_comparison(
                np.zeros(12), np.ones(12, bool),
                {"IFFT-LE": np.full(12, 0.5), "Sheikh-WPS": np.full(12, 0.5)},
                {"IFFT-LE": np.ones(12, bool), "Sheikh-WPS": np.ones(12, bool)},
                bad_ids,
            )
        except ValueError:
            rejected = True
        assert rejected is True, f"Failed to reject {bad_ids!r}"
    nlos_sum = summarize_nlos_cohort(
        {"SDR-ANM-QML": np.array([1.0, 2.0, 6.0, float("nan")])},
        {"SDR-ANM-QML": np.array([True, True, True, False])},
        penalty_m=45.0,
    )
    assert nlos_sum["SDR-ANM-QML"]["n_valid"] == 3 and nlos_sum["SDR-ANM-QML"]["gross_outlier_gt_5m_pct"] == 50.0
    print("Dry-run checks PASSED (length/NaN/blank/None/bool cluster-ID rejection & NLOS summary).")


if __name__ == "__main__":
    _dry_run()
