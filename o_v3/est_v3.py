"""v3 = frozen v2.1 fit + coarse leading-edge consistency gate (O 11-21).

Rule: fit exactly as v2.1 and select the first atom with |beta| >= 0.35 max (v2.1 output d_v2).
Compute the IFFT leading-edge estimate d_LE from the same procedure (src.baselines, unchanged).
If d_v2 - d_LE > DELTA_LATE (0.5 m), the super-resolved fit is later than the coarse leading edge,
which is the signature of a weak direct path suppressed by the two-way product (its auto-term
scales with a_1^2). Then re-select on the SAME fitted atoms with the lower threshold ALPHA_LATE (0.25).
Nothing is refit; no other change.
"""
import math
import numpy as np
from src.baselines import baseline_ifft_leading_edge
from src.model import C_LIGHT
from o_v3.fit_state import fit_v21_state

DELTA_LATE = 0.5
ALPHA_BASE = 0.35
ALPHA_LATE = 0.25


def _select(taus, betas, t2_min, alpha):
    amps = np.linalg.norm(betas, axis=1)
    keep = np.flatnonzero(taus > t2_min + 0.15 / C_LIGHT)
    if 0 < len(keep) < len(taus):
        taus, amps = taus[keep], amps[keep]
    for i in range(len(taus)):
        if amps[i] >= alpha * amps.max():
            return float(taus[i])
    return float(taus[int(np.argmax(amps))])


def estimate_v3(p, d_cal_m, delta_late=DELTA_LATE, alpha_late=ALPHA_LATE):
    f = fit_v21_state(p["step_idx"], p["grid_k"], p["ant_idx"], p["y"], p["q_i"], p["q_r"], d_cal_m=d_cal_m, d_max_m=p["d_max_m"])
    if not f.get("valid_estimate", False):
        return {"d_hat_m": float("nan"), "flag": "invalid"}
    d_v2 = max(0.0, 0.5 * C_LIGHT * _select(f["taus"], f["betas"], f["t2_min"], ALPHA_BASE) - d_cal_m)
    d_le = float(baseline_ifft_leading_edge(p["grid_k"], p["ant_idx"], p["y"], p["q_i"], p["q_r"], d_cal_m=d_cal_m, d_max_m=p["d_max_m"]))
    if math.isfinite(d_le) and d_v2 - d_le > delta_late:
        d = max(0.0, 0.5 * C_LIGHT * _select(f["taus"], f["betas"], f["t2_min"], alpha_late) - d_cal_m)
        return {"d_hat_m": d, "flag": "late_gate", "d_v2_m": d_v2, "d_le_m": d_le}
    return {"d_hat_m": d_v2, "flag": "pass", "d_v2_m": d_v2, "d_le_m": d_le}
