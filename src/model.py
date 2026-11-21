"""Unified BLE 6.0/6.2 Channel Sounding physical signal model and Zenodo loader."""

from __future__ import annotations

import csv
import math
import pathlib
import zipfile
from dataclasses import dataclass
from typing import Iterable

import numpy as np

C_LIGHT: float = 299792458.0
F0_HZ: float = 2402e6
DF_HZ: float = 1e6
N_GRID: int = 75  # BLE CS channels 2..76 mapped to grid_k = ch - 2 in {0..74}
ADV_GRID: frozenset[int] = frozenset({21, 22, 23})  # channels 23, 24, 25
PERMITTED_GRID: np.ndarray = np.array(
    [k for k in range(N_GRID) if k not in ADV_GRID], dtype=int
)
KNOWN_ZENODO_ERRATA: frozenset[str] = frozenset({"outdoor/2_meter.csv"})
# Audited filename-vs-metadata discrepancy: outdoor/2_meter.csv maps to 1000 mm in
# outdoor_mapping.csv while raw unrestricted IFFT ToF peaks at 2.7813 m (consistent
# with nominal 2.0 m + ~0.88 m global hardware offset between 1_meter and 3_meter).
AUDITED_MAPPING_DISCREPANCIES: frozenset[str] = KNOWN_ZENODO_ERRATA
SUPPORTED_CS_N_PATHS: tuple[int, ...] = (1, 2, 3, 4)


def get_cohort_d_max_m(member_name: str) -> float:
    """Return acquisition-known upper range search bound (m) by measurement cohort."""
    norm = member_name.replace("\\", "/")
    return 90.0 if norm.startswith("outdoor/") else 45.0


def compute_mode2_timing(
    n_phys_steps: int,
    n_paths: int,
    n_slots_retained: int | None = None,
    t_pm_us: float = 40.0,
    t_sw_us: float = 2.0,
    t_rd_us: float = 5.0,
    t_ip2_us: float = 30.0,
) -> dict:
    """Compute Bluetooth Core v6.2 Mode-2 slot dwell, two-way PM lower bound, and step duration."""
    n_slots_acquired = int(n_phys_steps * n_paths)
    n_slots_ret = n_slots_acquired if n_slots_retained is None else int(n_slots_retained)
    slot_pm_dwell_ms = float(n_slots_ret * t_pm_us * 1e-3)
    two_way_pm_lower_bound_ms = float(2.0 * n_phys_steps * (n_paths + 1) * t_pm_us * 1e-3)
    mode2_step_duration_us = float(
        2.0 * (t_pm_us + t_sw_us) * (n_paths + 1) + 2.0 * t_rd_us + t_ip2_us
    )
    total_mode2_duration_ms = float(n_phys_steps * mode2_step_duration_us * 1e-3)
    if n_slots_ret > 0 and n_paths > 0:
        idx_arr = np.arange(n_slots_ret, dtype=np.float64)
        slot_timestamps_us = (idx_arr // n_paths) * mode2_step_duration_us + 2.0 * (
            t_pm_us + t_sw_us
        ) * ((idx_arr % n_paths) + 1.0)
    else:
        slot_timestamps_us = np.zeros(0, dtype=np.float64)
    return {
        "n_slots_acquired": n_slots_acquired,
        "n_slots_retained": n_slots_ret,
        "slot_pm_dwell_ms": slot_pm_dwell_ms,
        "two_way_pm_lower_bound_ms": two_way_pm_lower_bound_ms,
        "mode2_step_duration_us": mode2_step_duration_us,
        "total_mode2_duration_ms": total_mode2_duration_ms,
        "slot_timestamps_us": slot_timestamps_us,
    }


def _erase_channels(erase_chs: Iterable[int]) -> np.ndarray:
    erase_grid = {ch - 2 for ch in erase_chs} | ADV_GRID
    return np.array([k for k in range(N_GRID) if k not in erase_grid], dtype=int)


def _keep_channels(keep_chs: Iterable[int]) -> np.ndarray:
    keep_grid = {ch - 2 for ch in keep_chs} - ADV_GRID
    return np.array(sorted(k for k in keep_grid if 0 <= k < N_GRID), dtype=int)


def make_mask(kind: str) -> np.ndarray:
    """Return sorted grid indices k in {0..74} (ch = k + 2) for standard masks."""
    if kind == "full":
        return PERMITTED_GRID.copy()
    if kind == "wifi_ch1":
        return _erase_channels(range(2, 22))
    if kind == "wifi_ch6":
        return _erase_channels(range(25, 46))
    if kind == "wifi_ch11":
        return _erase_channels(range(50, 71))
    if kind == "wifi_ch1_6":
        return _erase_channels(list(range(2, 22)) + list(range(25, 46)))
    if kind == "wifi_ch6_11":
        return _erase_channels(list(range(25, 46)) + list(range(50, 71)))
    if kind == "wifi_ch1_11":
        return _erase_channels(list(range(2, 22)) + list(range(50, 71)))
    if kind == "wifi_40mhz_center":
        return _erase_channels(range(20, 60))

    # Exact Zenodo channel_map masks
    zenodo_masks = {
        "20_center": _keep_channels(range(30, 50)),
        "20_left": _keep_channels(range(2, 22)),
        "20_right": _keep_channels(range(57, 77)),
        "20_spread": _keep_channels(list(range(2, 12)) + list(range(67, 77))),
        "30_center": _keep_channels(range(26, 56)),
        "30_left": _keep_channels(range(2, 35)),
        "30_right": _keep_channels(range(47, 77)),
        "30_spread": _keep_channels(list(range(2, 17)) + list(range(62, 77))),
        "40_center": _keep_channels(range(19, 61)),
        "40_left": _keep_channels(range(2, 45)),
        "40_right": _keep_channels(range(37, 77)),
        "40_spread": _keep_channels(list(range(2, 22)) + list(range(57, 77))),
        "50_center": _keep_channels(range(13, 65)),
        "50_left": _keep_channels(range(2, 55)),
        "50_right": _keep_channels(range(27, 77)),
        "50_spread": _keep_channels(list(range(2, 30)) + list(range(52, 77))),
    }
    if kind in zenodo_masks:
        return zenodo_masks[kind]
    raise ValueError(f"Unknown mask kind: {kind!r}")


def _build_schedule(
    omega_grid: np.ndarray,
    n_steps: int,
    n_paths: int,
    hop_mode: str,
    rng: np.random.Generator,
    p_schedule_mode: str = "core62_full_steps",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build slot index, grid channel k, antenna path p, physical step s, and slot index r."""
    m_ch = len(omega_grid)
    if m_ch == 0:
        raise ValueError("omega_grid must be non-empty")
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")

    if n_paths == 1:
        step_idx = np.arange(n_steps, dtype=int)
        phys_step_idx = step_idx.copy()
        slot_idx = np.zeros(n_steps, dtype=int)
        ant_idx = np.zeros(n_steps, dtype=int)
        if hop_mode == "monotonic":
            sorted_grid = np.sort(omega_grid)
            if n_steps <= m_ch:
                grid_k = sorted_grid[:n_steps].astype(int)
            else:
                reps = int(math.ceil(n_steps / m_ch))
                grid_k = np.repeat(sorted_grid, reps)[:n_steps].astype(int)
            return step_idx, grid_k, ant_idx, phys_step_idx, slot_idx
        if hop_mode in ("pseudorandom", "core62"):
            seq: list[int] = []
            while len(seq) < n_steps:
                perm = omega_grid.copy()
                rng.shuffle(perm)
                seq.extend(perm.tolist())
            grid_k = np.asarray(seq[:n_steps], dtype=int)
            return step_idx, grid_k, ant_idx, phys_step_idx, slot_idx
        raise ValueError(f"Unsupported hop_mode: {hop_mode!r}")

    if hop_mode == "monotonic":
        if p_schedule_mode == "legacy_round_robin":
            step_idx = np.arange(n_steps, dtype=int)
            sorted_grid = np.sort(omega_grid)
            if n_steps <= m_ch:
                grid_k = sorted_grid[:n_steps].astype(int)
            else:
                reps = int(math.ceil(n_steps / m_ch))
                grid_k = np.repeat(sorted_grid, reps)[:n_steps].astype(int)
            ant_idx = step_idx % n_paths
            return step_idx, grid_k, ant_idx, step_idx.copy(), np.zeros(n_steps, dtype=int)
        n_obs = n_steps * n_paths if p_schedule_mode == "core62_full_steps" else n_steps
        n_phys = int(math.ceil(n_obs / n_paths))
        sorted_grid = np.sort(omega_grid)
        step_idx = np.arange(n_obs, dtype=int)
        phys_step_idx = step_idx // n_paths
        slot_idx = step_idx % n_paths
        ant_idx = slot_idx.copy()
        grid_k = np.asarray([sorted_grid[s % m_ch] for s in phys_step_idx], dtype=int)
        return step_idx, grid_k, ant_idx, phys_step_idx, slot_idx

    if hop_mode in ("pseudorandom", "core62"):
        if p_schedule_mode == "legacy_round_robin":
            step_idx = np.arange(n_steps, dtype=int)
            per_ant_queues: list[list[int]] = []
            per_ant_need = (n_steps // n_paths) + 2
            for _ in range(n_paths):
                seq = []
                while len(seq) < per_ant_need:
                    perm = omega_grid.copy()
                    rng.shuffle(perm)
                    seq.extend(perm.tolist())
                per_ant_queues.append(seq)
            ant_idx = step_idx % n_paths
            grid_k = np.empty(n_steps, dtype=int)
            cursors = [0] * n_paths
            for n in range(n_steps):
                p = int(ant_idx[n])
                grid_k[n] = per_ant_queues[p][cursors[p]]
                cursors[p] += 1
            return step_idx, grid_k, ant_idx, step_idx.copy(), np.zeros(n_steps, dtype=int)

        if p_schedule_mode == "core62_full_steps":
            n_phys = n_steps
            n_obs = n_phys * n_paths
            phys_seq: list[int] = []
            while len(phys_seq) < n_phys:
                perm = omega_grid.copy()
                rng.shuffle(perm)
                phys_seq.extend(perm.tolist())
            phys_channels = phys_seq[:n_phys]
        elif p_schedule_mode == "core62_std_short_steps":
            # Standard unconditioned short-step schedule with complete final physical step
            n_phys = int(math.ceil(n_steps / n_paths))
            n_obs = n_phys * n_paths
            phys_seq = []
            while len(phys_seq) < n_phys:
                perm = omega_grid.copy()
                rng.shuffle(perm)
                phys_seq.extend(perm.tolist())
            phys_channels = phys_seq[:n_phys]
        elif p_schedule_mode in ("custom_anchor_matched_surrogate", "core62_matched_budget"):
            # Custom repeat-enriched 159-slot simulation surrogate guaranteeing >= 8 anchors
            n_obs = n_steps
            n_phys = int(math.ceil(n_obs / n_paths))
            if m_ch <= n_phys - 8:
                phys_seq = []
                while len(phys_seq) < n_phys:
                    perm = omega_grid.copy()
                    rng.shuffle(perm)
                    phys_seq.extend(perm.tolist())
                phys_channels = phys_seq[:n_phys]
            else:
                n_anchor = min(8, m_ch, max(1, n_phys // 4))
                n_unique = min(m_ch, max(1, n_phys - n_anchor))
                shuf = omega_grid.copy()
                rng.shuffle(shuf)
                unique_pool = np.sort(shuf[:n_unique])
                n_extra = n_phys - n_unique
                anchor_idx = np.linspace(0, n_unique - 1, num=min(n_extra, n_unique), dtype=int)
                anchors = unique_pool[anchor_idx].tolist()
                while len(anchors) < n_extra:
                    anchors.extend(unique_pool[anchor_idx].tolist())
                comb = np.asarray(unique_pool.tolist() + anchors[:n_extra], dtype=int)
                rng.shuffle(comb)
                phys_channels = comb.tolist()
        else:
            raise ValueError(f"Unsupported p_schedule_mode: {p_schedule_mode!r}")

        grid_list: list[int] = []
        ant_list: list[int] = []
        phys_list: list[int] = []
        slot_list: list[int] = []
        for s_idx in range(n_phys):
            k_s = int(phys_channels[s_idx])
            perm_p = rng.permutation(n_paths)
            for r_idx in range(n_paths):
                grid_list.append(k_s)
                ant_list.append(int(perm_p[r_idx]))
                phys_list.append(s_idx)
                slot_list.append(r_idx)

        step_idx = np.arange(n_obs, dtype=int)
        grid_k = np.asarray(grid_list[:n_obs], dtype=int)
        ant_idx = np.asarray(ant_list[:n_obs], dtype=int)
        phys_step_idx = np.asarray(phys_list[:n_obs], dtype=int)
        slot_idx = np.asarray(slot_list[:n_obs], dtype=int)
        return step_idx, grid_k, ant_idx, phys_step_idx, slot_idx

    raise ValueError(f"Unsupported hop_mode: {hop_mode!r}")


def simulate_cs_procedure(
    taus: np.ndarray,
    alphas: np.ndarray,
    omega_grid: np.ndarray | None = None,
    n_steps: int = 159,
    hop_mode: str = "pseudorandom",
    omega_step: float = 0.0,
    gamma_bi: float = 0.0,
    tau_sys: float = 0.0,
    snr_db: float = 20.0,
    sigma_phi_rad: float = 0.0,
    outlier_rate: float = 0.0,
    rng: np.random.Generator | None = None,
    p_schedule_mode: str = "core62_full_steps",
    n_subevents: int = 1,
    noise_model: str = "surrogate_awgn",
    subevent_reset_phase_sigma_rad: float = 0.0,
    use_physical_timestamps: bool = False,
) -> dict:
    """Simulate a single BLE CS bidirectional procedure returning Mode-2/3 IQ products.

    Follows verified IEEE/nRF54L15 sign convention:
        h_p(k) = sum_l alphas[p, l] * exp(-j * 2*pi * k * DF_HZ * (taus[l] + tau_sys))
        y[n] = exp(j * (omega_step * n_eff + gamma_bi * n_eff * k + delta_n)) * h_p(k)^2 + w[n]
    where raw y[n] directly models z_I[n] * z_R[n] (no conjugation).
    Supports Core 6.2 shared-step antenna-slot scheduling, Mode-0 subevent boundaries,
    and either surrogate post-product AWGN or exact raw noisy IQ multiplication (product_iq).
    """
    if rng is None:
        rng = np.random.default_rng()
    if omega_grid is None:
        omega_grid = PERMITTED_GRID

    taus = np.asarray(taus, dtype=np.float64)
    alphas = np.asarray(alphas, dtype=np.complex128)
    if alphas.ndim == 1:
        alphas = alphas[None, :]
    n_paths, n_multipath = alphas.shape
    if taus.shape != (n_multipath,):
        raise ValueError("taus length must match alphas.shape[1]")

    step_idx, grid_k, ant_idx, phys_step_idx, slot_idx = _build_schedule(
        omega_grid, n_steps, n_paths, hop_mode, rng, p_schedule_mode=p_schedule_mode
    )
    n_obs = len(step_idx)
    n_phys_steps = int(np.max(phys_step_idx) + 1) if n_obs > 0 else 0
    n_slots_acquired = n_phys_steps * n_paths if (n_paths > 1 and p_schedule_mode != "legacy_round_robin") else n_obs
    n_unique_tones = int(len(np.unique(grid_k))) if n_obs > 0 else 0
    final_step_paths = int(np.sum(phys_step_idx == (n_phys_steps - 1))) if n_obs > 0 else 0
    timing_info = compute_mode2_timing(n_phys_steps, n_paths, n_slots_retained=n_obs)
    slot_ts = timing_info["slot_timestamps_us"]
    mean_dt_us = float(np.mean(np.diff(slot_ts))) if len(slot_ts) > 1 else 1.0
    taus_eff = taus + tau_sys

    # Partition chronologically into Mode-0 subevents if requested
    n_sub = max(1, int(n_subevents))
    if n_sub == 1:
        subevent_idx = np.zeros(n_obs, dtype=int)
        if use_physical_timestamps and n_obs > 0:
            step_eff = (slot_ts - slot_ts[0]) / max(mean_dt_us, 1e-9)
        else:
            step_eff = step_idx.astype(np.float64)
        sub_jump = np.zeros(n_obs, dtype=np.float64)
    else:
        subevent_idx = np.minimum(n_sub - 1, (np.arange(n_obs) * n_sub) // max(n_obs, 1)).astype(int)
        if use_physical_timestamps and n_obs > 0:
            ts_starts = np.zeros(n_sub, dtype=np.float64)
            for m_s in range(n_sub):
                idxs_m = np.flatnonzero(subevent_idx == m_s)
                ts_starts[m_s] = float(slot_ts[idxs_m[0]]) if len(idxs_m) else 0.0
            step_eff = (slot_ts - ts_starts[subevent_idx]) / max(mean_dt_us, 1e-9)
        else:
            starts = np.zeros(n_sub, dtype=int)
            for m_s in range(n_sub):
                idxs_m = np.flatnonzero(subevent_idx == m_s)
                starts[m_s] = int(step_idx[idxs_m[0]]) if len(idxs_m) else 0
            step_eff = (step_idx - starts[subevent_idx]).astype(np.float64)
        if subevent_reset_phase_sigma_rad > 0.0:
            jumps = rng.normal(0.0, subevent_reset_phase_sigma_rad, size=n_sub)
            jumps[0] = 0.0
            sub_jump = jumps[subevent_idx]
        else:
            sub_jump = np.zeros(n_obs, dtype=np.float64)

    # One-way CFR per (slot): exp(-j * 2*pi * k * DF * tau_eff)
    phase_mat = -2.0 * np.pi * DF_HZ * np.outer(grid_k, taus_eff)
    e_mat = np.exp(1j * phase_mat)  # (n_obs, L)
    a_step = alphas[ant_idx, :]  # (n_obs, L)
    h_step = np.sum(a_step * e_mat, axis=1)  # (n_obs,)

    delta = rng.normal(0.0, sigma_phi_rad, size=n_obs) if sigma_phi_rad > 0 else 0.0
    drift_phase = omega_step * step_eff + gamma_bi * step_eff * grid_k + delta + sub_jump
    mu_clean = np.exp(1j * drift_phase) * (h_step ** 2)

    sig_pow = max(float(np.mean(np.abs(mu_clean) ** 2)), 1e-12)
    sigma2 = sig_pow / (10.0 ** (snr_db / 10.0))
    std_comp = math.sqrt(sigma2 / 2.0)

    if noise_model == "surrogate_awgn":
        noise = rng.normal(0.0, std_comp, size=n_obs) + 1j * rng.normal(0.0, std_comp, size=n_obs)
        y = mu_clean + noise
    elif noise_model == "product_iq":
        h_pow_mean = max(float(np.mean(np.abs(h_step) ** 2)), 1e-12)
        sigma2_raw = max(0.0, math.sqrt(h_pow_mean ** 2 + sigma2) - h_pow_mean)
        std_raw = math.sqrt(sigma2_raw / 2.0)
        theta_n = rng.uniform(-np.pi, np.pi, size=n_obs)
        w_r = rng.normal(0.0, std_raw, size=n_obs) + 1j * rng.normal(0.0, std_raw, size=n_obs)
        w_i = rng.normal(0.0, std_raw, size=n_obs) + 1j * rng.normal(0.0, std_raw, size=n_obs)
        z_r = np.exp(1j * theta_n) * h_step + w_r
        z_i = np.exp(-1j * theta_n) * np.exp(1j * drift_phase) * h_step + w_i
        y = z_i * z_r
    else:
        raise ValueError(f"Unsupported noise_model: {noise_model!r}")

    q_i = np.zeros(n_obs, dtype=int)
    q_r = np.zeros(n_obs, dtype=int)
    if outlier_rate > 0.0:
        is_bad = rng.random(n_obs) < outlier_rate
        n_bad = int(np.sum(is_bad))
        if n_bad > 0:
            q_i[is_bad] = 2
            q_r[is_bad] = rng.choice([0, 1, 2], size=n_bad)
            bad_phase = rng.uniform(-np.pi, np.pi, size=n_bad)
            y[is_bad] = np.abs(y[is_bad]) * np.exp(1j * bad_phase) + rng.normal(0, 3 * std_comp, size=n_bad)

    return {
        "step_idx": step_idx,
        "phys_step_idx": phys_step_idx,
        "slot_idx": slot_idx,
        "subevent_idx": subevent_idx,
        "grid_k": grid_k,
        "ant_idx": ant_idx,
        "y": y,
        "q_i": q_i,
        "q_r": q_r,
        "n_paths": n_paths,
        "n_phys_steps": n_phys_steps,
        "n_slots_total": n_obs,
        "n_slots_acquired": n_slots_acquired,
        "n_slots_retained": n_obs,
        "n_unique_tones": n_unique_tones,
        "final_step_paths": final_step_paths,
        "p_schedule_mode": p_schedule_mode,
        "schedule_provenance": (
            "custom_repeat_enriched_surrogate"
            if (n_paths > 1 and p_schedule_mode in ("custom_anchor_matched_surrogate", "core62_matched_budget"))
            else p_schedule_mode
        ),
        "timing_info": timing_info,
        "slot_timestamps_us": slot_ts,
        "noise_model": noise_model,
        "sigma2_true": sigma2,
        "taus_true": taus,
        "alphas_true": alphas,
        "omega_step_true": omega_step,
        "d_true_m": float(C_LIGHT * taus[0]),
        "mu_clean": mu_clean,
    }


@dataclass
class ZenodoLoader:
    """Loader for 10.5281/zenodo.17347695 BLE CS IQ procedures & millimeter ground truth."""

    data_dir: pathlib.Path
    exclude_known_errata: bool = True

    def __post_init__(self) -> None:
        self.data_dir = pathlib.Path(self.data_dir)
        self.zip_path = self.data_dir / "data.zip"
        self.dist_map: dict[str, float] = {}
        self.meta_map: dict[str, dict] = {}
        self._load_mappings()
        self._zf = zipfile.ZipFile(self.zip_path, "r")
        self.csv_names = sorted(n for n in self._zf.namelist() if n.endswith(".csv"))

    def _load_mappings(self) -> None:
        mpath = self.data_dir / "mapping.csv"
        if mpath.exists():
            with mpath.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    fn = row["filename"].replace("\\", "/")
                    dx = float(row["initiator_x"]) - float(row["reflector_x"])
                    dy = float(row["initiator_y"]) - float(row["reflector_y"])
                    dz = float(row["initiator_z"]) - float(row["reflector_z"])
                    # Coordinates in mapping.csv are in millimeters
                    self.dist_map[fn] = math.sqrt(dx * dx + dy * dy + dz * dz) / 1000.0
                    self.meta_map[fn] = row

        ompath = self.data_dir / "outdoor_mapping.csv"
        if ompath.exists():
            with ompath.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    fn = row["filename"].replace("\\", "/")
                    self.dist_map[fn] = float(row["distance(mm)"]) / 1000.0
                    self.meta_map[fn] = row

    def list_files(self, prefix: str = "", include_errata: bool = False) -> list[str]:
        skip = frozenset() if (include_errata or not self.exclude_known_errata) else KNOWN_ZENODO_ERRATA
        return [n for n in self.csv_names if n.startswith(prefix) and n not in skip]

    def get_true_distance(self, member_name: str) -> float:
        return self.dist_map[member_name]

    def load_file_procedures(
        self, member_name: str, max_procs: int | None = None
    ) -> list[dict]:
        """Parse procedures from a CSV inside data.zip with y = z_I * z_R (NO conj)."""
        text = self._zf.read(member_name).decode("utf-8", errors="replace")
        d_true = self.dist_map.get(member_name, float("nan"))
        procs: list[dict] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            steps_idx = []
            grids = []
            ants = []
            y_vals = []
            zi_vals = []
            zr_vals = []
            qi_vals = []
            qr_vals = []
            for idx, tok in enumerate(line.split(",")):
                tok = tok.strip()
                if not tok:
                    continue
                parts = tok.split(";")
                if len(parts) != 8:
                    continue
                q_i, q_r, ch, ant, ii, iq, ri, rq = (int(x) for x in parts)
                if not (2 <= ch <= 76):
                    continue
                zi = complex(ii, iq)
                zr = complex(ri, rq)
                steps_idx.append(idx)
                grids.append(ch - 2)
                ants.append(ant)
                zi_vals.append(zi)
                zr_vals.append(zr)
                y_vals.append(zi * zr)  # Verified positive-tau IFFT convention: no conj
                qi_vals.append(q_i)
                qr_vals.append(q_r)
            if steps_idx:
                procs.append(
                    {
                        "filename": member_name,
                        "proc_idx": len(procs),
                        "step_idx": np.asarray(steps_idx, dtype=int),
                        "phys_step_idx": np.asarray(steps_idx, dtype=int),
                        "slot_idx": np.zeros(len(steps_idx), dtype=int),
                        "subevent_idx": np.zeros(len(steps_idx), dtype=int),
                        "grid_k": np.asarray(grids, dtype=int),
                        "ant_idx": np.asarray(ants, dtype=int),
                        "y": np.asarray(y_vals, dtype=np.complex128),
                        "z_i": np.asarray(zi_vals, dtype=np.complex128),
                        "z_r": np.asarray(zr_vals, dtype=np.complex128),
                        "q_i": np.asarray(qi_vals, dtype=int),
                        "q_r": np.asarray(qr_vals, dtype=int),
                        "n_paths": 1,
                        "d_true_m": d_true,
                        "known_erratum": member_name in KNOWN_ZENODO_ERRATA,
                    }
                )
                if max_procs is not None and len(procs) >= max_procs:
                    break
        return procs

    def close(self) -> None:
        self._zf.close()
