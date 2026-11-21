"""BLE 6.0/6.2 Channel Sounding physical model, CRB, estimators, and benchmarks."""

from .model import (
    AUDITED_MAPPING_DISCREPANCIES,
    C_LIGHT,
    DF_HZ,
    F0_HZ,
    N_GRID,
    PERMITTED_GRID,
    SUPPORTED_CS_N_PATHS,
    ZenodoLoader,
    compute_mode2_timing,
    get_cohort_d_max_m,
    make_mask,
    simulate_cs_procedure,
)
from .crb import compute_fim_and_crb, verify_jacobian_fd
from .estimator import estimate_range_sdr_anm_qml
from .baselines import (
    baseline_fbss_music,
    baseline_ifft_argmax,
    baseline_ifft_leading_edge,
    baseline_kim_k15,
    baseline_sheikh_anm_music,
    baseline_sheikh_wps,
)

__all__ = [
    "AUDITED_MAPPING_DISCREPANCIES",
    "C_LIGHT",
    "DF_HZ",
    "F0_HZ",
    "N_GRID",
    "PERMITTED_GRID",
    "SUPPORTED_CS_N_PATHS",
    "ZenodoLoader",
    "compute_mode2_timing",
    "get_cohort_d_max_m",
    "make_mask",
    "simulate_cs_procedure",
    "compute_fim_and_crb",
    "verify_jacobian_fd",
    "estimate_range_sdr_anm_qml",
    "baseline_ifft_argmax",
    "baseline_ifft_leading_edge",
    "baseline_kim_k15",
    "baseline_fbss_music",
    "baseline_sheikh_wps",
    "baseline_sheikh_anm_music",
]
