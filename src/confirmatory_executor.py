"""Prespecified confirmatory cohort executor with manifest verification, OS-terminating timeout, and stats binding."""

from __future__ import annotations

import atexit
import hashlib
import importlib
import json
import marshal
import math
import multiprocessing as mp
import os
import pathlib
import pickle
import time
import types
import zipfile
from typing import Any, Callable

import numpy as np

from .baselines import (
    baseline_fbss_music,
    baseline_ifft_argmax,
    baseline_ifft_leading_edge,
    baseline_kim_k15,
    baseline_sheikh_anm_music,
    baseline_sheikh_wps,
)
from .confirmatory_stats import run_confirmatory_comparison, summarize_nlos_cohort
from .estimator import _filter_and_validate_inputs, estimate_range_sdr_anm_qml
from .model import C_LIGHT, ZenodoLoader, make_mask, simulate_cs_procedure

FROZEN_PROTOCOL_ID: str = "ble-cs-confirmatory-v1.0-20261120"
FROZEN_MANIFEST_SHA256: str = "1ef0de5a2186eb4a1843ab4532184058c4bca7afb6ad4021519304c108f3bd2a"
FROZEN_DATA_ZIP_SHA256: str = "1f3a34fbc0110a1906c7db5d63b20d9d152c90243b8a28a545664d2af48c8dba"
D_CAL_SHARED_M: float = 0.881149670238504
FROZEN_D_CAL_M: float = D_CAL_SHARED_M
PER_METHOD_TIMEOUT_S: float = 30.0
FROZEN_PER_METHOD_TIMEOUT_S: float = PER_METHOD_TIMEOUT_S
FROZEN_N_BOOT: int = 2000
FROZEN_BOOT_SEED: int = 20261120
FROZEN_ALPHA: float = 0.05
MAX_PROCS_PER_FILE: int = 15
_INVALID_TEXT_TOKENS = frozenset({"nan", "none", "nat", "null", "inf", "+inf", "-inf"})

PRESPECIFIED_METHODS: tuple[str, ...] = (
    "SDR-ANM-QML",
    "IFFT-Max",
    "IFFT-LE",
    "Adapted Kim-K15",
    "Adapted Wieme-FBSS",
    "Sheikh-WPS",
    "Adapted Sheikh-h2-ANM",
    "Longer-budget adapted Sheikh-h2-ANM",
)
_FROZEN_METHODS: tuple[str, ...] = PRESPECIFIED_METHODS
PRESPECIFIED_COMPARATORS: tuple[str, ...] = ("IFFT-LE", "Sheikh-WPS")
_FROZEN_COMPARATORS: tuple[str, ...] = PRESPECIFIED_COMPARATORS

_FROZEN_COHORT_SPECS: dict[str, tuple[int, int, int, float, str]] = {
    "primary_tier1_indoor_los": (78, 74, 1170, 45.0, "primary_confirmatory"),
    "secondary_tier1_outdoor_los": (22, 22, 330, 90.0, "secondary_confirmatory"),
    "secondary_tier2_indoor_los": (28, 1, 420, 45.0, "descriptive_only"),
    "confirmatory_tier1_indoor_nlos": (80, 76, 1200, 45.0, "secondary_confirmatory_nlos"),
    "confirmatory_tier2_indoor_nlos": (41, 14, 615, 45.0, "descriptive_only_nlos"),
}

COHORT_SPECS: dict[str, dict[str, Any]] = {
    "primary_tier1_indoor_los": {
        "n_files": 78,
        "n_clusters": 74,
        "n_procedures_first15": 1170,
        "d_max_m": 45.0,
        "inferential_role": "primary_confirmatory",
    },
    "secondary_tier1_outdoor_los": {
        "n_files": 22,
        "n_clusters": 22,
        "n_procedures_first15": 330,
        "d_max_m": 90.0,
        "inferential_role": "secondary_confirmatory",
    },
    "secondary_tier2_indoor_los": {
        "n_files": 28,
        "n_clusters": 1,
        "n_procedures_first15": 420,
        "d_max_m": 45.0,
        "inferential_role": "descriptive_only",
    },
    "confirmatory_tier1_indoor_nlos": {
        "n_files": 80,
        "n_clusters": 76,
        "n_procedures_first15": 1200,
        "d_max_m": 45.0,
        "inferential_role": "secondary_confirmatory_nlos",
    },
    "confirmatory_tier2_indoor_nlos": {
        "n_files": 41,
        "n_clusters": 14,
        "n_procedures_first15": 615,
        "d_max_m": 45.0,
        "inferential_role": "descriptive_only_nlos",
    },
}

_OPT_DIAG_KEYS: tuple[str, ...] = (
    "admm_converged",
    "admm_primal_rel",
    "admm_dual_rel",
    "n_admm_iter",
    "qml_status",
    "qml_optimality",
    "qml_nfev",
    "L_star",
    "w_step_hat",
    "drift_status",
    "subevent_align_status",
    "rss_norm",
)


def _validate_nonblank_str(val: object, field: str) -> str:
    if not isinstance(val, str):
        raise ValueError(f"{field} must be a string, got {type(val).__name__}")
    s = val.strip()
    if (not s) or (s.lower() in _INVALID_TEXT_TOKENS):
        raise ValueError(f"{field} is blank or invalid: {val!r}")
    return s


def _validate_finite_float(val: object, field: str, allow_zero: bool = True) -> float:
    if isinstance(val, bool) or val is None:
        raise ValueError(f"{field} must be a real number, got {val!r}")
    try:
        out = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not convertible to float: {val!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{field} must be finite, got {val!r}")
    if (out < 0.0) if allow_zero else (out <= 0.0):
        raise ValueError(f"{field} out of range: {out!r}")
    return out


def _validate_nonneg_int(val: object, field: str) -> int:
    if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, np.integer)):
        raise ValueError(f"{field} must be an integer, got {val!r}")
    iv = int(val)
    if iv < 0:
        raise ValueError(f"{field} must be >= 0, got {iv}")
    return iv


def _select_cohort_rows(files: list[dict], cohort_name: str) -> list[dict]:
    if cohort_name == "primary_tier1_indoor_los":
        return [
            f for f in files
            if f["partition"] == "confirmatory_tier1_unseen_geometry"
            and f["los_nlos"] == "LOS"
            and not f["filename"].startswith("outdoor/")
        ]
    if cohort_name == "secondary_tier1_outdoor_los":
        return [
            f for f in files
            if f["partition"] == "confirmatory_tier1_unseen_geometry"
            and f["los_nlos"] == "LOS"
            and f["filename"].startswith("outdoor/")
        ]
    if cohort_name == "secondary_tier2_indoor_los":
        return [
            f for f in files
            if f["partition"] == "confirmatory_tier2_shared_geometry_new_config"
            and f["los_nlos"] == "LOS"
        ]
    if cohort_name == "confirmatory_tier1_indoor_nlos":
        return [
            f for f in files
            if f["partition"] == "confirmatory_tier1_unseen_geometry"
            and f["los_nlos"] == "NLOS"
        ]
    if cohort_name == "confirmatory_tier2_indoor_nlos":
        return [
            f for f in files
            if f["partition"] == "confirmatory_tier2_shared_geometry_new_config"
            and f["los_nlos"] == "NLOS"
        ]
    raise ValueError(f"Unknown cohort_name: {cohort_name!r}")


def load_cohort_manifest_bundle(manifest_path: str | pathlib.Path, cohort_name: str) -> dict:
    """Load manifest cohort entries, validate metadata/counts, and build independent expected proc keys."""
    mpath = pathlib.Path(manifest_path)
    raw_bytes = mpath.read_bytes()
    manifest_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    data = json.loads(raw_bytes.decode("utf-8"))
    zip_sha = _validate_nonblank_str(data.get("data_zip_sha256", ""), "data_zip_sha256")
    if len(zip_sha) != 64:
        raise ValueError(f"Malformed data_zip_sha256 in manifest: {zip_sha!r}")

    sel = sorted(_select_cohort_rows(data["files"], cohort_name), key=lambda r: r["filename"])
    spec = COHORT_SPECS[cohort_name]
    expected_proc_meta: dict[tuple[str, int], dict] = {}
    file_hashes: dict[str, str] = {}

    for row in sel:
        fn = _validate_nonblank_str(row.get("filename"), "filename")
        gid = _validate_nonblank_str(row.get("geometry_id"), f"geometry_id[{fn}]")
        csv_sha = _validate_nonblank_str(row.get("csv_sha256"), f"csv_sha256[{fn}]")
        if len(csv_sha) != 64:
            raise ValueError(f"Malformed csv_sha256 for {fn}: {csv_sha!r}")
        n_procs = _validate_nonneg_int(row.get("n_procedures"), f"n_procedures[{fn}]")
        if n_procs < 1:
            raise ValueError(f"n_procedures must be >= 1 for {fn}, got {n_procs}")
        d_true_m = _validate_finite_float(row.get("d_true_m"), f"d_true_m[{fn}]", allow_zero=True)
        d_max_m = _validate_finite_float(row.get("d_max_cohort_m"), f"d_max_cohort_m[{fn}]", allow_zero=False)
        if abs(d_max_m - float(spec["d_max_m"])) > 1e-9:
            raise ValueError(f"Manifest d_max mismatch for {fn}: {d_max_m} != {spec['d_max_m']}")
        if fn in file_hashes:
            raise ValueError(f"Duplicate filename in manifest selection: {fn!r}")
        file_hashes[fn] = csv_sha
        for p_idx in range(min(MAX_PROCS_PER_FILE, n_procs)):
            expected_proc_meta[(fn, p_idx)] = {
                "geometry_id": gid,
                "d_true_m": d_true_m,
                "d_max_cohort_m": d_max_m,
                "csv_sha256": csv_sha,
                "n_paths": 1,
            }

    n_clusters = len({row["geometry_id"] for row in sel})
    n_procs_first15 = len(expected_proc_meta)
    if (
        len(sel) != spec["n_files"]
        or n_clusters != spec["n_clusters"]
        or n_procs_first15 != spec["n_procedures_first15"]
    ):
        raise ValueError(
            f"Cohort {cohort_name} count mismatch: got files={len(sel)}, clusters={n_clusters}, "
            f"procs={n_procs_first15}, expected {spec}"
        )
    return {
        "cohort_name": cohort_name,
        "spec": spec,
        "manifest_path": str(mpath),
        "manifest_sha256": manifest_sha256,
        "data_zip_sha256": zip_sha,
        "rows": sel,
        "file_hashes": file_hashes,
        "expected_proc_meta": expected_proc_meta,
    }


def load_cohort_manifest_rows(manifest_path: str | pathlib.Path, cohort_name: str) -> list[dict]:
    """Load sorted manifest entries for a named prespecified cohort and assert exact counts."""
    return load_cohort_manifest_bundle(manifest_path, cohort_name)["rows"]


class _MarshaledClosureFreeCallable:
    """Picklable transport wrapper for closure-free module/scalar inline functions."""

    def __init__(
        self,
        code_bytes: bytes,
        name: str,
        defaults: tuple | None,
        kwdefaults: dict | None,
        module_imports: dict[str, str],
        scalar_globals: dict[str, Any],
    ) -> None:
        self.code_bytes = code_bytes
        self.name = name
        self.defaults = defaults
        self.kwdefaults = kwdefaults
        self.module_imports = module_imports
        self.scalar_globals = scalar_globals

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        globs: dict[str, Any] = {"__builtins__": __builtins__}
        for alias, mod_name in self.module_imports.items():
            globs[alias] = importlib.import_module(mod_name)
        globs.update(self.scalar_globals)
        code_obj = marshal.loads(self.code_bytes)
        fn = types.FunctionType(code_obj, globs, self.name, self.defaults)
        if self.kwdefaults:
            fn.__kwdefaults__ = self.kwdefaults
        return fn(*args, **kwargs)


def _prepare_picklable_callable(fn: Callable[..., Any]) -> Callable[..., Any]:
    try:
        pickle.dumps(fn)
        return fn
    except Exception:
        if isinstance(fn, types.FunctionType) and fn.__closure__ is None:
            mod_imports: dict[str, str] = {}
            scalar_globals: dict[str, Any] = {}
            for name in fn.__code__.co_names:
                if name in fn.__globals__:
                    val = fn.__globals__[name]
                    if isinstance(val, types.ModuleType):
                        mod_imports[name] = val.__name__
                    else:
                        pickle.dumps(val)
                        scalar_globals[name] = val
            wrapped = _MarshaledClosureFreeCallable(
                code_bytes=marshal.dumps(fn.__code__),
                name=fn.__name__,
                defaults=fn.__defaults__,
                kwdefaults=fn.__kwdefaults__,
                module_imports=mod_imports,
                scalar_globals=scalar_globals,
            )
            pickle.dumps(wrapped)
            return wrapped
        raise TypeError(
            "Worker callable must be picklable (or closure-free) so timeout termination can isolate it in a subprocess"
        )


def _worker_loop(child_conn: Any) -> None:
    """Subprocess worker loop executing one method call at a time under single-threaded BLAS."""
    for env_key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[env_key] = "1"
    # Pre-import heavy scientific modules during warm-up before any per-method timer begins
    import numpy as _np  # noqa: F401
    import scipy.optimize as _sp_opt  # noqa: F401

    child_conn.send(("ready", None))
    while True:
        try:
            msg = child_conn.recv()
        except EOFError:
            break
        if msg is None:
            break
        fn, args, kwargs = msg
        try:
            res = fn(*args, **kwargs)
            child_conn.send(("ok", res))
        except BaseException as exc:  # noqa: BLE001
            child_conn.send(("err", (type(exc).__name__, str(exc))))


class _PersistentProcessWorker:
    """Managed OS worker process supporting hard deadline terminate() + join() cleanup."""

    def __init__(self) -> None:
        self._ctx = mp.get_context("spawn")
        self._proc: Any = None
        self._parent_conn: Any = None

    def ensure_ready(self) -> None:
        if self._proc is not None and self._proc.is_alive() and self._parent_conn is not None:
            return
        self.close()
        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        proc = self._ctx.Process(target=_worker_loop, args=(child_conn,), daemon=False)
        proc.start()
        child_conn.close()
        if not parent_conn.poll(15.0):
            proc.terminate()
            proc.join(1.0)
            if proc.is_alive():
                proc.kill()
                proc.join(1.0)
            parent_conn.close()
            raise RuntimeError("Worker subprocess failed ready handshake within 15.0 s")
        status, _ = parent_conn.recv()
        if status != "ready":
            parent_conn.close()
            raise RuntimeError(f"Unexpected worker handshake status: {status!r}")
        self._proc = proc
        self._parent_conn = parent_conn

    def terminate_and_join(self) -> None:
        proc = self._proc
        conn = self._parent_conn
        self._proc = None
        self._parent_conn = None
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        if proc is not None:
            if proc.is_alive():
                proc.terminate()
                proc.join(0.25)
                if proc.is_alive():
                    proc.kill()
                    proc.join(0.5)
            else:
                proc.join(0.05)
            if proc.is_alive():
                raise RuntimeError("Timed-out worker subprocess survived terminate() and kill()")

    def close(self) -> None:
        proc = self._proc
        conn = self._parent_conn
        self._proc = None
        self._parent_conn = None
        if conn is not None:
            try:
                if proc is not None and proc.is_alive():
                    conn.send(None)
            except (OSError, BrokenPipeError):
                pass
            try:
                conn.close()
            except OSError:
                pass
        if proc is not None:
            proc.join(0.25)
            if proc.is_alive():
                proc.terminate()
                proc.join(0.25)
                if proc.is_alive():
                    proc.kill()
                    proc.join(0.25)

    def run_call(
        self, fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any], timeout_s: float
    ) -> tuple[str, Any, float, bool]:
        self.ensure_ready()
        assert self._parent_conn is not None
        t0 = time.perf_counter()
        self._parent_conn.send((fn, args, kwargs))
        if not self._parent_conn.poll(timeout_s):
            self.terminate_and_join()
            dt_ms = (time.perf_counter() - t0) * 1e3
            return "timeout", None, dt_ms, True
        try:
            tag, payload = self._parent_conn.recv()
        except EOFError:
            self.terminate_and_join()
            dt_ms = (time.perf_counter() - t0) * 1e3
            return "err", ("EOFError", "Worker closed connection unexpectedly"), dt_ms, True
        dt_ms = (time.perf_counter() - t0) * 1e3
        return tag, payload, dt_ms, False


_DEFAULT_WORKER = _PersistentProcessWorker()
atexit.register(_DEFAULT_WORKER.close)


def _extract_common_procedure_meta(pr: dict) -> dict[str, Any]:
    """Validate common discrete/IQ procedure inputs once before method dispatch."""
    required = ("step_idx", "grid_k", "ant_idx", "y", "q_i", "q_r")
    if not all(k in pr for k in required):
        return {
            "n_rows_input": None,
            "n_rows_admissible": None,
            "n_rows_discarded": None,
            "n_unique_usable_tones": None,
        }
    ok, stat, meta = _filter_and_validate_inputs(
        pr["step_idx"],
        pr["grid_k"],
        pr["ant_idx"],
        pr["y"],
        pr["q_i"],
        pr["q_r"],
        n_paths=int(pr.get("n_paths", 1)),
        subevent_idx=pr.get("subevent_idx"),
    )
    meta_d = meta if isinstance(meta, dict) else {}
    n_in = meta_d.get("n_rows_input", len(pr["y"]) if hasattr(pr["y"], "__len__") else None)
    return {
        "common_input_valid": bool(ok),
        "common_input_status": stat,
        "n_rows_input": n_in,
        "n_rows_admissible": meta_d.get("n_rows_admissible", 0 if (not ok and n_in == 0) else None),
        "n_rows_discarded": meta_d.get("n_rows_discarded", n_in if (not ok and n_in is not None) else None),
        "n_unique_usable_tones": meta_d.get("n_unique_usable_tones", 0 if stat == "zero_signal" else None),
    }


def _run_method_with_guard(
    fn: Callable[..., dict],
    timeout_s: float,
    *,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    common_meta: dict[str, Any] | None = None,
    worker: _PersistentProcessWorker | None = None,
) -> dict:
    """Execute one method call in an OS-terminable subprocess with timeout, exception, and metadata preservation."""
    kw = {} if kwargs is None else dict(kwargs)
    cm = {} if common_meta is None else common_meta
    safe_fn = _prepare_picklable_callable(fn)
    w = _DEFAULT_WORKER if worker is None else worker

    tag, payload, dt_ms, terminated = w.run_call(safe_fn, args, kw, float(timeout_s))
    base_counts = {
        "n_rows_input": cm.get("n_rows_input"),
        "n_rows_admissible": cm.get("n_rows_admissible"),
        "n_rows_discarded": cm.get("n_rows_discarded"),
        "n_unique_usable_tones": cm.get("n_unique_usable_tones"),
    }
    if tag == "timeout":
        return {
            "d_hat_m": float("nan"),
            "valid_estimate": False,
            "input_status": "timeout",
            "exception_type": "TimeoutError",
            "exception_message": f"Method exceeded {float(timeout_s):.3f}s deadline and worker was terminated",
            "worker_terminated": terminated,
            "qml_success": None,
            "elapsed_ms": dt_ms,
            **base_counts,
        }
    if tag == "err":
        exc_type, exc_msg = payload
        return {
            "d_hat_m": float("nan"),
            "valid_estimate": False,
            "input_status": "exception",
            "exception_type": exc_type,
            "exception_message": exc_msg,
            "worker_terminated": terminated,
            "qml_success": None,
            "elapsed_ms": dt_ms,
            **base_counts,
        }

    out = payload
    d_hat = float(out.get("d_hat_m", float("nan")))
    ok = bool(out.get("valid_estimate", False)) and math.isfinite(d_hat)
    stat = str(out.get("input_status", "invalid"))
    if bool(out.get("valid_estimate", False)) and not math.isfinite(d_hat):
        stat = "non_finite_output"

    rec: dict[str, Any] = {
        "d_hat_m": d_hat if ok else float("nan"),
        "valid_estimate": ok,
        "input_status": "valid" if ok else stat,
        "exception_type": None,
        "exception_message": None,
        "worker_terminated": False,
        "qml_success": None if out.get("qml_success") is None else bool(out["qml_success"]),
        "elapsed_ms": dt_ms,
        "n_rows_input": out.get("n_rows_input", base_counts["n_rows_input"]),
        "n_rows_admissible": out.get("n_rows_admissible", base_counts["n_rows_admissible"]),
        "n_rows_discarded": out.get("n_rows_discarded", base_counts["n_rows_discarded"]),
        "n_unique_usable_tones": out.get("n_unique_usable_tones", base_counts["n_unique_usable_tones"]),
    }
    for dk in _OPT_DIAG_KEYS:
        if dk in out:
            rec[dk] = out[dk]
    return rec


def _dispatch_prespecified_method(mname: str, pr: dict, d_cal_m: float) -> dict:
    """Invoke a prespecified estimator/baseline forwarding actual step_idx and optional subevent_idx."""
    dmx = float(pr["d_max_cohort_m"])
    np_cnt = int(pr.get("n_paths", 1))
    step_arr = pr["step_idx"]
    sub_arr = pr.get("subevent_idx")
    args = (pr["grid_k"], pr["ant_idx"], pr["y"], pr["q_i"], pr["q_r"])
    if mname == "SDR-ANM-QML":
        return estimate_range_sdr_anm_qml(
            step_arr, *args, n_paths=np_cnt, d_cal_m=d_cal_m, d_max_m=dmx, subevent_idx=sub_arr
        )
    if mname == "IFFT-Max":
        return baseline_ifft_argmax(
            *args, n_paths=np_cnt, d_cal_m=d_cal_m, d_max_m=dmx,
            step_idx=step_arr, apply_stage1_drift=False, subevent_idx=sub_arr, return_info=True,
        )
    if mname == "IFFT-LE":
        return baseline_ifft_leading_edge(
            *args, n_paths=np_cnt, d_cal_m=d_cal_m, d_max_m=dmx,
            step_idx=step_arr, apply_stage1_drift=False, subevent_idx=sub_arr, return_info=True,
        )
    if mname == "Adapted Kim-K15":
        return baseline_kim_k15(
            *args, n_paths=np_cnt, d_cal_m=d_cal_m, d_max_m=dmx,
            step_idx=step_arr, apply_stage1_drift=False, subevent_idx=sub_arr, return_info=True,
        )
    if mname == "Adapted Wieme-FBSS":
        return baseline_fbss_music(
            *args, n_paths=np_cnt, d_cal_m=d_cal_m, d_max_m=dmx, first_only=True,
            step_idx=step_arr, apply_stage1_drift=False, subevent_idx=sub_arr, return_info=True,
        )
    if mname == "Sheikh-WPS":
        return baseline_sheikh_wps(
            *args, n_paths=np_cnt, d_cal_m=d_cal_m, d_max_m=dmx,
            step_idx=step_arr, apply_stage1_drift=False, subevent_idx=sub_arr, return_info=True,
        )
    if mname == "Adapted Sheikh-h2-ANM":
        return baseline_sheikh_anm_music(
            *args, n_paths=np_cnt, d_cal_m=d_cal_m, d_max_m=dmx,
            step_idx=step_arr, apply_stage1_drift=False, subevent_idx=sub_arr, return_info=True,
        )
    if mname == "Longer-budget adapted Sheikh-h2-ANM":
        return baseline_sheikh_anm_music(
            *args, n_paths=np_cnt, d_cal_m=d_cal_m, d_max_m=dmx,
            step_idx=step_arr, apply_stage1_drift=False, subevent_idx=sub_arr,
            admm_max_iter=300, admm_tol=1e-4, return_info=True,
        )
    raise ValueError(f"Unsupported method: {mname!r}")


def _validate_expected_meta_dict(expected_proc_meta: dict[tuple[str, int], dict]) -> None:
    if not expected_proc_meta:
        raise ValueError("expected_proc_meta must be non-empty")
    for (fn, pidx), meta in expected_proc_meta.items():
        _validate_nonblank_str(fn, "expected_proc_meta.filename")
        _validate_nonneg_int(pidx, f"expected_proc_meta.proc_idx[{fn}]")
        _validate_nonblank_str(meta.get("geometry_id"), f"expected_proc_meta.geometry_id[{fn}, {pidx}]")
        _validate_finite_float(meta.get("d_true_m"), f"expected_proc_meta.d_true_m[{fn}, {pidx}]", allow_zero=True)
        _validate_finite_float(meta.get("d_max_cohort_m"), f"expected_proc_meta.d_max_cohort_m[{fn}, {pidx}]", allow_zero=False)


def validate_and_pivot_records(
    records: list[dict],
    expected_proc_meta: dict[tuple[str, int], dict],
    method_names: tuple[str, ...],
) -> dict:
    """Bind every (filename, proc_idx, method) record to independent expected metadata and check completeness."""
    _validate_expected_meta_dict(expected_proc_meta)
    expected_keys = {
        (fn, pidx, mname)
        for (fn, pidx) in expected_proc_meta
        for mname in method_names
    }
    by_key: dict[tuple[str, int, str], dict] = {}
    for rec in records:
        fn = _validate_nonblank_str(rec.get("filename"), "record.filename")
        pidx = _validate_nonneg_int(rec.get("proc_idx"), f"record.proc_idx[{fn}]")
        mname = _validate_nonblank_str(rec.get("method"), f"record.method[{fn}, {pidx}]")
        key = (fn, pidx, mname)
        if key in by_key:
            raise ValueError(f"Duplicate execution record identity: {key!r}")
        if key not in expected_keys:
            raise ValueError(f"Unexpected/foreign execution record identity: {key!r}")
        meta = expected_proc_meta[(fn, pidx)]
        rec_gid = _validate_nonblank_str(rec.get("geometry_id"), f"record.geometry_id{key!r}")
        if rec_gid != str(meta["geometry_id"]):
            raise ValueError(f"Geometry mismatch for {key!r}: {rec_gid} != {meta['geometry_id']}")
        rec_dmax = _validate_finite_float(rec.get("d_max_cohort_m"), f"record.d_max_cohort_m{key!r}", allow_zero=False)
        if abs(rec_dmax - float(meta["d_max_cohort_m"])) > 1e-9:
            raise ValueError(f"Cohort d_max mismatch for {key!r}: {rec_dmax} != {meta['d_max_cohort_m']}")
        rec_dtrue = _validate_finite_float(rec.get("d_true_m"), f"record.d_true_m{key!r}", allow_zero=True)
        if abs(rec_dtrue - float(meta["d_true_m"])) > 1e-9:
            raise ValueError(f"Ground truth mismatch for {key!r}: {rec_dtrue} != {meta['d_true_m']}")
        by_key[key] = rec

    missing = expected_keys - set(by_key.keys())
    if missing:
        raise ValueError(f"Missing {len(missing)} execution record(s); first={sorted(missing)[:3]}")

    proc_keys = sorted(expected_proc_meta.keys())
    cids = np.asarray([str(expected_proc_meta[pk]["geometry_id"]) for pk in proc_keys])
    penalties = np.asarray([float(expected_proc_meta[pk]["d_max_cohort_m"]) for pk in proc_keys], dtype=np.float64)

    method_arrays: dict[str, dict[str, np.ndarray]] = {}
    failure_accounting: dict[str, dict[str, Any]] = {}
    for mname in method_names:
        errs = np.empty(len(proc_keys), dtype=np.float64)
        signed_errs = np.empty(len(proc_keys), dtype=np.float64)
        vals = np.empty(len(proc_keys), dtype=bool)
        statuses: list[str] = []
        qml_flags: list[bool | None] = []
        for idx_p, (fn, pidx) in enumerate(proc_keys):
            r = by_key[(fn, pidx, mname)]
            is_v = bool(r["valid_estimate"]) and math.isfinite(float(r["d_hat_m"]))
            vals[idx_p] = is_v
            diff_signed = float(r["d_hat_m"]) - float(expected_proc_meta[(fn, pidx)]["d_true_m"]) if is_v else float("nan")
            signed_errs[idx_p] = diff_signed
            errs[idx_p] = abs(diff_signed) if is_v else float("nan")
            statuses.append(str(r["input_status"]))
            raw_q = r.get("qml_success")
            qml_flags.append(None if raw_q is None else bool(raw_q))
        has_qml = any(q is not None for q in qml_flags)
        method_arrays[mname] = {"abs_err": errs, "signed_err": signed_errs, "valid": vals}
        failure_accounting[mname] = {
            "n_total": len(proc_keys),
            "n_valid_finite": int(np.sum(vals)),
            "n_invalid_or_failed": int(len(proc_keys) - np.sum(vals)),
            "n_exception": sum(1 for s in statuses if s == "exception"),
            "n_timeout": sum(1 for s in statuses if s == "timeout"),
            "n_no_contiguous_subband": sum(1 for s in statuses if s == "no_contiguous_subband"),
            "n_insufficient_tones": sum(1 for s in statuses if s == "insufficient_tones"),
            "n_other_invalid": sum(
                1 for s in statuses
                if s not in ("valid", "exception", "timeout", "no_contiguous_subband", "insufficient_tones")
            ),
            "n_qml_success": sum(1 for q in qml_flags if q is True) if has_qml else None,
            "n_qml_nonconverged_finite_retained": (
                sum(1 for idx_p, q in enumerate(qml_flags) if (q is False) and vals[idx_p])
                if has_qml else None
            ),
        }

    return {
        "proc_keys": proc_keys,
        "cluster_ids": cids,
        "penalties_m": penalties,
        "method_arrays": method_arrays,
        "failure_accounting": failure_accounting,
    }


def _execute_verified_cohort_procedures(
    procedures: list[dict],
    expected_proc_meta: dict[tuple[str, int], dict],
    receipt_base: dict[str, Any],
    *,
    d_cal_m: float = D_CAL_SHARED_M,
    per_method_timeout_s: float = PER_METHOD_TIMEOUT_S,
    n_boot: int = 2000,
    boot_seed: int = 20261120,
    extra_method_hooks: dict[str, Callable[[dict], dict]] | None = None,
    _inprocess_mock_dispatch: Callable[[str, dict], dict] | None = None,
    include_nlos_summary: bool = False,
) -> dict:
    """Execute prespecified methods against an independent expected_proc_meta identity map."""
    _validate_expected_meta_dict(expected_proc_meta)
    seen_proc_keys: set[tuple[str, int]] = set()
    for pr in procedures:
        fn = _validate_nonblank_str(pr.get("filename"), "procedure.filename")
        pidx = _validate_nonneg_int(pr.get("proc_idx"), f"procedure.proc_idx[{fn}]")
        pk = (fn, pidx)
        if pk in seen_proc_keys:
            raise ValueError(f"Duplicate input procedure identity: {pk!r}")
        if pk not in expected_proc_meta:
            raise ValueError(f"Foreign procedure not registered in expected manifest cohort: {pk!r}")
        exp = expected_proc_meta[pk]
        if _validate_nonblank_str(pr.get("geometry_id"), f"procedure.geometry_id{pk!r}") != exp["geometry_id"]:
            raise ValueError(f"Procedure geometry mismatch against manifest for {pk!r}")
        dtrue = _validate_finite_float(pr.get("d_true_m"), f"procedure.d_true_m{pk!r}", allow_zero=True)
        dmax = _validate_finite_float(pr.get("d_max_cohort_m"), f"procedure.d_max_cohort_m{pk!r}", allow_zero=False)
        if abs(dtrue - exp["d_true_m"]) > 1e-9 or abs(dmax - exp["d_max_cohort_m"]) > 1e-9:
            raise ValueError(f"Procedure truth/bound mismatch against manifest for {pk!r}")
        seen_proc_keys.add(pk)

    if seen_proc_keys != set(expected_proc_meta.keys()):
        missing = sorted(set(expected_proc_meta.keys()) - seen_proc_keys)
        raise ValueError(f"Missing {len(missing)} expected manifest procedure(s); first={missing[:3]}")

    records: list[dict] = []
    worker = None if _inprocess_mock_dispatch is not None else _PersistentProcessWorker()
    try:
        for pr in procedures:
            pk = (str(pr["filename"]), int(pr["proc_idx"]))
            exp = expected_proc_meta[pk]
            common_meta = _extract_common_procedure_meta(pr)
            for mname in PRESPECIFIED_METHODS:
                if _inprocess_mock_dispatch is not None:
                    res = dict(_inprocess_mock_dispatch(mname, pr))
                elif extra_method_hooks and mname in extra_method_hooks:
                    res = _run_method_with_guard(
                        extra_method_hooks[mname],
                        per_method_timeout_s,
                        args=(pr,),
                        common_meta=common_meta,
                        worker=worker,
                    )
                else:
                    res = _run_method_with_guard(
                        _dispatch_prespecified_method,
                        per_method_timeout_s,
                        args=(mname, pr, float(d_cal_m)),
                        common_meta=common_meta,
                        worker=worker,
                    )
                records.append(
                    {
                        "filename": pk[0],
                        "proc_idx": pk[1],
                        "geometry_id": exp["geometry_id"],
                        "method": mname,
                        "d_true_m": exp["d_true_m"],
                        "d_max_cohort_m": exp["d_max_cohort_m"],
                        **res,
                    }
                )
    finally:
        if worker is not None:
            worker.close()

    pivoted = validate_and_pivot_records(records, expected_proc_meta, PRESPECIFIED_METHODS)
    m_arrs = pivoted["method_arrays"]
    stats = run_confirmatory_comparison(
        sdr_abs_err=m_arrs["SDR-ANM-QML"]["abs_err"],
        sdr_valid=m_arrs["SDR-ANM-QML"]["valid"],
        comp_abs_errs={c: m_arrs[c]["abs_err"] for c in PRESPECIFIED_COMPARATORS},
        comp_valids={c: m_arrs[c]["valid"] for c in PRESPECIFIED_COMPARATORS},
        cluster_ids=pivoted["cluster_ids"],
        penalty_m=pivoted["penalties_m"],
        comparators=PRESPECIFIED_COMPARATORS,
        n_boot=n_boot,
        seed=boot_seed,
    )
    nlos_sum = None
    if include_nlos_summary or ("nlos" in str(receipt_base.get("cohort_name", "")).lower()):
        nlos_sum = summarize_nlos_cohort(
            {m: m_arrs[m]["signed_err"] for m in PRESPECIFIED_METHODS},
            {m: m_arrs[m]["valid"] for m in PRESPECIFIED_METHODS},
            penalty_m=pivoted["penalties_m"],
        )
    receipt = {
        **receipt_base,
        "d_cal_m": float(d_cal_m),
        "per_method_timeout_s": float(per_method_timeout_s),
        "methods": PRESPECIFIED_METHODS,
        "comparators": PRESPECIFIED_COMPARATORS,
        "n_boot": int(n_boot),
        "boot_seed": int(boot_seed),
        "alpha": float(stats["alpha"]),
    }
    return {
        "receipt": receipt,
        "records": records,
        "failure_accounting": pivoted["failure_accounting"],
        "stats": stats,
        "nlos_summary": nlos_sum,
    }


def verify_cohort_source_and_load_procedures(
    manifest_path: str | pathlib.Path,
    data_dir: str | pathlib.Path,
    cohort_name: str,
) -> tuple[list[dict], dict[tuple[str, int], dict], dict[str, Any]]:
    """Verify ZIP/CSV hashes, procedure counts, truth agreement, and load only the cohort's first-15 procedures."""
    bundle = load_cohort_manifest_bundle(manifest_path, cohort_name)
    ddir = pathlib.Path(data_dir)
    zip_path = ddir if ddir.is_file() else (ddir / "data.zip")
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset archive not found: {zip_path}")

    zip_sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    if zip_sha != bundle["data_zip_sha256"]:
        raise ValueError(f"data.zip sha256 mismatch: got {zip_sha}, manifest has {bundle['data_zip_sha256']}")

    loader: ZenodoLoader | None = None
    if (ddir / "mapping.csv").exists() or (ddir / "outdoor_mapping.csv").exists():
        loader = ZenodoLoader(ddir, exclude_known_errata=False)

    procedures: list[dict] = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names_in_zip = set(zf.namelist())
            for row in bundle["rows"]:
                fn = row["filename"]
                if fn not in names_in_zip:
                    raise ValueError(f"Manifest file missing from zip: {fn!r}")
                csv_bytes = zf.read(fn)
                actual_sha = hashlib.sha256(csv_bytes).hexdigest()
                if actual_sha != row["csv_sha256"]:
                    raise ValueError(f"CSV sha256 mismatch for {fn}: {actual_sha} != {row['csv_sha256']}")

                nonempty_lines = [ln.strip() for ln in csv_bytes.decode("utf-8", errors="replace").splitlines() if ln.strip()]
                if len(nonempty_lines) != int(row["n_procedures"]):
                    raise ValueError(
                        f"Procedure count mismatch in {fn}: zip has {len(nonempty_lines)}, manifest has {row['n_procedures']}"
                    )

                if loader is not None:
                    loader_d = float(loader.get_true_distance(fn))
                    if abs(loader_d - float(row["d_true_m"])) > 1e-6:
                        raise ValueError(
                            f"Ground-truth distance mismatch for {fn}: mapping={loader_d}, manifest={row['d_true_m']}"
                        )
                    loaded_procs = loader.load_file_procedures(fn, max_procs=MAX_PROCS_PER_FILE)
                else:
                    loaded_procs = _parse_first_csv_lines_without_mapping(fn, nonempty_lines[:MAX_PROCS_PER_FILE], row)

                exp_count = min(MAX_PROCS_PER_FILE, int(row["n_procedures"]))
                if len(loaded_procs) != exp_count:
                    raise ValueError(f"Loaded {len(loaded_procs)} procedures for {fn}, expected {exp_count}")
                for idx_p, p_dict in enumerate(loaded_procs):
                    if int(p_dict["proc_idx"]) != idx_p:
                        raise ValueError(f"Non-sequential proc_idx in {fn}: got {p_dict['proc_idx']}, expected {idx_p}")
                    p_dict["geometry_id"] = row["geometry_id"]
                    p_dict["d_true_m"] = float(row["d_true_m"])
                    p_dict["d_max_cohort_m"] = float(row["d_max_cohort_m"])
                    procedures.append(p_dict)
    finally:
        if loader is not None:
            loader.close()

    receipt_base = {
        "execution_mode": "production_manifest_bound",
        "cohort_name": cohort_name,
        "inferential_role": bundle["spec"]["inferential_role"],
        "is_primary_confirmatory": cohort_name == "primary_tier1_indoor_los",
        "manifest_path": bundle["manifest_path"],
        "manifest_sha256": bundle["manifest_sha256"],
        "data_zip_sha256": zip_sha,
        "n_files": bundle["spec"]["n_files"],
        "n_clusters": bundle["spec"]["n_clusters"],
        "n_procedures": bundle["spec"]["n_procedures_first15"],
        "d_max_m": float(bundle["spec"]["d_max_m"]),
        "verified_csv_hashes_count": len(bundle["file_hashes"]),
    }
    return procedures, bundle["expected_proc_meta"], receipt_base


def _parse_first_csv_lines_without_mapping(fn: str, lines: list[str], row: dict) -> list[dict]:
    """Minimal CSV parser matching ZenodoLoader row schema for manifest-bound fixture testing."""
    procs: list[dict] = []
    for p_idx, line in enumerate(lines):
        steps_idx, grids, ants, y_vals, qi_vals, qr_vals = [], [], [], [], [], []
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
            steps_idx.append(idx)
            grids.append(ch - 2)
            ants.append(ant)
            y_vals.append(complex(ii, iq) * complex(ri, rq))
            qi_vals.append(q_i)
            qr_vals.append(q_r)
        procs.append(
            {
                "filename": fn,
                "proc_idx": p_idx,
                "step_idx": np.asarray(steps_idx, dtype=int),
                "grid_k": np.asarray(grids, dtype=int),
                "ant_idx": np.asarray(ants, dtype=int),
                "y": np.asarray(y_vals, dtype=np.complex128),
                "q_i": np.asarray(qi_vals, dtype=int),
                "q_r": np.asarray(qr_vals, dtype=int),
                "n_paths": 1,
                "d_true_m": float(row["d_true_m"]),
            }
        )
    return procs


def _compute_executable_pins() -> dict[str, str]:
    """Compute SHA-256 hashes of the executable modules and evaluation contract bound to a production receipt."""
    root = pathlib.Path(__file__).resolve().parents[1]
    targets = {
        "estimator_py": root / "src" / "estimator.py",
        "baselines_py": root / "src" / "baselines.py",
        "confirmatory_stats_py": root / "src" / "confirmatory_stats.py",
        "confirmatory_executor_py": root / "src" / "confirmatory_executor.py",
        "evaluation_contract_md": root / "notes" / "evaluation_contract.md",
    }
    return {
        k: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "missing"
        for k, p in targets.items()
    }


def validate_frozen_production_protocol(
    manifest_path: str | pathlib.Path,
    cohort_name: str,
    *,
    d_cal_m: float,
    per_method_timeout_s: float,
    n_boot: int,
    boot_seed: int,
    expected_manifest_sha256: str = FROZEN_MANIFEST_SHA256,
    expected_zip_sha256: str = FROZEN_DATA_ZIP_SHA256,
) -> dict:
    """Validate manifest hash, frozen cohort spec, calibration, timeout, bootstrap, and method pins before IQ load."""
    if cohort_name not in _FROZEN_COHORT_SPECS or cohort_name not in COHORT_SPECS:
        raise ValueError(f"Unregistered cohort_name for frozen execution: {cohort_name!r}")
    f_nfiles, f_ncl, f_nprocs, f_dmax, f_role = _FROZEN_COHORT_SPECS[cohort_name]
    cur_spec = COHORT_SPECS[cohort_name]
    if (
        cur_spec.get("n_files") != f_nfiles
        or cur_spec.get("n_clusters") != f_ncl
        or cur_spec.get("n_procedures_first15") != f_nprocs
        or abs(float(cur_spec.get("d_max_m", -1.0)) - f_dmax) > 1e-9
        or cur_spec.get("inferential_role") != f_role
    ):
        raise ValueError(f"COHORT_SPECS[{cohort_name!r}] mutated from frozen spec")
    if PRESPECIFIED_METHODS != _FROZEN_METHODS or PRESPECIFIED_COMPARATORS != _FROZEN_COMPARATORS:
        raise ValueError("Method or comparator configuration mutated from frozen protocol")

    d_cal_val = _validate_finite_float(d_cal_m, "d_cal_m", allow_zero=True)
    if abs(d_cal_val - FROZEN_D_CAL_M) > 1e-12:
        raise ValueError(f"Frozen d_cal_m mismatch: got {d_cal_val}, expected {FROZEN_D_CAL_M}")
    to_val = _validate_finite_float(per_method_timeout_s, "per_method_timeout_s", allow_zero=False)
    if abs(to_val - FROZEN_PER_METHOD_TIMEOUT_S) > 1e-12:
        raise ValueError(
            f"Frozen per_method_timeout_s mismatch: got {to_val}, expected {FROZEN_PER_METHOD_TIMEOUT_S}"
        )
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot != FROZEN_N_BOOT:
        raise ValueError(f"Frozen n_boot mismatch: got {n_boot!r}, expected {FROZEN_N_BOOT}")
    if isinstance(boot_seed, bool) or not isinstance(boot_seed, int) or boot_seed != FROZEN_BOOT_SEED:
        raise ValueError(f"Frozen boot_seed mismatch: got {boot_seed!r}, expected {FROZEN_BOOT_SEED}")

    bundle = load_cohort_manifest_bundle(manifest_path, cohort_name)
    if bundle["manifest_sha256"] != expected_manifest_sha256:
        raise ValueError(
            f"Frozen manifest sha256 mismatch: got {bundle['manifest_sha256']}, "
            f"expected {expected_manifest_sha256}"
        )
    if bundle["data_zip_sha256"] != expected_zip_sha256:
        raise ValueError(
            f"Frozen data_zip sha256 mismatch: got {bundle['data_zip_sha256']}, "
            f"expected {expected_zip_sha256}"
        )
    return bundle


def _run_production_cohort_pipeline(
    manifest_path: str | pathlib.Path,
    data_dir: str | pathlib.Path,
    cohort_name: str,
    *,
    d_cal_m: float,
    per_method_timeout_s: float,
    n_boot: int,
    boot_seed: int,
    _iq_loader: Callable[[str | pathlib.Path, str | pathlib.Path, str], tuple[list[dict], dict[tuple[str, int], dict], dict[str, Any]]] | None = None,
    _inprocess_mock_dispatch: Callable[[str, dict], dict] | None = None,
) -> dict:
    """Enforce frozen-protocol validation prior to ANY IQ loading callback, then run & bind receipt."""
    bundle = validate_frozen_production_protocol(
        manifest_path,
        cohort_name,
        d_cal_m=d_cal_m,
        per_method_timeout_s=per_method_timeout_s,
        n_boot=n_boot,
        boot_seed=boot_seed,
    )
    loader_fn = verify_cohort_source_and_load_procedures if _iq_loader is None else _iq_loader
    procedures, expected_proc_meta, receipt_base = loader_fn(manifest_path, data_dir, cohort_name)
    if receipt_base.get("manifest_sha256") != bundle["manifest_sha256"]:
        raise ValueError("Loaded receipt manifest_sha256 does not match pre-validated frozen bundle")
    if set(expected_proc_meta.keys()) != set(bundle["expected_proc_meta"].keys()):
        raise ValueError("Loader expected_proc_meta keys differ from frozen manifest bundle")
    receipt_enriched = {
        **receipt_base,
        "frozen_protocol_id": FROZEN_PROTOCOL_ID,
        "frozen_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "frozen_data_zip_sha256": FROZEN_DATA_ZIP_SHA256,
        "freeze_checks_passed": True,
        "adapter_mode": "injected_test_adapter" if (_iq_loader is not None or _inprocess_mock_dispatch is not None) else "production_real_archive",
        "executable_pins": _compute_executable_pins(),
    }
    return _execute_verified_cohort_procedures(
        procedures,
        bundle["expected_proc_meta"],
        receipt_enriched,
        d_cal_m=d_cal_m,
        per_method_timeout_s=per_method_timeout_s,
        n_boot=n_boot,
        boot_seed=boot_seed,
        extra_method_hooks=None,
        _inprocess_mock_dispatch=_inprocess_mock_dispatch,
    )


def execute_confirmatory_cohort(
    manifest_path: str | pathlib.Path,
    data_dir: str | pathlib.Path,
    cohort_name: str = "primary_tier1_indoor_los",
    d_cal_m: float = FROZEN_D_CAL_M,
    per_method_timeout_s: float = FROZEN_PER_METHOD_TIMEOUT_S,
    n_boot: int = FROZEN_N_BOOT,
    boot_seed: int = FROZEN_BOOT_SEED,
) -> dict:
    """Production entry point enforcing frozen manifest hash and registered settings before any IQ load."""
    return _run_production_cohort_pipeline(
        manifest_path,
        data_dir,
        cohort_name,
        d_cal_m=d_cal_m,
        per_method_timeout_s=per_method_timeout_s,
        n_boot=n_boot,
        boot_seed=boot_seed,
    )


def execute_procedures_and_analyze(
    procedures: list[dict],
    d_cal_m: float = D_CAL_SHARED_M,
    per_method_timeout_s: float = PER_METHOD_TIMEOUT_S,
    extra_method_hooks: dict[str, Callable[[dict], dict]] | None = None,
    n_boot: int = 2000,
    boot_seed: int = 20261120,
    expected_proc_meta: dict[tuple[str, int], dict] | None = None,
    include_nlos_summary: bool = False,
) -> dict:
    """Fixture/test-path executor; non-confirmatory unless supplied independent manifest expected_proc_meta."""
    if expected_proc_meta is None:
        derived_meta: dict[tuple[str, int], dict] = {}
        for pr in procedures:
            fn = _validate_nonblank_str(pr.get("filename"), "procedure.filename")
            pidx = _validate_nonneg_int(pr.get("proc_idx"), f"procedure.proc_idx[{fn}]")
            pk = (fn, pidx)
            if pk in derived_meta:
                raise ValueError(f"Duplicate input procedure identity: {pk!r}")
            derived_meta[pk] = {
                "geometry_id": _validate_nonblank_str(pr.get("geometry_id"), f"procedure.geometry_id{pk!r}"),
                "d_true_m": _validate_finite_float(pr.get("d_true_m"), f"procedure.d_true_m{pk!r}", allow_zero=True),
                "d_max_cohort_m": _validate_finite_float(pr.get("d_max_cohort_m"), f"procedure.d_max_cohort_m{pk!r}", allow_zero=False),
            }
        expected_meta = derived_meta
        receipt_base = {
            "execution_mode": "test_fixture_non_confirmatory",
            "cohort_name": None,
            "inferential_role": "non_confirmatory_test_fixture",
            "is_primary_confirmatory": False,
            "has_test_hooks": bool(extra_method_hooks),
        }
    else:
        expected_meta = expected_proc_meta
        receipt_base = {
            "execution_mode": "bound_expected_meta_test" if extra_method_hooks else "bound_expected_meta",
            "cohort_name": None,
            "inferential_role": "test_verified_subset" if extra_method_hooks else "bound_cohort_subset",
            "is_primary_confirmatory": False,
            "has_test_hooks": bool(extra_method_hooks),
        }
    return _execute_verified_cohort_procedures(
        procedures,
        expected_meta,
        receipt_base,
        d_cal_m=d_cal_m,
        per_method_timeout_s=per_method_timeout_s,
        n_boot=n_boot,
        boot_seed=boot_seed,
        extra_method_hooks=extra_method_hooks,
        include_nlos_summary=include_nlos_summary,
    )


# Picklable top-level smoke workers so run_smoke_test exercises real OS subprocess termination
def _smoke_slow_finite_worker() -> dict:
    time.sleep(0.25)
    return {"d_hat_m": 1.0, "valid_estimate": True, "input_status": "valid"}


def _smoke_nonterminating_worker() -> dict:
    while True:
        time.sleep(0.02)


def _smoke_le_hook(p: dict) -> dict:
    if p["proc_idx"] == 9:
        raise RuntimeError("injected exception")
    if p["proc_idx"] == 10:
        time.sleep(1.5)
        return {"d_hat_m": 1.0, "valid_estimate": True, "input_status": "valid"}
    return baseline_ifft_leading_edge(
        p["grid_k"], p["ant_idx"], p["y"], p["q_i"], p["q_r"],
        n_paths=1, d_cal_m=0.0, d_max_m=45.0,
        step_idx=p["step_idx"], apply_stage1_drift=False, subevent_idx=p.get("subevent_idx"),
        return_info=True,
    )


def _smoke_sdr_hook(p: dict) -> dict:
    res = estimate_range_sdr_anm_qml(
        p["step_idx"], p["grid_k"], p["ant_idx"], p["y"], p["q_i"], p["q_r"],
        n_paths=1, d_cal_m=0.0, d_max_m=45.0, subevent_idx=p.get("subevent_idx"),
    )
    if p["proc_idx"] == 1 and res["valid_estimate"]:
        res["qml_success"] = False  # verify finite non-converged is retained, NOT penalized
    return res


def run_smoke_test() -> None:
    """Exercise OS-terminating timeouts, manifest/hash/metadata rejections, fractional steps, and stats checks."""
    import tempfile

    root = pathlib.Path(__file__).resolve().parents[1]
    manifest = root / "notes" / "zenodo_partition_manifest.json"
    if manifest.exists():
        bundle = load_cohort_manifest_bundle(manifest, "primary_tier1_indoor_los")
        assert bundle["spec"]["n_files"] == 78 and len(bundle["expected_proc_meta"]) == 1170

    # 1. Verify hard terminating timeout on BOTH slow finite worker AND infinite non-terminating worker
    w = _PersistentProcessWorker()
    try:
        w.ensure_ready()
        t0 = time.perf_counter()
        rec_slow = _run_method_with_guard(_smoke_slow_finite_worker, timeout_s=0.04, worker=w)
        ext_slow = time.perf_counter() - t0
        assert rec_slow["input_status"] == "timeout" and rec_slow["worker_terminated"] is True
        assert 0.03 < ext_slow < 0.16 and 30.0 < rec_slow["elapsed_ms"] < 160.0, (ext_slow, rec_slow["elapsed_ms"])

        w.ensure_ready()
        t1 = time.perf_counter()
        rec_inf = _run_method_with_guard(_smoke_nonterminating_worker, timeout_s=0.04, worker=w)
        ext_inf = time.perf_counter() - t1
        assert rec_inf["input_status"] == "timeout" and rec_inf["worker_terminated"] is True
        assert 0.03 < ext_inf < 0.16 and w._proc is None, (ext_inf, w._proc)
    finally:
        w.close()

    # 2. Full fixture execution smoke (valid, 1-tone, WPS-no-subband, exception, slow timeout, non-conv finite)
    rng = np.random.default_rng(42)
    mask = make_mask("wifi_ch6")
    cids = ["G0", "G0", "G0", "G1", "G1", "G2", "G2", "G2", "G2", "G3", "G3", "G3"]
    procs: list[dict] = []
    for idx, gid in enumerate(cids):
        d_true = 4.0 + 0.8 * idx
        taus = np.array([d_true, d_true + 4.5, d_true + 9.0]) / C_LIGHT
        alphas = np.array([[1.0 + 0j, 0.7 - 0.3j, 0.4 + 0.2j]])
        sim = simulate_cs_procedure(taus, alphas, mask, snr_db=20.0, rng=rng)
        procs.append(
            {
                "filename": f"fab/{gid}.csv",
                "proc_idx": idx,
                "geometry_id": gid,
                "d_true_m": d_true,
                "d_max_cohort_m": 45.0,
                **sim,
            }
        )

    procs[2]["grid_k"][:3] = 75  # mixed out-of-range
    ks_scat = np.array([0, 2, 4, 6, 8, 10, 12, 14] * 19 + [0, 2, 4, 6, 8, 10, 12])
    procs[5]["grid_k"] = ks_scat
    procs[5]["y"] = np.exp(-2j * np.pi * 1e6 * ks_scat * (2.0 * procs[5]["d_true_m"] / C_LIGHT))
    for key_z in ("grid_k", "step_idx", "ant_idx", "q_i", "q_r", "y", "subevent_idx"):
        procs[7][key_z] = (np.ones(20, complex) if key_z == "y" else np.zeros(20, int))
    procs[7]["step_idx"] = np.arange(20, dtype=int)  # 1-tone

    res = execute_procedures_and_analyze(
        procs,
        d_cal_m=0.0,
        per_method_timeout_s=1.0,
        extra_method_hooks={"IFFT-LE": _smoke_le_hook, "SDR-ANM-QML": _smoke_sdr_hook},
        n_boot=200,
    )
    assert res["receipt"]["is_primary_confirmatory"] is False
    fa = res["failure_accounting"]
    for mn in ("SDR-ANM-QML", "IFFT-LE", "Sheikh-WPS"):
        print(" ", mn, fa[mn])
    assert fa["SDR-ANM-QML"]["n_insufficient_tones"] == 1 and fa["SDR-ANM-QML"]["n_valid_finite"] == 11
    assert fa["SDR-ANM-QML"]["n_qml_nonconverged_finite_retained"] >= 1
    assert fa["IFFT-LE"]["n_exception"] == 1 and fa["IFFT-LE"]["n_timeout"] == 1
    assert fa["Sheikh-WPS"]["n_no_contiguous_subband"] == 1 and fa["Sheikh-WPS"]["n_insufficient_tones"] == 1
    # Verify common row metadata preserved on timeout & exception AND n_unique_usable_tones exported for SDR
    rec_sdr0 = next(r for r in res["records"] if r["proc_idx"] == 0 and r["method"] == "SDR-ANM-QML")
    rec_le_to = next(r for r in res["records"] if r["proc_idx"] == 10 and r["method"] == "IFFT-LE")
    assert isinstance(rec_sdr0["n_unique_usable_tones"], int) and rec_sdr0["n_unique_usable_tones"] > 10
    assert rec_le_to["n_rows_input"] == len(procs[10]["y"]) and rec_le_to["worker_terminated"] is True

    # 3. Fractional step_idx rejected uniformly across ALL 7 methods, plus subevent_idx forwarding
    sim_frac = simulate_cs_procedure(
        np.array([5.0, 8.5, 13.0]) / C_LIGHT,
        np.array([[1.0 + 0.2j, 0.7 - 0.3j, 0.4 + 0.2j]]),
        mask, snr_db=20.0, rng=np.random.default_rng(11),
    )
    f_proc = {
        "filename": "fab/frac.csv",
        "proc_idx": 0,
        "geometry_id": "GF",
        "d_true_m": 5.0,
        "d_max_cohort_m": 45.0,
        **sim_frac,
        "step_idx": sim_frac["step_idx"].astype(float),
        "subevent_idx": np.zeros_like(sim_frac["step_idx"]),
    }
    f_proc["step_idx"][0] += 0.5
    res_frac = execute_procedures_and_analyze([f_proc], d_cal_m=0.0, per_method_timeout_s=5.0, n_boot=20)
    assert all((not r["valid_estimate"]) and r["input_status"] == "non_integer_index" for r in res_frac["records"])

    # 3b. Verify NLOS signed excess-delay & outlier summary + 8th Longer-budget comparator telemetry
    res_nlos = execute_procedures_and_analyze(
        procs[:3], d_cal_m=0.0, per_method_timeout_s=2.0, n_boot=20, include_nlos_summary=True,
    )
    assert "Longer-budget adapted Sheikh-h2-ANM" in res_nlos["failure_accounting"]
    assert res_nlos["nlos_summary"] is not None and "signed_bias_median_m" in res_nlos["nlos_summary"]["SDR-ANM-QML"]

    # 4. Duplicate, missing, foreign, NaN-truth rejections + manifest/ZIP hash & foreign/omitted checks
    meta_map = {
        (f"fab/{cid}.csv", i): {"geometry_id": cid, "d_true_m": 4.0 + 0.8 * i, "d_max_cohort_m": 45.0}
        for i, cid in enumerate(cids)
    }
    for mut in (
        lambda recs: recs + [recs[0]],
        lambda recs: recs[:-1],
        lambda recs: [{**recs[0], "d_true_m": float("nan")}] + recs[1:],
        lambda recs: [{**recs[0], "filename": "fab/foreign.csv"}] + recs[1:],
    ):
        caught = False
        try:
            validate_and_pivot_records(mut(res["records"]), meta_map, PRESPECIFIED_METHODS)
        except ValueError:
            caught = True
        assert caught

    # 5. Pre-IQ freeze guard: reject EVERY manifest/setting mismatch BEFORE invoking IQ loader callback
    iq_called = [0]

    def _sentinel_iq_loader(_mp: object, _dd: object, _cn: str) -> tuple[list[dict], dict, dict]:
        iq_called[0] += 1
        raise AssertionError("IQ loader should not be called on frozen-protocol mismatch")

    if manifest.exists():
        # Every setting deviation rejects BEFORE touching _sentinel_iq_loader
        for bad_kw in (
            {"d_cal_m": 0.0},
            {"per_method_timeout_s": 1.0},
            {"n_boot": 20},
            {"boot_seed": 1},
        ):
            hit_set = False
            try:
                _run_production_cohort_pipeline(
                    manifest,
                    root / "data" / "zenodo_17347695",
                    "primary_tier1_indoor_los",
                    d_cal_m=bad_kw.get("d_cal_m", FROZEN_D_CAL_M),
                    per_method_timeout_s=bad_kw.get("per_method_timeout_s", FROZEN_PER_METHOD_TIMEOUT_S),
                    n_boot=bad_kw.get("n_boot", FROZEN_N_BOOT),
                    boot_seed=bad_kw.get("boot_seed", FROZEN_BOOT_SEED),
                    _iq_loader=_sentinel_iq_loader,
                )
            except ValueError:
                hit_set = True
            assert hit_set and iq_called[0] == 0, bad_kw

    # Modified manifest metadata copy (truth + 1.0 m) also rejected BEFORE _sentinel_iq_loader
    with tempfile.TemporaryDirectory() as tmpd:
        td = pathlib.Path(tmpd)
        if manifest.exists():
            real_obj = json.loads(manifest.read_text(encoding="utf-8"))
            for row in real_obj["files"]:
                if row["partition"] == "confirmatory_tier1_unseen_geometry" and row["los_nlos"] == "LOS" and not row["filename"].startswith("outdoor/"):
                    row["d_true_m"] += 1.0
                    break
            mod_mpath = td / "modified_manifest.json"
            mod_mpath.write_text(json.dumps(real_obj), encoding="utf-8")
            hit_mod = False
            try:
                _run_production_cohort_pipeline(
                    mod_mpath,
                    root / "data" / "zenodo_17347695",
                    "primary_tier1_indoor_los",
                    d_cal_m=FROZEN_D_CAL_M,
                    per_method_timeout_s=FROZEN_PER_METHOD_TIMEOUT_S,
                    n_boot=FROZEN_N_BOOT,
                    boot_seed=FROZEN_BOOT_SEED,
                    _iq_loader=_sentinel_iq_loader,
                )
            except ValueError:
                hit_mod = True
            assert hit_mod and iq_called[0] == 0

            # Exact untouched frozen configuration accepted cleanly via fabricated metadata-only adapter (ZERO real ZIP read!)
            def _fab_frozen_loader(_mp: object, _dd: object, _cn: str) -> tuple[list[dict], dict, dict]:
                bnd = load_cohort_manifest_bundle(manifest, "primary_tier1_indoor_los")
                fab_procs = [
                    {
                        "filename": fn,
                        "proc_idx": pidx,
                        "geometry_id": m["geometry_id"],
                        "d_true_m": m["d_true_m"],
                        "d_max_cohort_m": m["d_max_cohort_m"],
                    }
                    for (fn, pidx), m in sorted(bnd["expected_proc_meta"].items())
                ]
                return fab_procs, bnd["expected_proc_meta"], {
                    "execution_mode": "production_manifest_bound",
                    "cohort_name": "primary_tier1_indoor_los",
                    "inferential_role": "primary_confirmatory",
                    "is_primary_confirmatory": True,
                    "manifest_sha256": bnd["manifest_sha256"],
                    "data_zip_sha256": bnd["data_zip_sha256"],
                    "n_files": 78,
                    "n_clusters": 74,
                    "n_procedures": 1170,
                }

            frozen_res = _run_production_cohort_pipeline(
                manifest,
                root / "data" / "zenodo_17347695",
                "primary_tier1_indoor_los",
                d_cal_m=FROZEN_D_CAL_M,
                per_method_timeout_s=FROZEN_PER_METHOD_TIMEOUT_S,
                n_boot=FROZEN_N_BOOT,
                boot_seed=FROZEN_BOOT_SEED,
                _iq_loader=_fab_frozen_loader,
                _inprocess_mock_dispatch=lambda _m, pr: {"d_hat_m": pr["d_true_m"], "valid_estimate": True, "input_status": "valid", "qml_success": None},
            )
            assert (
                frozen_res["receipt"]["is_primary_confirmatory"] is True
                and frozen_res["receipt"]["freeze_checks_passed"] is True
                and frozen_res["receipt"]["frozen_protocol_id"] == FROZEN_PROTOCOL_ID
                and frozen_res["stats"]["n_procedures"] == 1170
                and frozen_res["stats"]["n_clusters"] == 74
                and frozen_res["stats"]["n_boot"] == 2000
            )

        # Mini-archive hash & foreign checks
        csv_line = ",".join(f"0;0;{int(procs[0]['grid_k'][i] + 2)};0;10;0;10;0" for i in range(4)) + "\n"
        zpath = td / "data.zip"
        with zipfile.ZipFile(zpath, "w") as zf:
            zf.writestr("mock/g0.csv", csv_line)
        zsha = hashlib.sha256(zpath.read_bytes()).hexdigest()
        csha = hashlib.sha256(csv_line.encode("utf-8")).hexdigest()
        m_obj = {
            "data_zip_sha256": zsha,
            "files": [
                {
                    "filename": "mock/g0.csv",
                    "csv_sha256": csha,
                    "n_procedures": 1,
                    "geometry_id": "G0",
                    "d_true_m": 5.0,
                    "d_max_cohort_m": 45.0,
                    "los_nlos": "LOS",
                    "partition": "confirmatory_tier1_unseen_geometry",
                }
            ],
        }
        mpath = td / "manifest.json"
        for bad_sha_case, mut_fn in [
            ("spec_count", lambda o: o),
            ("bad_csv_hash", lambda o: {**o, "files": [{**o["files"][0], "csv_sha256": "0" * 64}]}),
        ]:
            mpath.write_text(json.dumps(mut_fn(m_obj)), encoding="utf-8")
            hit = False
            try:
                load_cohort_manifest_bundle(mpath, "primary_tier1_indoor_los")
                verify_cohort_source_and_load_procedures(mpath, td, "primary_tier1_indoor_los")
            except ValueError:
                hit = True
            assert hit, bad_sha_case
        exp_one = {("mock/g0.csv", 0): {"geometry_id": "G0", "d_true_m": 5.0, "d_max_cohort_m": 45.0}}
        foreign_caught = False
        try:
            _execute_verified_cohort_procedures([procs[0]], exp_one, {"execution_mode": "test"})
        except ValueError:
            foreign_caught = True
        assert foreign_caught

    print("Confirmatory executor smoke test PASSED (freeze guard, OS-kill timeouts, manifest/hash/truth/cluster/fractional/subevent).")


if __name__ == "__main__":
    run_smoke_test()
