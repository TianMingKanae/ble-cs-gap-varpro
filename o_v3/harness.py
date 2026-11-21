"""Run estimators on cached procedures in parallel; save per-procedure d_hat."""
import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import importlib, json, math, pathlib, pickle, sys, time
import numpy as np
from multiprocessing import Pool
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
D_CAL = 0.881149670238504


def _method(name):
    from src import baselines as B
    from src.estimator_v2_1 import estimate_range_dc_anm_varpro_v2_1 as V21
    if name == "v2.1":
        return lambda p: V21(p["step_idx"], p["grid_k"], p["ant_idx"], p["y"], p["q_i"], p["q_r"], d_cal_m=D_CAL, d_max_m=p["d_max_m"])
    if name == "IFFT-LE":
        return lambda p: B.baseline_ifft_leading_edge(p["grid_k"], p["ant_idx"], p["y"], p["q_i"], p["q_r"], d_cal_m=D_CAL, d_max_m=p["d_max_m"])
    if name == "Sheikh-WPS":
        return lambda p: B.baseline_sheikh_wps(p["grid_k"], p["ant_idx"], p["y"], p["q_i"], p["q_r"], d_cal_m=D_CAL, d_max_m=p["d_max_m"])
    if name == "Wieme-FBSS":
        return lambda p: B.baseline_fbss_music(p["grid_k"], p["ant_idx"], p["y"], p["q_i"], p["q_r"], d_cal_m=D_CAL, d_max_m=p["d_max_m"], first_only=True)
    mod, fn = name.split(":")  # e.g. o_v3.est_v3:estimate  (kwargs via env O_V3_KW json)
    f = getattr(importlib.import_module(mod), fn)
    kw = json.loads(os.environ.get("O_V3_KW", "{}"))
    return lambda p: f(p, d_cal_m=D_CAL, **kw)


_F = None
def _init(name):
    global _F
    _F = _method(name)

def _one(p):
    t0 = time.perf_counter()
    try:
        r = _F(p)
        d = float(r["d_hat_m"]) if isinstance(r, dict) else float(r)
        info = {k: r[k] for k in r if k in ("atom_delays_2way_ns", "atom_amplitudes_l2", "sel_idx", "flag")} if isinstance(r, dict) else {}
    except Exception as e:  # keep failures
        d, info = float("nan"), {"exc": repr(e)}
    return d, (time.perf_counter() - t0) * 1e3, info


def run(name, procs, nproc=14):
    with Pool(nproc, initializer=_init, initargs=(name,)) as pool:
        res = pool.map(_one, procs, chunksize=8)
    return res


def summarize(procs, dh, label=""):
    rows = {}
    for c in sorted(set(p["cohort"] for p in procs)):
        idx = [i for i, p in enumerate(procs) if p["cohort"] == c]
        e = np.array([dh[i] - procs[i]["d_true_m"] if math.isfinite(dh[i]) else procs[i]["d_max_m"] for i in idx])
        a = np.abs(e)
        rows[c] = dict(n=len(idx), mae=a.mean(), p90=np.quantile(a, .9), gt5=(a > 5).mean(), med=np.median(e))
    return rows


if __name__ == "__main__":
    name = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else name.replace(":", "_").replace("/", "_")
    procs = pickle.load(open(ROOT / "o_v3" / "burned_cache.pkl", "rb"))
    t0 = time.time()
    res = run(name, procs)
    dh = [r[0] for r in res]
    pickle.dump({"name": name, "kw": os.environ.get("O_V3_KW", "{}"), "res": res}, open(ROOT / "o_v3" / f"res_{tag}.pkl", "wb"))
    print(name, tag, "wall %.0fs" % (time.time() - t0))
    for c, s in summarize(procs, dh).items():
        print(f"  {c:16s} n={s['n']:5d} MAE {s['mae']:.3f} P90 {s['p90']:.3f} >5m {s['gt5']:.3f} med {s['med']:+.3f}")
