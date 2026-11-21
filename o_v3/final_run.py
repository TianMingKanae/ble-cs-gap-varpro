"""Registered fresh-data run (O v3, 11-21). Loads held-out procedures >= 15 (guarded), runs 5 methods, saves raw."""
import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import pathlib, pickle, sys, time, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]
METHODS = ["v2.5", "v3", "IFFT-LE", "Sheikh-WPS", "Wieme-FBSS"]
NAME = {"v2.5": "v2.1", "v3": "o_v3.est_v3:estimate_v3"}
EXPECT = {'t1_indoor_los': 6630, 't1_indoor_nlos': 6800, 't1_outdoor_los': 220, 't2_indoor_los': 580, 't2_indoor_nlos': 1685}

if __name__ == "__main__":
    from o_v3.data_guard import load
    from o_v3.harness import run
    procs = load("fresh")
    cnt = collections.Counter(p["cohort"] for p in procs)
    assert dict(cnt) == EXPECT, cnt
    assert all(p["proc_idx"] >= 15 for p in procs)
    out = {"meta": [{k: p[k] for k in ("file", "proc_idx", "geometry_id", "d_true_m", "d_max_m", "cohort")} for p in procs], "res": {}}
    for m in METHODS:
        t0 = time.time()
        out["res"][m] = run(NAME.get(m, m), procs)
        print(m, "wall %.0fs" % (time.time() - t0), flush=True)
    pickle.dump(out, open(ROOT / "o_v3" / "final_raw.pkl", "wb"))
    print("saved o_v3/final_raw.pkl", len(procs))
