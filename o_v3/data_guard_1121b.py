# 1121b: successor of data_guard.py (registered source kept unchanged); rejects selectors other than burned/fresh (C 11-21 finding 2).
"""O v3 data access with a hard guard on the never-scored procedures.

Burned data (may be used for development):
  - calibration + development files: all procedures
  - held-out files: procedures 0..14 (scored by v1, v2, v2.1 before 11-21)
Fresh data (final registered test only):
  - held-out files: procedures >= 15. Never loaded by any earlier script (all used max_procs=15).
Loading fresh data requires env O_V3_FINAL=<sha256 of the registration file>.
"""
import hashlib, json, os, pathlib, pickle, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import ZenodoLoader  # noqa: E402

MAN = json.load(open(ROOT / "notes" / "zenodo_partition_manifest.json", encoding="utf-8"))
HELD = {"confirmatory_tier1_unseen_geometry", "confirmatory_tier2_shared_geometry_new_config"}
REG = ROOT / "o_v3" / "REGISTRATION_final.md"


def cohort_of(f):
    fn, part = f["filename"], f["partition"]
    if part not in HELD:
        return part
    tier = "t1" if part.startswith("confirmatory_tier1") else "t2"
    if fn.startswith("outdoor/"):
        return tier + "_outdoor_los"
    return f"{tier}_indoor_{f['los_nlos'].lower()}"


def load(which="burned"):
    if which not in ("burned", "fresh"):
        raise ValueError(f"unknown selector {which!r}")
    ds = ZenodoLoader(ROOT / "data" / "zenodo_17347695")
    if which == "fresh":
        sha = hashlib.sha256(REG.read_bytes()).hexdigest() if REG.exists() else None
        if not sha or os.environ.get("O_V3_FINAL") != sha:
            raise SystemExit("fresh data locked: registration missing or hash mismatch")
    out = []
    for f in MAN["files"]:
        part = f["partition"]
        if part == "excluded_discrepancy":
            continue
        held = part in HELD
        if which == "burned":
            procs = ds.load_file_procedures(f["filename"], max_procs=15 if held else None)
        else:
            if not held:
                continue
            procs = ds.load_file_procedures(f["filename"])[15:]
        for p in procs:
            out.append({k: p[k] for k in ("step_idx", "grid_k", "ant_idx", "y", "z_i", "z_r", "q_i", "q_r", "proc_idx")} | {
                "file": f["filename"], "geometry_id": f["geometry_id"], "d_true_m": float(f["d_true_m"]),
                "d_max_m": float(f["d_max_cohort_m"]), "cohort": cohort_of(f), "partition": part, "los": f["los_nlos"]})
    ds.close()
    return out


if __name__ == "__main__":
    P = load("burned")
    pickle.dump(P, open(ROOT / "o_v3" / "burned_cache.pkl", "wb"))
    import collections
    print(len(P), collections.Counter(p["cohort"] for p in P))
