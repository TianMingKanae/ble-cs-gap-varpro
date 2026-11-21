"""Prints exactly the registered lines from o_v3/final_raw.pkl (written before the run was read)."""
import pickle, numpy as np, collections
R = pickle.load(open("o_v3/final_raw.pkl", "rb"))
meta = R["meta"]
co = np.array([m["cohort"] for m in meta]); geo = np.array([m["geometry_id"] for m in meta])
dt = np.array([m["d_true_m"] for m in meta]); dmax = np.array([m["d_max_m"] for m in meta])
D = {k: np.array([r[0] for r in v]) for k, v in R["res"].items()}
AE = {k: np.where(np.isfinite(v), np.abs(v - dt), dmax) for k, v in D.items()}
SE = {k: np.where(np.isfinite(v), v - dt, np.nan) for k, v in D.items()}


def boot(c, a, b, B=4000, seed=1015):
    m = co == c; g = geo[m]; x = AE[a][m]; y = AE[b][m]
    G = sorted(set(g)); idx = [np.flatnonzero(g == k) for k in G]
    rng = np.random.default_rng(seed)
    pt = (x.mean() - y.mean(), np.quantile(x, .9) - np.quantile(y, .9))
    bs = np.empty((B, 2))
    for i in range(B):
        s = np.concatenate([idx[j] for j in rng.integers(0, len(G), len(G))])
        bs[i] = (x[s].mean() - y[s].mean(), np.quantile(x[s], .9) - np.quantile(y[s], .9))
    p = [(1 + 2 * min((bs[:, j] <= 0).sum(), (bs[:, j] >= 0).sum())) / (B + 1) for j in range(2)]
    ci = [np.quantile(bs[:, j], [.025, .975]) for j in range(2)]
    return pt, p, ci


def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def family(name, c, pairs, seed=1015):
    rows = []
    for a, b in pairs:
        pt, p, ci = boot(c, a, b, seed=seed)
        rows += [(f"{a} - {b} dMAE", pt[0], p[0], ci[0]), (f"{a} - {b} dP90", pt[1], p[1], ci[1])]
    adj = holm(np.array([r[2] for r in rows]))
    print(f"== {name} [{c}, n={int((co == c).sum())}, geometries={len(set(geo[co == c]))}, B=4000, seed={seed}]")
    for r, q in zip(rows, adj):
        print(f"   {r[0]:28s} {r[1]:+.4f} m  95% CI [{r[3][0]:+.4f}, {r[3][1]:+.4f}]  p={r[2]:.4f}  p_holm={q:.4f}")
    return {r[0]: (r[1], q, r[3]) for r, q in zip(rows, adj)}


print("SANITY counts", dict(collections.Counter(co)))
flags = collections.Counter(r[2].get("flag") for r in R["res"]["v3"])
pas = np.array([r[2].get("flag") == "pass" for r in R["res"]["v3"]])
print("SANITY v3 flags", dict(flags), "| pass rows identical to v2.5:", bool(np.allclose(D["v3"][pas], D["v2.5"][pas], equal_nan=True)))
print("SANITY invalid/NaN per method", {k: int((~np.isfinite(v)).sum()) for k, v in D.items()})

H1 = family("H1 primary replication (paper rule)", "t1_indoor_los", [("v2.5", "IFFT-LE"), ("v2.5", "Sheikh-WPS")])
for comp in ("IFFT-LE", "Sheikh-WPS"):
    a, b = H1[f"v2.5 - {comp} dMAE"], H1[f"v2.5 - {comp} dP90"]
    ok = a[0] < 0 and b[0] < 0 and a[1] < 0.05 and b[1] < 0.05
    print(f"   VERDICT H1 v2.5 vs {comp}: {'SUPERIORITY SHOWN' if ok else 'NOT SHOWN'} on fresh Tier-1 indoor LOS")
H2 = family("H2 secondary gate (NLOS)", "t1_indoor_nlos", [("v3", "v2.5"), ("v3", "IFFT-LE")])
for comp in ("v2.5", "IFFT-LE"):
    a, b = H2[f"v3 - {comp} dMAE"], H2[f"v3 - {comp} dP90"]
    ok = a[0] < 0 and b[0] < 0 and a[1] < 0.05 and b[1] < 0.05
    print(f"   VERDICT H2 v3 vs {comp}: {'SUPERIORITY SHOWN' if ok else 'NOT SHOWN'} on fresh Tier-1 indoor NLOS")
pt, p, ci = boot("t1_indoor_los", "v3", "v2.5")
ok = ci[0][1] < 0.02 and ci[1][1] < 0.05
print(f"== H3 non-inferiority v3 vs v2.5 on t1_indoor_los: dMAE {pt[0]:+.4f} CI [{ci[0][0]:+.4f}, {ci[0][1]:+.4f}] (margin +0.02); "
      f"dP90 {pt[1]:+.4f} CI [{ci[1][0]:+.4f}, {ci[1][1]:+.4f}] (margin +0.05) -> {'NON-INFERIOR' if ok else 'NOT SHOWN'}")
near = [k for F in (H1, H2) for k, v in F.items() if 0.02 <= v[1] <= 0.08]
if near:
    print("ROBUSTNESS second seed 2027 (some p_holm in [0.02, 0.08]):", near)
    family("H1 seed2", "t1_indoor_los", [("v2.5", "IFFT-LE"), ("v2.5", "Sheikh-WPS")], seed=2027)
    family("H2 seed2", "t1_indoor_nlos", [("v3", "v2.5"), ("v3", "IFFT-LE")], seed=2027)
print("== DESCRIPTIVE (not tested): MAE / MedAE / P90 / >5m % / median signed (m); v3 gate rate")
for c in ['t1_indoor_los', 't1_outdoor_los', 't2_indoor_los', 't1_indoor_nlos', 't2_indoor_nlos']:
    m = co == c
    gate = np.mean([R["res"]["v3"][i][2].get("flag") == "late_gate" for i in np.flatnonzero(m)])
    print(f"  {c} (n={m.sum()}, G={len(set(geo[m]))}, v3 gate {gate*100:.1f}%)")
    for k in D:
        a = AE[k][m]
        print(f"     {k:11s} {a.mean():.3f} / {np.median(a):.3f} / {np.quantile(a, .9):.3f} / {(a > 5).mean()*100:4.1f}% / {np.nanmedian(SE[k][m]):+.3f}")
rt = {k: np.median([r[1] for r in v]) for k, v in R["res"].items()}
print("  median runtime ms (14-process pool, descriptive):", {k: round(float(v), 1) for k, v in rt.items()})
