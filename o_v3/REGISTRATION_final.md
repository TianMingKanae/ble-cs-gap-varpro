# O v3 fresh-procedure registration (H's BLE CS paper, 2026-11-21, written before any fresh procedure was loaded)

Request: H 11-21 chat ("絞盡腦汁想演算法，做到好為止").

## Data
- Fresh set = every procedure with index >= 15 in the 249 held-out files of notes/zenodo_partition_manifest.json.
  All earlier scripts loaded these files with max_procs=15 (grep of scripts/ and src/ on 11-21), so these
  procedures have never been scored. Counts: t1_indoor_los 6630 (74 geometries), t1_indoor_nlos 6800 (76),
  t1_outdoor_los 220 (22), t2_indoor_los 580 (1), t2_indoor_nlos 1685 (14).
- They are new measurement repetitions of the SAME geometries whose first 15 procedures were scored before
  (and used to choose v3's constants). This tests replication over new captures, not new geometries.
- data.zip sha256 1f3a34fbc0110a1906c7db5d63b20d9d152c90243b8a28a545664d2af48c8dba
- Guard: o_v3/data_guard.py refuses to load them unless O_V3_FINAL equals the sha256 of this file.

## Methods (frozen)
- v2.5 = src/estimator_v2_1.py defaults (the paper's DC-ANM-VarPro; sha 4facc78e993db0c6e1df1a2bdf87e99a489509a22ccf0f2f577d98fb4b6ada52), d_cal 0.881149670238504 m.
- v3 = o_v3/est_v3.py (sha dd8cb496d121b857a73427f48e87502b073963917eb20bfed385872a007dc247): v2.5 fit + leading-edge consistency gate, DELTA_LATE 0.5 m,
  ALPHA_LATE 0.25; fit code o_v3/fit_state.py (sha 675c4cd0e7b67058313e4b3ac6ab2b68858830811d819bcf0c953c6ee974f856). Constants chosen on burned data
  (dev files + held-out first 15); insensitive there over DELTA 0.5-1.5 m and ALPHA 0.2-0.25.
- Comparators unchanged: IFFT-LE, Sheikh-WPS, Wieme-FBSS (src/baselines.py sha d634e42fcfbc25eea171476c80232ea0525ba20a5755175f72356a388a28e25d).
- Shared code: src/estimator.py sha 9c60a1ab02e061dfe4902d1a1909aead05e6c3038f44e6de44714d9a4aac3016, src/model.py sha 8687f650c0e06c38c2e7fa52153633187bff98a38ee1243a9ca65c99f63dce56, o_v3/harness.py sha 7e4d68c940243c4c1fdb5b70cd4b19c9e3aa99e21cef1d92d0b9087839eaa4cc, o_v3/data_guard.py sha 0c4b217f6d05dfc2085bf7bf56c28438325c41eed3f2b6175b8c1f7f75f5a94b.
- Failure rule (paper): invalid or non-finite estimate -> absolute error = cohort d_max.
- Run: o_v3/final_run.py (sha 264908fda3c0ea1a893f21edd200b46e23846a7a419b999c9b1c46e3bc90b968); read-out: o_v3/verdict.py (sha 507ebb55478dfdffa0280114c3e862b86f7b3f31e9fe714d95b062aa7e539433),
  which prints exactly the tests below. It was dry-run on burned data (o_v3/dryrun_burned_verdict.txt) before
  registration. No test will be added after the fresh run is read.

## Tests (geometry-cluster bootstrap, B = 4000, seed 1015, two-sided percentile p, Holm within family)
- H1 primary (the paper's own confirmatory rule, on fresh data), t1_indoor_los:
  v2.5 - IFFT-LE and v2.5 - Sheikh-WPS, each dMAE and dP90 (difference of marginal 90th percentiles); 4 contrasts.
  Superiority over comparator j is SHOWN iff dMAE_j < 0 and dP90_j < 0, both with p_holm < 0.05; otherwise NOT SHOWN.
- H2 secondary (the new gate), t1_indoor_nlos: v3 - v2.5 and v3 - IFFT-LE, dMAE and dP90, 4 contrasts, same rule.
  On burned data this rule gave NOT SHOWN for both (P90 p_holm 0.077 / 0.60); expected to be hard to pass.
- H3 secondary, t1_indoor_los: v3 non-inferior to v2.5 iff the 95% CI upper bound of dMAE < +0.02 m and of dP90 < +0.05 m.
- If any H1/H2 p_holm falls in [0.02, 0.08], seed 2027 is also reported.
- Descriptive only: all five cohorts x five methods (MAE, MedAE, P90, >5 m rate, median signed error), v3 gate rate, runtime.

## Sanity checks
Cohort counts as above; all proc_idx >= 15; v3 rows with flag "pass" identical to v2.5; NaN counts per method.

## Limits stated in advance
Same geometries as the burned first 15; one dataset, one chipset, SISO; NLOS >5 m rates are driven by a few
geometries (burned: 4 of 76 geometries), so they are read at geometry level, not as procedure rates.
