# Gap-Constrained Two-Way Variable Projection (`DC-ANM-VarPro` and `DC-ANM-VarPro-G`)

Reference Python implementation and registered evaluation suite for **"Gap-Constrained Two-Way Variable Projection for Bluetooth Channel Sounding Ranging under Wi-Fi Erasures and Multipath"** (Yu-Han Chen, National Yang Ming Chiao Tung University).

> **License:** MIT (see `LICENSE`). No raw dataset files are bundled in this release; download them from Zenodo as described below.

## 1. Repository Layout

- `src/` — Core physical model, Slepian--Bangs surrogate CRB, baselines, and gap-constrained Variable Projection estimators:
  - `model.py`: Core v6.0/v6.2 72-tone CS channel model, timing schedules, Mode-2/3 product noise, and `ZenodoLoader`.
  - `crb.py`: Slepian--Bangs Fisher Information Matrix (FIM) and local $\mathrm{RCRB}_d$.
  - `baselines.py`: `IFFT-Max`, `IFFT-LE`, `Sheikh-WPS`, `h^2-ANM`, `Wieme-FBSS`, and `Kim-K15`.
  - `estimator.py`, `estimator_v2.py`, `estimator_v2_1.py`: Stage-1 lag-coherence drift compensation, Stage-2 gap-triggered MMV-ANM ADMM seeding, and Stage-3 radial-simplex gap-constrained Variable Projection (`DC-ANM-VarPro`).
  - `confirmatory_executor.py`, `confirmatory_stats.py`: Held-out procedure execution and cluster-bootstrap statistical helpers.
- `o_v3/` — Frozen gated variant (`DC-ANM-VarPro-G`) and registered 15,915-repetition evaluation reproducing **Table II** and **H1/H2/H3**:
  - `fit_state.py`: Byte-identical `v2.1` fitting pipeline returning fitted atoms (`taus`, `betas`, `t2_min`).
  - `est_v3.py`: Zero-refit leading-edge consistency gate (`DC-ANM-VarPro-G`, $\Delta_{\mathrm{late}}=0.5$~m, $\alpha_{\mathrm{late}}=0.25$).
  - `data_guard.py`, `data_guard_1121b.py`: Partition-locked data loader enforcing `O_V3_FINAL` SHA-256 verification before accessing `proc_idx >= 15` (`data_guard_1121b.py` adds strict selector validation).
  - `harness.py`: Multiprocessing evaluation harness (`14` workers by default, $\hat{d}_{\mathrm{cal}} = 0.881149670238504$~m).
  - `final_run.py`: Registered runner evaluating all five methods across the 15,915 additional same-geometry repetitions (`o_v3/final_raw.pkl`).
  - `verdict.py`: Prespecified read-out script printing H1/H2/H3 bootstrap Holm tests, seed-2027 checks, Table II descriptive metrics, and runtimes.
  - `REGISTRATION_final.md`: Frozen pre-registration specification (`sha256: 57bd906f91eed78ec7bb8646735df33276cd5ac9b4ce778d1cff4efa83764856`).
- `notes/zenodo_partition_manifest.json` — Auditable 279-file partition manifest (`10.5281/zenodo.17347695`).
- `requirements.txt` — Minimal runtime dependencies (`numpy>=2.0`, `scipy>=1.14`; tested on Python `3.13.14`, NumPy `2.5.1`, SciPy `1.18.1`).

## 2. Dataset Download Instructions (Zenodo DOI: `10.5281/zenodo.17347695`)

The Nordic nRF54L15-DK commercial Bluetooth Channel Sounding I/Q archive is **not bundled** in this repository. Download the archive from Zenodo:

- **DOI:** `10.5281/zenodo.17347695` (<https://doi.org/10.5281/zenodo.17347695>)

Place the downloaded dataset files under `data/zenodo_17347695/` relative to the repository root:

```text
data/
  zenodo_17347695/
    data.zip              # expected sha256: 1f3a34fbc0110a1906c7db5d63b20d9d152c90243b8a28a545664d2af48c8dba
    mapping.csv
    outdoor_mapping.csv
```

Verify the SHA-256 digest of `data.zip` matches `1f3a34fbc0110a1906c7db5d63b20d9d152c90243b8a28a545664d2af48c8dba` before running evaluation scripts. Note that `outdoor/2_meter.csv` (25 procedures) is excluded as an audited distance-metadata erratum in `notes/zenodo_partition_manifest.json`.

## 3. Reproducing Table II and the Registered Run (H1 / H2 / H3)

Run all commands from the repository root so `src` and `o_v3` resolve cleanly.

### Install dependencies

```bash
pip install -r requirements.txt
```

### Execute the registered repetition run and print Table II / verdicts

`o_v3/data_guard.py` requires the environment variable `O_V3_FINAL` to equal the exact SHA-256 digest of `o_v3/REGISTRATION_final.md` (`57bd906f91eed78ec7bb8646735df33276cd5ac9b4ce778d1cff4efa83764856`).

**Linux / macOS (bash):**

```bash
O_V3_FINAL=57bd906f91eed78ec7bb8646735df33276cd5ac9b4ce778d1cff4efa83764856 python -m o_v3.final_run
python o_v3/verdict.py
```

**Windows (PowerShell):**

```powershell
$env:O_V3_FINAL = "57bd906f91eed78ec7bb8646735df33276cd5ac9b4ce778d1cff4efa83764856"
python -m o_v3.final_run
python o_v3/verdict.py
```

### Expected outputs

- `python -m o_v3.final_run` loads the 15,915 repetition procedures (`proc_idx >= 15` across `t1_indoor_los=6630`, `t1_indoor_nlos=6800`, `t1_outdoor_los=220`, `t2_indoor_los=580`, `t2_indoor_nlos=1685`), runs all five methods (`v2.5`, `v3`, `IFFT-LE`, `Sheikh-WPS`, `Wieme-FBSS`) across a 14-process worker pool by default, and writes `o_v3/final_raw.pkl`.
- `python o_v3/verdict.py` reads `o_v3/final_raw.pkl` and prints sanity checks, **H1** (Tier-1 Indoor LOS replication), **H2** (Tier-1 Indoor NLOS gating), **H3** (Tier-1 Indoor LOS non-inferiority), secondary seed-2027 robustness checks, **Table II** descriptive metrics (`MAE / MedAE / P90 / >5m % / median signed error`), gate trigger rates, and 14-process median runtimes.

## 4. Provenance

Both `notes/zenodo_partition_manifest.json` (`sha256: 1ef0de5a2186eb4a1843ab4532184058c4bca7afb6ad4021519304c108f3bd2a`) and `o_v3/REGISTRATION_final.md` (`sha256: 57bd906f91eed78ec7bb8646735df33276cd5ac9b4ce778d1cff4efa83764856`) are kept byte-identical to their frozen versions because their SHA-256 hashes are pinned (and `O_V3_FINAL` verifies the exact hash of `o_v3/REGISTRATION_final.md` at runtime). Historical phrases in those two files—such as `"zero runs executed on any of the 249 held-out files"`, `"fresh"`, and `"never loaded / never scored"`—are retained solely as historical project records, not current exposure guarantees.

The authoritative reading is that of the manuscript: the 15,915 procedures evaluated by `o_v3/final_run.py` (`proc_idx >= 15`) are additional measurement repetitions collected at the same previously evaluated geometries; earlier retained score files and reviewed scoring paths cover procedure indices `0–14` only; an earlier schema inspection parsed all procedures; historical non-scoring outside retained records and registration timing are not independently certified; and the `DC-ANM-VarPro-G` gate constants (`delta_late = 0.5 m`, `alpha_late = 0.25`) were selected on previously evaluated data (development files plus held-out indices `0–14`).
