# Table 5 Reliability Diagnosis

NOTE: the Before values in this document were proposals, not measurements.
They were measured on 2026-09-17; see results/tables/ for the computed values.

Scope: diagnosis only, per request. Nothing in the manuscript, README, or scripts was
changed. All numbers below are either read directly from committed CSVs / the manuscript
docx, or produced by *running existing, unmodified scripts* and diffing the output
against the committed CSV before restoring it (see §5).

---

## [1] Why `table5_repro.csv` (backup) and `table5_repro_canonical.csv` (canonical) differ

Read `scripts/35_table5_repro.py` in full (its own docstring already states most of this;
the following independently verifies it against the data).

- **`source=backup`** (default) reads a frozen file,
  `data/processed_crossveh_backup/{dataset}_{h}s.csv`. Verified column count = **24**
  feature columns (`ego_speed … risk_x_gapdiff`), containing **no** lateral/road-frame
  columns.
- **`source=canonical`** rebuilds samples on the fly from
  `data/interim/canonical/{ds}/*.parquet` + `data/interim/events_{ds}.csv`, then calls
  `build_features(..., use_road_frame=True)`. Verified feature count = **34** columns —
  the 24 backup columns plus 10 lateral/road-frame columns
  (`lat_offset, lat_offset_norm, lat_speed, lat_acc, lat_drift, lat_offset_slope5,
  lat_speed_std5, abs_lat_speed_max5, lat_speed_x_gapdiff, lat_drift_x_vosdiff`).
- **Event detection and negative sampling are identical in both paths.** Both call the
  same `build_samples()` + `balance()` from `src/features/kinematic.py` with the same
  `random_state=42` (this is literally the same code path `scripts/02_build_features.py`,
  the pre-refactor "conventional" builder, used to originally produce the backup CSVs).
  The *only* code difference between the two `35_table5_repro.py` branches is the
  `use_road_frame` flag passed to `build_features()` — i.e., whether the 10 lateral
  columns exist and get used by the model. There is no difference in which samples are
  drawn, only in which features describe them.
- One consequence worth flagging separately: `src/features/kinematic.py::build_features`
  now defaults to `use_road_frame=True`, but `scripts/02_build_features.py` (the script
  that originally produced the frozen 24-feature backup CSVs) calls
  `build_features(s, canon)` with no override. Run today, that call would silently
  produce 34-column output, not the 24-column backup files. In other words,
  `scripts/02_build_features.py` as currently checked in **can no longer regenerate its
  own historical output** — the backup CSVs are a frozen artifact of an earlier,
  unrecorded code state (git history for `kinematic.py` and `02_build_features.py` is a
  single squashed commit, so the earlier state itself isn't recoverable from this repo).

**Which one matches manuscript §3.2–§4.1's protocol?** Backup, with evidence:

Comparing the manuscript's Table 5 (4 rows) against both CSVs, per-cell (Purity /
AUC-all / AUC-mixed / Δ, computed as `auc_all − auc_mixed_only`). Note: the manuscript's
own Table 5 prints Δ as `auc_mixed − auc_all` (negative, e.g. "−0.140"); the magnitudes
below are directly comparable, only the sign convention differs per 근거.txt's requested
formula.

| Row | Manuscript R5 | backup CSV | \|manuscript−backup\| | canonical CSV | \|manuscript−canonical\| |
|---|---|---|---|---|---|
| highD purity | 0.96 | 0.9614 | 0.001 | 0.9607 | 0.001 |
| highD AUC-all | 0.884 | 0.8779 | 0.006 | 0.9293 | 0.045 |
| highD AUC-mixed | 0.744 | 0.7426 | 0.001 | 0.8500 | 0.106 |
| highD Δ | 0.140 | 0.1353 | 0.005 | 0.0793 | 0.061 |
| NGSIM purity | 0.79 | 0.788 | 0.002 | 0.7861 | 0.004 |
| NGSIM AUC-all | 0.791 | 0.7911 | 0.000 | 0.8590 | 0.068 |
| NGSIM AUC-mixed | 0.729 | 0.7213 | 0.008 | 0.7933 | 0.064 |
| NGSIM Δ | 0.062 | 0.0698 | 0.008 | 0.0657 | 0.004 |
| MiTra purity | 0.67 | 0.6727 | 0.003 | 0.6758 | 0.006 |
| MiTra AUC-all | 0.870 | 0.8647 | 0.005 | 0.9253 | 0.055 |
| MiTra AUC-mixed | 0.785 | 0.7740 | 0.011 | 0.8856 | 0.101 |
| MiTra Δ | 0.085 | 0.0907 | 0.006 | 0.0397 | 0.045 |
| exiD(7s) purity | 0.91 | 0.9073 | 0.003 | 0.9006 | 0.009 |
| exiD(7s) AUC-all | 0.981 | 0.9804 | 0.001 | 0.9945 | 0.014 |
| exiD(7s) AUC-mixed | 0.909 | 0.9120 | 0.003 | 0.9596 | 0.051 |
| exiD(7s) Δ | 0.072 | 0.0684 | 0.004 | 0.0349 | 0.037 |

Max abs diff to **backup ≈ 0.011**; max abs diff to **canonical ≈ 0.106**. Every one of
the 16 cells is closer to backup than to canonical, by roughly an order of magnitude on
the AUC/Δ columns.

Counting exact matches at the manuscript's own display precision (2–3 significant
figures): **Purity 4/4 match backup exactly; AUC-all 1/4; AUC-mixed 0/4; Δ 0/4 → 5/16
cells match, 11/16 differ.** This reproduces the 근거.txt background claim precisely
("16칸 중 11칸이 다르다… purity 열만 갱신되고 AUC·Δ 열은 옛 값이 남음"): the purity
column was evidently refreshed against a CSV at some point, but AUC/Δ were not
re-synced, and they carry small (~0.005–0.011) residual drift against the *backup* CSV
that is currently committed — small enough that this is very unlikely to be
canonical-vs-backup confusion, and more likely either (a) a slightly different XGBoost/
lib version at manuscript-writing time, or (b) the manuscript numbers predate the
currently-committed `table5_repro.csv` by one more regeneration than is visible in git
history.

**Verdict: backup에 가깝다 (closer to backup).** The manuscript's §3.2–§4.1 narrative
("lateral features absent from earlier iterations", §4.1) is also textually consistent
with backup being the older, feature-poor pipeline the paper is contrasting itself
against — not the protocol Table 5's numbers were drawn from. Canonical is not a
plausible source: its deviations from the manuscript (up to 0.106 AUC) are 10–20× larger
than backup's, and go in a consistent direction (canonical AUCs are all *higher* than
backup/manuscript, because the 10 extra lateral features are strong predictors — see
§4.1's own claim that lateral features move highD in-domain AUC by 0.08).

---

## [2] Decision table and Δ-range scenarios

Δ = `auc_all − auc_mixed_only`, computed directly from each CSV (not re-derived from
rounded manuscript numbers).

| Dataset | Item | Manuscript R5 | backup CSV | canonical CSV |
|---|---|---|---|---|
| highD (3s) | Purity | 0.96 | 0.9614 | 0.9607 |
| highD (3s) | AUC (all) | 0.884 | 0.8779 | 0.9293 |
| highD (3s) | AUC (mixed only) | 0.744 | 0.7426 | 0.8500 |
| highD (3s) | Δ | 0.140 | 0.1353 | 0.0793 |
| NGSIM (3s) | Purity | 0.79 | 0.7880 | 0.7861 |
| NGSIM (3s) | AUC (all) | 0.791 | 0.7911 | 0.8590 |
| NGSIM (3s) | AUC (mixed only) | 0.729 | 0.7213 | 0.7933 |
| NGSIM (3s) | Δ | 0.062 | 0.0698 | 0.0657 |
| MiTra (3s) | Purity | 0.67 | 0.6727 | 0.6758 |
| MiTra (3s) | AUC (all) | 0.870 | 0.8647 | 0.9253 |
| MiTra (3s) | AUC (mixed only) | 0.785 | 0.7740 | 0.8856 |
| MiTra (3s) | Δ | 0.085 | 0.0907 | 0.0397 |
| exiD (7s) | Purity | 0.91 | 0.9073 | 0.9006 |
| exiD (7s) | AUC (all) | 0.981 | 0.9804 | 0.9945 |
| exiD (7s) | AUC (mixed only) | 0.909 | 0.9120 | 0.9596 |
| exiD (7s) | Δ | 0.072 | 0.0684 | 0.0349 |

**Δ range by scenario:**

- **Scenario A (backup adopted): Δ min–max = 0.068 – 0.135** (NGSIM 0.0698 to highD
  0.1353). Consistent with the manuscript's stated "between 0.06 and 0.14" — both bounds
  land inside the claimed interval.
- **Scenario B (canonical adopted): Δ min–max = 0.035 – 0.079** (exiD 0.0349 to highD
  0.0793). **Inconsistent** with "between 0.06 and 0.14": 2 of 4 rows (MiTra 0.040, exiD
  0.035) fall below the claimed 0.06 floor, and the ceiling (0.079) never reaches the
  claimed 0.14.

This is further evidence for [1]: the "0.06–0.14" sentence, repeated 5–6 times through
the manuscript (lines 123, 169, 935–936, 944, 1540, 1641 in the extracted text), is only
defensible under Scenario A (backup). Adopting canonical would require rewriting that
range downward to roughly "0.03–0.08" in every one of those places.

---

## [3] Purity range claim ("about 0.65–0.96 across all seven datasets")

Full 7 × 3 = 21 combination purity table (both CSVs agree to ≤0.005 everywhere except
exiD@5s and exiD@7s, where the canonical rebuild uses a different, larger vehicle set):

| Dataset | 3s (backup / canonical) | 5s (backup / canonical) | 7s (backup / canonical) |
|---|---|---|---|
| highD | 0.9614 / 0.9607 | 0.9771 / 0.9771 | 0.9859 / 0.9853 |
| NGSIM | 0.7880 / 0.7861 | 0.7852 / 0.7805 | 0.8095 / 0.8082 |
| MiTra | 0.6727 / 0.6758 | 0.6920 / 0.6938 | 0.7122 / 0.7142 |
| ETRI  | 0.7869 / 0.7627 | 0.7927 / 0.7927 | 0.7679 / 0.7679 |
| EMT   | 0.7186 / 0.7124 | 0.7225 / 0.7195 | 0.7258 / 0.7288 |
| uniD  | 0.6484 / 0.6510 | 0.6940 / 0.6855 | 0.7265 / 0.7223 |
| exiD  | **0.1848 / 0.1843** | 0.7631 / 0.7200 | 0.9073 / 0.9006 |

**The claim as a blanket statement about "the conventional protocol" is false**: min
across all 21 cells is exiD@3s ≈ **0.18** (both CSVs agree tightly, so this is not a
canonical/backup artifact), far below the claimed floor of 0.65. The claimed ceiling of
0.96 is also exceeded by highD@5s (≈0.977) and highD@7s (≈0.985–0.986) in both CSVs.

**Condition under which "0.65–0.96" does hold**: only when each dataset is evaluated at
the single horizon Table 5 actually reports for it — i.e., 3s for the other six datasets
and **7s specifically for exiD** (which is exactly the row selection Table 5 already
uses, and the manuscript's own line "On exiD, purity reaches 0.91 at the 7 s horizon"
hints at this without stating it as a condition). Under that one-horizon-per-dataset
selection: highD 0.96, NGSIM 0.79, MiTra 0.67, ETRI 0.76–0.79, EMT 0.71–0.72, uniD
0.65 (backup: 0.6484, marginally under 0.65), exiD(7s) 0.90–0.91 — a set that matches
"about 0.65–0.96" to within rounding, with highD and uniD sitting right at the two
boundaries.

At 3s — the horizon actually used for 6 of Table 5's rows, and the horizon the sentence's
context ("purity of about 0.65–0.96 across all seven datasets", §3.3, no horizon
qualifier) would lead a reader to assume applies uniformly — exiD purity is 0.18, a
dramatic outlier that breaks the claim.

**Proposed corrected sentence** (not applied to the manuscript):

> "Group-label purity under the conventional protocol is about 0.65–0.96 for six of the
> seven datasets at the 3 s horizon; exiD is a outlier at 3 s (purity ≈ 0.18, because
> entry/exit-ramp vehicles create many single-label groups only at longer horizons) and
> only falls in this range at the 7 s horizon (purity ≈ 0.90–0.91), which is the horizon
> Table 5 reports for it."

---

## [4] Reproducibility of the four "before" values in §3.2/§4.1

All four "after" values are independently confirmed already present in committed CSVs:
`diag_purity_all.csv` row `exiD,7,...,auc_full=0.9742` ≈ manuscript's 0.974; the same
file's methodology is what produces road-frame, sign-corrected numbers throughout. None
of the four "before" values appear in any committed CSV, and the precomputed adapter
ablation stack (`data/processed/{ds}_V1/_V2/_V3_{h}s.csv`, used for
`results/tables/adapter_ablation.csv`) does **not** isolate them either — `V2` combines
`--no_roadframe` **and** `--no_latsign` together, and `V1` adds `--no_neighbors` on top,
so no existing file isolates "road-frame off, sign correction on" or "road-frame on,
sign correction off" the way the manuscript's two claims require.

| # | Claim | Before | After (confirmed in CSV) | Reproducible? |
|---|---|---|---|---|
| 1 | exiD lat_speed solo, global vs road frame | 0.928 | 0.774 | **Possible** |
| 2 | exiD in-domain, global vs road frame | 0.996 | 0.974 (`diag_purity_all.csv`) | **Possible** |
| 3 | ETRI sign correction | 0.41 | 0.58 (≈ `purity_auc.csv` clean_auc=0.5734) | **Possible** |
| 4 | highD lateral features | 0.82 | 0.90 (≈ `purity_auc.csv` clean_auc=0.9001) | **Possible** |

All four are possible because `scripts/02_build_features_v2.py` already exposes the
exact ablation flags needed, individually (not just as the bundled V1/V2/V3 stack), and
no code path has been removed:

- **#1 and #2 (exiD, frame only):** regenerate with road-frame off but sign correction
  left on:
  `python scripts/02_build_features_v2.py --dataset exiD --no_roadframe --suffix _globalframe`
  → produces `data/processed/exiD_globalframe_{3,5,7}s.csv`. Then evaluate with the same
  method `diag_shortcut.py`/`diag_purity_all.py` already use
  (`in_domain_auc(df, cols, seed)` on `(recording,vehicle)`-group 70/30 splits, seeds
  42/0/1): point `load()` at `exiD_globalframe_7s.csv` instead of `exiD_gt_7s.csv`, run
  once with `cols=["lat_speed"]` (should recover ≈0.928) and once with the full feature
  list (should recover ≈0.996). Only a one-line path change to an existing evaluation
  routine is needed; no new logic.
- **#3 (ETRI, sign only):** regenerate with sign correction off but road-frame on:
  `python scripts/02_build_features_v2.py --dataset ETRI --no_latsign --suffix _nosign`
  → `data/processed/ETRI_nosign_{3,5,7}s.csv`. Evaluate the full feature list at 3s
  (the horizon whose "after" value, 0.573–0.58, is already confirmed) the same way as
  above, pointed at `ETRI_nosign_3s.csv`.
- **#4 (highD, lateral features on/off):** no regeneration needed at all — the current
  `highD_gt_3s.csv` already carries both the 24 base columns and the 10 lateral columns.
  "Before" = re-run the existing in-domain evaluation restricted to
  `[c for c in cols if c not in LAT_COLS_plus_interactions]`; "after" is already
  `purity_auc.csv`'s `clean_auc` for highD (0.9001, matching the manuscript's "0.90").
  This is the cheapest of the four — a feature-column filter only, no rebuild.

**Estimated run time** (not executed, per instructions):
- #4 highD: seconds — reuses the already-materialized `highD_gt_3s.csv` (20,514 rows),
  just an XGBoost fit twice (with/without 10 columns).
- #3 ETRI: well under a minute — ETRI's canonical data is 5 parquet files / 3.6 MB total
  (`ETRI_gt_3s.csv` is 262 rows); `02_build_features_v2.py` rebuilds all 3 horizons per
  invocation (no per-horizon flag exists), but ETRI is the smallest of the seven
  datasets.
- #1/#2 exiD: the dominant cost. exiD's canonical data is 186 parquet files / 987 MB, the
  largest of the seven datasets. `scripts/32_purity_auc.py`, which builds
  cross-vehicle samples + features + fits XGBoost for all 7 datasets (leaky and clean
  passes) end-to-end, took **8m15s wall-clock** in this environment (see §5); exiD alone
  is a large fraction of that. A single-dataset, 3-horizon `02_build_features_v2.py`
  run for exiD is estimated at roughly **2–6 minutes**, plus a few seconds for each
  evaluation pass.

---

## [5] Execution environment

**`verify_report.md` does not exist anywhere on this server or in git history** (checked
`git log --all -- '*verify*'` — no hits; `find /` for the filename — no hits). It is
listed in `.gitignore` (`verify_*.md`), so if it once existed it was never committed and
is not currently present in this checkout. This diagnosis proceeds without it, using an
independent survey of Python environments on the machine instead.

Found five relevant environments with `catboost`+`shap`+`torch`+`xgboost` all installed:
`base` (conda), `gavogo_env`, `hackathon_env` (partial — no catboost/shap), `lambda`,
and one other with only `torch`. Checked each against every pin in `requirements.txt`:

| package | required | `lambda` | `gavogo_env` |
|---|---|---|---|
| numpy | 1.26.4 | 1.26.4 ✓ | 2.4.6 ✗ |
| pandas | 2.2.2 | 2.2.2 ✓ | 3.0.3 ✗ |
| pyarrow | 24.0.0 | 24.0.0 ✓ | 24.0.0 ✓ |
| scikit-learn | 1.4.1.post1 | 1.4.1.post1 ✓ | 1.9.0 ✗ |
| scipy | 1.13.1 | 1.13.1 ✓ | 1.18.0 ✗ |
| matplotlib | 3.8.4 | 3.8.4 ✓ | 3.11.0 ✗ |
| tqdm | 4.68.4 | 4.68.4 ✓ | 4.68.3 ✗ |
| xgboost | 3.3.0 | 3.3.0 ✓ | 3.3.0 ✓ |
| lightgbm | 4.6.0 | 4.6.0 ✓ | 4.6.0 ✓ |
| catboost | 1.2.10 | 1.2.10 ✓ | 1.2.10 ✓ |
| shap | 0.46.0 | 0.46.0 ✓ | 0.52.0 ✗ |
| torch | 2.7.0 | 2.7.0 ✓ | 2.11.0+cu128 ✗ |

`/home/futurenetworklab/lambda` matches **all 12** `requirements.txt` pins exactly.
`gavogo_env` (despite the project-suggestive name) matches only pyarrow/xgboost/
lightgbm/catboost and is otherwise a materially newer stack (numpy 2.x, pandas 3.x,
sklearn 1.9, torch 2.11). `lambda` is the only environment on this machine that is
consistent with `requirements.txt` as committed.

**Cheapest verification, run**: `results/tables/purity_auc.csv` was backed up, then
regenerated with `/home/futurenetworklab/lambda/bin/python3 scripts/32_purity_auc.py`
(unmodified script), then compared, then the committed file was restored via `git
checkout` so the working tree is unchanged. Result:

| dataset | conv_purity Δ | leaky_auc Δ | clean_auc Δ | auc_inflation Δ |
|---|---|---|---|---|
| highD | +0.0013 | −0.0006 | +0.0003 | −0.0008 |
| NGSIM | +0.0001 | −0.0022 | −0.0054 | +0.0033 |
| MiTra | −0.0026 | +0.0020 | +0.0003 | +0.0017 |
| ETRI | **+0.0242** | −0.0013 | 0.0000 | −0.0012 |
| EMT | +0.0062 | −0.0044 | −0.0024 | −0.0022 |
| uniD | −0.0026 | +0.0102 | +0.0019 | +0.0083 |
| exiD | +0.0001 | −0.0018 | +0.0001 | −0.0018 |

**Not an exact match** (max abs diff 0.024, on ETRI purity; max AUC diff 0.010, on
uniD).

To rule out run-to-run nondeterminism (XGBoost's `n_jobs=-1` histogram method is not
generally bit-reproducible across thread scheduling) as the explanation, the script was
run a **second, independent time** in the same `lambda` environment and diffed against
the first run. Result: **run 2 reproduces run 1 to within floating-point noise (≤1e-16,
i.e. exactly)** on every cell, including ETRI. So the `lambda` environment is internally
fully deterministic for this script — the ~0.005–0.024 gap against the *committed*
`purity_auc.csv` is not run-to-run noise. It is a real, reproducible discrepancy between
"the code run in `lambda` today" and "whatever produced the committed file," most
plausibly explained by one of: (a) a different machine/BLAS backend than `lambda`
(identical package *versions* do not guarantee identical floating-point summation order
across different CPUs/BLAS builds, which would explain the AUC-column drift), or (b) the
underlying canonical parquet/events input for at least ETRI having changed on disk since
`purity_auc.csv` was committed — this second possibility is the more likely explanation
for the ETRI purity gap specifically (0.0242), since purity is a deterministic groupby
over `build_samples()`/`balance()` with fixed `random_state=42` and has no dependency on
model fitting or thread scheduling at all; a change in model-independent input data is
the only way that number moves. This was not run down further within the "cheapest
verification" scope requested.

**Conclusion for the match/mismatch branching in the request: mismatch.** Per
instructions ("불일치하면 그 사실과 차이값을 그대로 보고해라"), the diff table above is
reported as-is rather than treated as confirming `lambda` as the paper environment. What
can be said positively: of every environment found on this machine, `lambda` is the only
one whose *pinned package versions* match `requirements.txt` exactly, and it is
internally deterministic — so it remains the best available candidate — but it does not
bit-reproduce the committed `purity_auc.csv`, and the repo's git history does not
contain enough information (single squashed initial commit) to confirm or rule out a
different machine, BLAS build, or slightly different input snapshot as the true source
of the committed numbers.

**Per instructions, `requirements.txt` was not modified.** If the user wants a `pip
freeze` captured from `lambda` for the manuscript's `[AUTHOR INPUT REQUIRED]` placeholder
(line 882–887 of the docx), that is a one-command follow-up
(`/home/futurenetworklab/lambda/bin/pip freeze`), not done here since it wasn't asked
for as part of this diagnosis.
