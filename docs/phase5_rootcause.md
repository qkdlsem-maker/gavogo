# PHASE 5 — root-cause identification for regen vs. committed divergence

Per r6.txt: read-only, no code/data edits, no commits, no .docx opened, no conclusions
about which version is "correct" — mechanical differences only. Every claim below cites
an exact file path or exact command output.

---

## 1. 3-second-horizon feature files under `data/processed/`: row counts, columns, mtimes; search for old/backup copies

Full table (row count = data rows excluding header, column count includes `label`/meta):

| Dataset | plain `_3s` | `_gt_3s` | `_gtdi_3s` | `_V1_3s` | `_V2_3s` | `_V3_3s` |
|---|---|---|---|---|---|---|
| highD | 21,070 rows / 38 cols / **2026-09-15 13:07** | 21,070 / 52 / 13:20 | 21,070 / 54 / 13:30 | 20,514 / 38 / **2026-07-14 10:59** | 20,514 / 38 / 07-14 11:00 | 20,514 / 38 / 07-14 11:02 |
| NGSIM | 4,270 / 38 / 13:08 | 4,270 / 52 / 13:21 | 4,270 / 54 / 13:31 | 4,268 / 38 / 07-14 11:04 | 4,268 / 38 / 07-14 11:04 | 4,268 / 38 / 07-14 11:05 |
| MiTra | 27,500 / 38 / 13:10 | 27,500 / 52 / 13:21 | 27,500 / 54 / 13:31 | 27,465 / 38 / 07-14 11:06 | 27,465 / 38 / 07-14 11:11 | 27,465 / 38 / 07-14 11:16 |
| ETRI | 322 / 38 / 13:13 | 322 / 52 / 13:22 | 322 / 54 / 13:32 | 261 / 38 / 07-14 11:19 | 261 / 38 / 07-14 11:19 | 261 / 38 / 07-14 11:19 |
| EMT | 2,904 / 38 / 13:13 | 2,904 / 52 / 13:22 | 2,904 / 54 / 13:32 | 1,444 / 38 / 07-14 11:19 | 1,444 / 38 / 07-14 11:19 | 1,444 / 38 / 07-14 11:19 |
| uniD | 12,970 / 38 / 13:13 | 12,970 / 52 / 13:23 | 12,970 / 54 / 13:32 | 11,099 / 38 / 07-14 11:20 | 11,099 / 38 / 07-14 11:20 | 11,099 / 38 / 07-14 11:21 |
| exiD | 355,802 / 38 / 13:16 | 355,802 / 52 / 13:27 | 355,802 / 54 / 13:35 | 82,924 / 38 / 07-14 11:22 | 82,924 / 38 / 07-14 11:24 | 82,924 / 38 / 07-14 11:26 |

**Key fact: `_V1`/`_V2`/`_V3` files are all dated 2026-07-14 — untouched by the
2026-09-15 12:49–17:34 PHASE-1 regeneration run** (confirmed: their mtimes fall
outside the `regen_timing.log` PHASE1 START/DONE window entirely). The plain,
`_gt`, and `_gtdi` files are all dated 2026-09-15, inside that window, and their row
counts are **larger** than the corresponding `_V1`/`_V2`/`_V3` counts for every
dataset except highD/NGSIM/MiTra (near-identical, +0.3–2.7%) — exiD is the extreme
case (355,802 vs 82,924, **4.29×**), EMT is 2.01×, uniD is 1.17×, ETRI is 1.23×.
Within a given vintage (either all `-V1/V2/V3` or all plain/`_gt`/`_gtdi`), row
counts for a given dataset are identical across variants — only column count differs
(38 → 52 → 54), confirming these later stages add columns without re-sampling rows
(see §5 for exact mechanism).

**Old/backup file search**, whole accessible filesystem:
- `data/processed_crossveh_backup/` exists in this repo (the "backup"/24-feature
  pipeline already documented in `table5_diagnosis.md` [1] — a separate, previously
  known frozen directory, unrelated to the `_V1/V2/V3` files above).
- No `.bak` files, no other backup directory, and **no old snapshot of
  `data/processed/*_3s.csv` (plain/`_gt`/`_gtdi`) predating 2026-09-15** was found
  anywhere under `/home/futurenetworklab` (searched for `*backup*`, `*.bak`,
  `*_old*`, `*_orig*`; also searched for any `events_*.csv` copy outside this repo's
  `data/interim/` — none found).
- `regen_timing.log` records `2026-09-15T17:33:56+09:00 results/ restored from
  results_committed_backup` — a directory used to redirect PHASE-1 output into
  `results_regen/` without touching `results/`. It no longer exists on disk (already
  cleaned up after the restore); **확인 불가** whether it held anything beyond a copy
  of `results/tables|models|figures` (it would not have held `data/processed/`
  snapshots, since PHASE 1's redirect trick — per `regen_timing.log` — only
  concerned the `results/` output tree, not `data/`).
- A structurally similar but code-only copy of this project exists at
  `/home/futurenetworklab/gavogo` (separate git history, 4 commits, most recent
  "Add leak-free sampling, purity diagnostic, and reproduction scripts"). It
  contains `src/features/kinematic_old.py`, an older version of `kinematic.py`
  predating the road-frame (`v4`) lateral-feature rewrite, using per-lane-centroid
  geometry instead of the current `add_road_frame`/`add_global_frame` design. Its
  `build_samples()`/`balance()` functions were diffed directly against this repo's
  current `kinematic.py` and found **byte-identical** (used as the reference copy
  cited in §4/§5 below) — only the feature-building code differs between the two,
  not the sampler. This is source-code history, not a feature-file backup, and it is
  **not confirmed** to be temporally related to whichever run produced this repo's
  committed `results/tables/*.csv` — flagging its existence, not asserting relevance.

## 2. Why `results_regen/tables/dataset_stats.csv` is byte-identical to `results/tables/dataset_stats.csv`

`scripts/29_dataset_stats.py` (lines 17–20) reads **only**
`config.INTERIM_DIR / "canonical" / {ds} / *.parquet`, columns
`["recording_id", "vehicle_id", "frame"]`, and aggregates `nunique()`/`len()`/
frame-range-derived seconds per dataset — no processed/sample files, no model
fitting, nothing hardcoded or cached.

`diff results/tables/dataset_stats.csv results_regen/tables/dataset_stats.csv` →
**empty diff, byte-identical, confirmed directly.**

Mechanism: `data/interim/canonical/*/*.parquet` is itself rebuilt by
`scripts/01_build_events.py` (`adapter.get_canonical()`, line 42/59) as the very
first PHASE-1 step. Its file mtimes (checked directly, e.g.
`data/interim/canonical/highD/01.parquet` → `2026-09-15T12:49:10.82`) fall at the
exact same epoch second as `regen_timing.log`'s `PHASE1 START` marker
(`1789444150`), confirming the canonical parquet layer **was** freshly rebuilt in
PHASE 1, not reused from an old snapshot. Since the source (`data/raw/`, which
PHASE 1 never touches) is static, and `01_build_events.py` is unmodified code (see
§4), rebuilding canonical parquet from unchanged raw input with unchanged code
deterministically reproduces the same per-recording trajectory/frame counts every
time — which is exactly what `29_dataset_stats.py` measures. That is the byte-level
explanation for the identical output.

**On the ~3s runtime**: consistent with what the script does. It reads only 3 of
~15+ available columns per parquet file (columnar format, cheap), across 373 total
`.parquet` files (see `results_regen/tables/data_manifest.csv`-style counts: highD
60, NGSIM 3, MiTra 6, ETRI 5, EMT 20, uniD 13, exiD 186), performs only
`nunique()`/`len()`/min-max on already-loaded in-memory arrays — no feature
engineering, no XGBoost fit, no disk writes beyond one small final CSV. Nothing
about this workload should take longer than a few seconds on this machine (32
logical CPUs, GPUs not used by this script at all). **No anomaly found.**

## 3. Tracing `23_ood_ci.py`'s pooled n = 371,998

`scripts/23_ood_ci.py::load()` (line 24–27): `pd.read_csv(config.PROCESSED_DIR /
f"{ds}_gt_{H}s.csv")` with `H = 3` (line 20), called once per `TARGETS = [
config.HOLDOUT_DATASET] + config.OOD_DATASETS` = `["ETRI", "EMT", "uniD", "exiD"]`
(line 21, `config.py` lines 32–34). Pooled n = `len(y_all)` = sum of rows across
those 4 files (line 69, `np.concatenate(ys_all)`).

Exact row counts (from §1's table, `_gt_3s` column):

```
ETRI_gt_3s.csv  =    322
EMT_gt_3s.csv   =  2,904
uniD_gt_3s.csv  = 12,970
exiD_gt_3s.csv  = 355,802
                  -------
sum             = 371,998   ← EXACT MATCH to results_regen/tables/ood_ci.csv POOLED n
```

**For the sum to equal 95,728** (the committed/manuscript value), the same four
files would need to have row counts summing to 95,728. The `_V1_3s.csv` row counts
for exactly these four datasets (§1 table) sum to:

```
ETRI_V1_3s.csv  =    261
EMT_V1_3s.csv   =  1,444
uniD_V1_3s.csv  = 11,099
exiD_V1_3s.csv  = 82,924
                  -------
sum             = 95,728   ← EXACT MATCH to the committed/manuscript pooled n
```

This is an exact arithmetic match, not an approximation. It shows the committed
`ood_ci.csv` (n=95,728) is numerically consistent with having been built from
`_gt_3s.csv`-equivalent files whose row counts matched today's frozen `_V1_3s.csv`
vintage (2026-07-14), while the regenerated `ood_ci.csv` (n=371,998) was built from
today's (2026-09-15) `_gt_3s.csv` files, which have more rows. **This does not by
itself prove `_gt_3s.csv` had 95,728-matching row counts historically — only that
the arithmetic is exactly consistent with that being true.** Whether the actual
historical `_gt_3s.csv` files (not just the `_V1_3s.csv` proxy) ever had these exact
row counts cannot be directly confirmed — no snapshot of the old `_gt_3s.csv` files
exists (§1) — **확인 불가** for direct proof, though the arithmetic coincidence
across all 4 datasets simultaneously is a strong (not proven) mechanical signal.

## 4. `02_build_features.py` vs `28_horizon1.py` sample-construction, side by side

These two scripts use **two entirely different sampler functions**, not the same
function with different parameters:

| | `02_build_features.py` (feeds `_3s.csv`, and `_gt_3s.csv` via `03_build_game_features.py` adding columns on top) | `28_horizon1.py` (1s horizon, matches committed exactly) |
|---|---|---|
| Sampler | `kinematic.py::build_samples()` (line 30, self-labeled `"구 프로토콜(cross-vehicle) 샘플러"`) | `sampling.py::build_samples_within()` |
| `require_both` | **Not a parameter of this function at all** — no per-vehicle "has both pos and neg" filter exists | Explicit `require_both=True` (script line 47) |
| `margin_sec` | **Not a parameter of this function at all.** Negative candidates are instead spaced by a fixed frame stride `lk_interval=25` (function default, kinematic.py line 30), independent of horizon length and with no explicit exclusion zone around event frames | Explicit `margin_sec=MARGIN=2.0` (script line 27; passed into `build_samples_within`, which converts it to a frame-based exclusion window around every event, `sampling.py` lines 41, 86–96) |
| `use_road_frame` default | `build_features()` called with no override (`02_build_features.py` line 41: `build_features(s, canon)`) → uses the function's own default, `use_road_frame=True` (`kinematic.py` line 64 signature) | Explicit `use_road_frame=True` (script line 51) — same default, explicitly stated |
| Samples per vehicle | Positives: exactly 1 per detected lane-change event (`event_frame − H`, kinematic.py lines 34–36). Negatives: one candidate **every 25 frames** for as long as the lane stays constant across the full horizon (lines 40–42) — count scales with each vehicle's trajectory length ÷ 25, not with its event count | Positives: 1 per event (same idea). Negatives: capped at `neg_per_pos=3` per positive, with a minimum temporal spacing of `neg_min_spacing_sec=1.0` between negatives (`sampling.py` lines 27, 113–121) — explicitly bounded, not stride-driven |
| Group-split criterion downstream | `02_build_features.py` itself does **no group split** — it only balances pos:neg 1:1 twice (once per-file via `balance()`, once globally in `main()`, lines 47–53). Whatever script consumes `_gt_3s.csv` later (e.g. `diag_shortcut.py`, `23_ood_ci.py`) applies its own group split independently | `28_horizon1.py` calls `split_by_vehicle()` from `src/models/train.py` (script line 82) directly inside the same script — sampling and the eventual train/test split are done together in one pass |

**Mechanism assessment (reported, not concluded)**: `build_samples()`'s negative
generator produces one candidate per 25-frame stride across essentially a vehicle's
entire lane-constant trajectory, with no cap tied to that vehicle's own event count
and no per-event exclusion margin — so its raw negative-candidate pool size scales
with total available lane-constant frame-time in the dataset. `build_samples_within`
instead hard-caps negatives at `3×` the positive count per vehicle and enforces a 2 s
exclusion margin around every event. Because both paths get downstream-rebalanced to
roughly 1:1 pos:neg, the **final row count is governed mainly by the positive
(event) count** for `build_samples()`-based files, whereas
`build_samples_within`-based files are governed by whichever is smaller: `3×`
positives, or the actually-available margin-respecting negative pool. This is a
structural, code-level explanation for why row counts *could* diverge between the
two samplers given the same input, but **it does not by itself explain why exiD's
`build_samples()`-based row count grew 4.29× between the 07-14 vintage and the
09-15 regen**, since both runs used the identical `build_samples()`/`balance()` code
(confirmed byte-identical against the `/home/futurenetworklab/gavogo` reference
copy — see §5) applied, per §2, to a deterministically-reproduced canonical parquet
layer. **The specific cause of the row-count growth between the two `build_samples()`
vintages could not be confirmed — 확인 불가.** The two candidate explanations that
would be consistent with everything observed — (a) `events_{ds}.csv` (lane-change
event counts) differed between the two vintages despite unchanged code and unchanged
canonical parquet content, for a reason not identified here, or (b) the set of raw
recordings actually processed by `01_build_events.py` differed between vintages
(e.g. `config.DEFAULT_RECS` coverage, or which files existed under `data/raw/` at
each point in time) — were not distinguishable with the evidence available in this
pass, since no snapshot of the 07-14-vintage `events_{ds}.csv` or `data/raw/` exists
to compare against.

## 5. `14_adapter_ablation.py`: which files V1–V4 read

`scripts/14_adapter_ablation.py::load()` (lines 44–50):

```python
def load(ds, suffix, use_game):
    name = f"{ds}_gt_{H}s.csv" if use_game else f"{ds}{suffix}_{H}s.csv"
    ...
```

`VARIANTS` (lines 34–39):

| Variant | `suffix` | `use_game` | File actually read |
|---|---|---|---|
| V1 Raw | `_V1` | `False` | `{ds}_V1_3s.csv` |
| V2 + Neighbor reconstruction | `_V2` | `False` | `{ds}_V2_3s.csv` |
| V3 + Normalization | `_V3` | `False` | `{ds}_V3_3s.csv` |
| V4 Full GAVOGO (+ game theory) | `""` | `True` | `{ds}_gt_3s.csv` |

**V4 reads a genuinely different file than V1–V3** — not the same file with a
different column subset. V1/V2/V3 read the `2026-07-14`-vintage, never-regenerated
files documented in §1 (row counts e.g. exiD 82,924). V4 reads `{ds}_gt_3s.csv`,
the `2026-09-15`-vintage file freshly produced by PHASE 1 (row counts e.g. exiD
355,802) — the same file traced in §3. This exactly and fully explains the already-
documented pattern (`regen_diff.md` Table 10): V1–V3 reproduce the committed
`adapter_ablation.csv` to ≤0.0005 because they are reading **bit-identical,
untouched input files** (PHASE 1's script list, per `canonical.txt`, includes
`14_adapter_ablation.py` itself but never re-invokes
`02_build_features_v2.py --suffix _V1/_V2/_V3` to regenerate their inputs — those
inputs were never part of the PHASE-1 regeneration run), while V4 reads input that
**was** regenerated and has a substantially different row count.

---

## Summary of what remains confirmed vs. 확인 불가

**Confirmed, exact, reproducible:**
- §1: `_V1/_V2/_V3` files dated 2026-07-14, untouched by PHASE 1; plain/`_gt`/`_gtdi`
  files dated 2026-09-15, all with equal-to-or-larger row counts (up to 4.29× for
  exiD).
- §2: `dataset_stats.csv` byte-identical committed vs. regen; fully explained by
  `29_dataset_stats.py` reading only the canonical parquet layer, which is
  deterministically rebuilt from unchanged raw input by unchanged code.
- §3: regen pooled n (371,998) = exact sum of today's 4 target `_gt_3s.csv` row
  counts. The same arithmetic, using the `_V1_3s.csv` row counts for the same 4
  datasets, exactly equals the committed pooled n (95,728).
- §5: V4 reads a different, regenerated file (`_gt_3s.csv`) than V1–V3 (frozen
  `_V1/_V2/_V3_3s.csv`) — fully explains why only V4 diverges from committed.
- §4: the two sampler functions (`build_samples` vs. `build_samples_within`) are
  structurally different in every dimension checked (require_both, margin, negative
  count logic), confirmed by direct code reading.

**확인 불가 (could not confirm):**
- Why `build_samples()`'s output row count itself grew between the 07-14 and 09-15
  vintages, given identical code (verified byte-identical against an independent
  reference copy) and identical upstream canonical-parquet content (verified via
  §2's byte-identical `dataset_stats.csv`). No snapshot of the 07-14-vintage
  `events_{ds}.csv` or of `data/raw/` at that time exists to test the two candidate
  explanations offered in §4.
- Whether `results_committed_backup` (mentioned in `regen_timing.log`, no longer on
  disk) ever contained anything beyond a copy of `results/`.
- Whether `/home/futurenetworklab/gavogo`'s separate git history has any temporal
  relationship to when this repo's committed `results/tables/*.csv` were produced.

No conclusion is offered on which vintage (07-14 or 09-15) is "correct" — per
instruction, only what differs and the traced mechanism is reported.
