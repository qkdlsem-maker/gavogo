# PHASE 6 — is the regenerated feature data leak-free?

Per `leak.txt`: read-only except for verification output confined to a new isolated
directory (`data/processed_v2test/`, created solely for this check; nothing under
`data/processed/` or any `results*/` tree was overwritten). No `.docx` opened. Nothing
committed. Numbers are reported as computed; no interpretation is offered on which
protocol is "correct" beyond stating the mechanical facts below.

---

## 1. Group-label purity of the regenerated `data/processed/{ds}_gt_3s.csv` files

Function used: **`scripts/32_purity_auc.py :: purity(s)`** (line 20–22), loaded via
`importlib.util.spec_from_file_location` and called directly — not reimplemented.
Group key inside that function is `(recording_id, vehicle_id)`, applied per-dataset
(the file is already single-dataset, so `dataset` doesn't need to be a separate key
component).

| Dataset | n rows | group_label_purity |
|---|---|---|
| highD | 21,070 | **0.9620** |
| NGSIM | 4,270 | **0.7862** |
| MiTra | 27,500 | **0.6732** |
| ETRI | 322 | **0.7869** |
| EMT | 2,904 | **0.7204** |
| uniD | 12,970 | **0.6484** |
| exiD | 355,802 | **0.1848** |

The manuscript claims the leak-free benchmark's purity is 0.000. **None of the 7
regenerated `_gt_3s.csv` files have purity = 0** — values range 0.1848 (exiD) to
0.9620 (highD). Reported exactly as computed.

## 2. `02_build_features.py` vs `02_build_features_v2.py`, side by side

| | `scripts/02_build_features.py` | `scripts/02_build_features_v2.py` |
|---|---|---|
| mtime | **2026-07-06 10:18** | **2026-07-14 10:58** |
| Sampler imported | `from src.features.kinematic import build_samples, balance, ...` (line 11) — the function `kinematic.py` itself labels, in a comment at its own definition (line 29), `"구 프로토콜(cross-vehicle) 샘플러. 비교/재현용"` (old-protocol cross-vehicle sampler, for comparison/reproduction) | `from src.features.sampling import build_samples_within, balance_within, report_purity` (line 29) |
| Call site | `build_samples(canon, events, H, random_state=config.RANDOM_STATE)` (line 41) | `build_samples_within(canon, events, H, fps=fps, margin_sec=a.margin_sec, neg_per_pos=a.neg_per_pos, require_both=True, random_state=config.RANDOM_STATE)` (lines 90–93) |
| `require_both` | **Not a parameter of `build_samples()` at all** — the function signature (`kinematic.py` line 30: `build_samples(canon, events, horizon_frames, lk_interval=25, random_state=42)`) has no such concept | Explicit, hardcoded `True` (line 93) |
| `margin_sec` | **Not a parameter of `build_samples()` at all** | Explicit CLI arg, default `2.0` (`ap.add_argument("--margin_sec", type=float, default=2.0)`, line 55) |
| `neg_per_pos` | **Not a parameter** — negatives instead come from a fixed 25-frame stride (`lk_interval=25` default) across the vehicle's lane-constant trajectory, independent of that vehicle's own event count | Explicit CLI arg, default `3` (`ap.add_argument("--neg_per_pos", type=int, default=3)`, line 54) |
| Output filename, no extra args | `config.PROCESSED_DIR / f"{a.dataset}_{hsec}s.csv"` (line 59) — plain, e.g. `highD_3s.csv` | `config.PROCESSED_DIR / f"{a.dataset}{a.suffix}_{hsec}s.csv"` with `a.suffix` defaulting to `""` (line 57: `ap.add_argument("--suffix", default="")`) — **also plain `highD_3s.csv` when run with no `--suffix`** |

**Filename collision, noted as a mechanical fact:** both scripts write to the
identical plain filename (`{dataset}_{horizon}s.csv`) when invoked with no special
flags. Nothing in the output filename indicates which sampler produced it — running
either script overwrites whatever the other one most recently wrote, silently.

## 3. Was `02_build_features_v2.py` in `canonical.txt`'s PHASE 1 script list?

`canonical.txt` specified this exact PHASE 1 execution order (quoted from the file,
read directly earlier in this session; the file has since been removed from the
directory, so this is transcribed from that direct reading, not reconstructed):

```
scripts/01_build_events.py
scripts/02_build_features.py
scripts/03_build_game_features.py
scripts/03b_build_game_di.py
[... then 04_train.py, 14_adapter_ablation.py, 16_domain_invariant.py, 34_coral.py,
17_game_di_eval.py, 18_tiv_remaining.py, 19_baselines_all.py, 05_domain_adapt_v2.py,
20_adapt_control.py, 21_margin_sweep.py, 22_discard_audit.py, 23_ood_ci.py,
24_roadtype.py, 28_horizon1.py, 29_dataset_stats.py, 30_lodo7.py,
31_simple_baselines.py, 32_purity_auc.py, 33_domain_confusion.py, diag_purity_all.py,
diag_shortcut.py, 26_leakage_quant.py, 27_conventional_leakage.py, 11_significance.py,
12_extras.py, 13_dl_latency.py, 15_loco.py, 07_lodo.py ]
```

**`02_build_features_v2.py` does not appear anywhere in this list.** PHASE 1 ran
`02_build_features.py` (the old cross-vehicle sampler) for the base feature-building
step; `02_build_features_v2.py` was never invoked during PHASE 1.

## 4. Verification run: `02_build_features_v2.py`, no `--suffix`, all 7 datasets

Run via an isolated-config driver (`config.PROCESSED_DIR` monkeypatched to
`data/processed_v2test/`, `config.TABLES_DIR` to
`data/processed_v2test/tables_v2test/`, before invoking the script — verified before
the full run, on ETRI alone, that this leaves real `data/processed/` and `results*/`
file mtimes unchanged). Row counts obtained directly via `wc -l` on the isolated
output, and independently cross-checked against `phase5_rootcause.md`'s already-
documented `_V1_3s.csv` counts:

| Dataset | `02_build_features_v2.py` (no suffix) row count | `_V1_3s.csv` row count | Match? |
|---|---|---|---|
| highD | 20,514 | 20,514 | **exact** |
| NGSIM | 4,268 | 4,268 | **exact** |
| MiTra | 27,465 | 27,465 | **exact** |
| ETRI | 261 | 261 | **exact** |
| EMT | 1,444 | 1,444 | **exact** |
| uniD | 11,099 | 11,099 | **exact** |
| exiD | 82,924 | 82,924 | **exact** |

**All 7 of 7 match exactly.** Running `02_build_features_v2.py` with no flags
reproduces the frozen 2026-07-14 `_V1_3s.csv` row counts precisely, for every
dataset.

## 5. Game features + `23_ood_ci.py` on the leak-free-sampler output

`03_build_game_features.py` was run (same isolated-config driver) on top of the
step-4 output, producing `data/processed_v2test/{ds}_gt_3s.csv` for all 7 datasets.
`23_ood_ci.py` was then run the same way, reading exclusively from
`data/processed_v2test/` and writing `ood_ci.csv` to
`data/processed_v2test/tables_v2test/` (verified: this directory is separate from
both `results/tables/` and `results_regen/tables/`; neither was touched).

Actual output (`data/processed_v2test/tables_v2test/ood_ci.csv`), reported exactly
as produced, not adjusted:

| Target | n | AUC | 95% CI |
|---|---|---|---|
| ETRI | 261 | 0.4931 | [0.4173, 0.5650] |
| EMT | 1,444 | 0.5765 | [0.5465, 0.6057] |
| uniD | 11,099 | 0.5021 | [0.4909, 0.5128] |
| exiD | 82,924 | 0.4924 | [0.4884, 0.4964] |
| **POOLED** | **95,728** | **0.4968** | **[0.4930, 0.5006]** |

Manuscript's claimed values, for reference: n=95,728, ETRI 0.481, EMT 0.583, uniD
0.499, exiD 0.480 (pooled AUC not independently restated here beyond what
`regen_diff.md` already recorded as 0.486).

**Pooled n = 95,728 is an exact match to the committed/manuscript value.** Per-target
AUCs: ETRI 0.4931 vs. 0.481 (diff 0.012), EMT 0.5765 vs. 0.583 (diff 0.0065), uniD
0.5021 vs. 0.499 (diff 0.003), exiD 0.4924 vs. 0.480 (diff 0.012) — all five numbers
(4 targets + pooled) land in the same near-chance band (0.49–0.58) as the
manuscript's claims, in contrast to the PHASE-1-regen `ood_ci.csv`
(`results_regen/tables/ood_ci.csv`, already documented in `regen_diff.md` Table A.3)
which reported a pooled AUC of 0.5416 on n=371,998 with ETRI/EMT/uniD/exiD all
0.53–0.68.

## 6. Does `README.md`'s Pipeline section document `02_build_features.py` or `_v2.py`?

`README.md` line 88:
```
python scripts/02_build_features.py      # canonical + events → 34 kinematic/spatial/lateral features
```
**`README.md` documents `02_build_features.py`** (the old cross-vehicle sampler), not
`02_build_features_v2.py`. `02_build_features_v2.py` is not mentioned anywhere in the
Pipeline section (checked full section, lines 84 onward).

---

## Summary of what was measured (no verdict offered beyond the numbers)

- §1: regenerated `_gt_3s.csv` purity is non-zero for all 7 datasets (0.18–0.96),
  against the manuscript's claimed 0.000 for its leak-free benchmark.
- §2: the two scripts use structurally different sampler functions; only
  `02_build_features_v2.py`'s sampler has `require_both`/`margin_sec`/`neg_per_pos`
  concepts at all. Both default to writing the same plain filename with no
  disambiguating suffix.
- §3: `02_build_features_v2.py` was absent from PHASE 1's script list; PHASE 1 ran
  `02_build_features.py`.
- §4: `02_build_features_v2.py` (no flags) reproduces `_V1_3s.csv` row counts exactly
  for all 7 datasets.
- §5: running the pipeline through `02_build_features_v2.py` → `03_build_game_features.py`
  → `23_ood_ci.py` (isolated directory, nothing overwritten) yields pooled n=95,728
  (exact match to committed) and 4 target AUCs within 0.003–0.012 of the manuscript's
  claimed values — a materially different result from PHASE 1's regenerated
  `results_regen/tables/ood_ci.csv`.
- §6: `README.md`'s documented pipeline step is `02_build_features.py`, not `_v2.py`.

Not investigated here (out of scope for PHASE 6 as instructed): why `README.md` and
`canonical.txt` both point to `02_build_features.py` if `02_build_features_v2.py`
produces the sample counts matching the frozen/committed files — i.e., which script
was actually used to produce the *original* committed results is not directly
observed here, only inferred from the row-count/purity/AUC arithmetic above.
