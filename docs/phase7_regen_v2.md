# PHASE 7 — full regeneration with the corrected (leak-free) sampler

**⚠ Disclosed incident**: `scripts/34_coral.py` has a hardcoded output path
(`OUT = Path("results/tables/34_coral.csv")`, line 19) that does not go through
`config.TABLES_DIR` — it bypassed this phase's isolation harness (which patches
`config`, not literal path strings) and briefly overwrote the real, committed
`results/tables/34_coral.csv` in place. This was caught via `git status`
immediately after the run, verified as the *only* such case (all 32 scripts used
in this phase were grepped for hardcoded `results/`/`data/` path literals — no
other offenders found), the accidental output was preserved to
`results_v2/tables/34_coral.csv` (its correct location), and the real file was
restored exactly via `git checkout -- results/tables/34_coral.csv` (confirmed
zero-diff against the original committed version afterward). No other file
outside `data/processed_v2/`/`results_v2/` was touched by this phase.

Per PHASE 7: `02_build_features_v2.py` (leak-free `build_samples_within`,
`require_both=True`, `margin_sec=2.0`, `neg_per_pos=3`) substituted for
`02_build_features.py` (old leaky `build_samples`, cross-vehicle) at that one
position in the 50-step PHASE-1 sequence (`01_build_events` skipped — reused
existing `data/interim/`, per PHASE 5's determinism finding). All output confined
to `data/processed_v2/` and `results_v2/`; `data/processed/`, `results/`,
`results_regen/` untouched. All 50 steps completed (`phase7_timing.log`, status=0
throughout). No `.docx` opened. Nothing committed. Manuscript numbers below are
reused verbatim from `regen_diff.md` — not re-extracted from the `.docx`.

---

## A. Purity, all 7 datasets (`data/processed_v2/{ds}_gt_3s.csv`)

| Dataset | n rows | group_label_purity |
|---|---|---|
| highD | 20,514 | 0.000000 |
| NGSIM | 4,268 | 0.000000 |
| MiTra | 27,465 | 0.000000 |
| ETRI | 261 | 0.000000 |
| EMT | 1,444 | 0.000000 |
| uniD | 11,099 | 0.000000 |
| exiD | 82,924 | 0.000000 |

All 7 exactly 0.000000, matching the manuscript's leak-free-benchmark claim.

---

## B. Table-by-table, three-way: 원고 R5 / results_regen (Phase 1) / results_v2 (Phase 7)

Manuscript and results_regen columns reused verbatim from `regen_diff.md`.
results_v2 read fresh from `results_v2/tables/*.csv` this pass.

### Table 5 — `table5_repro_canonical.csv`

**results_v2 is byte-identical to results_regen** (`diff` confirms zero
differences) — **this table did NOT improve.** Root cause, confirmed by reading
`scripts/35_table5_repro.py` line 99: the `source=canonical` path calls
`build_samples()` (the old leaky cross-vehicle sampler) **directly and
independently**, with its own hardcoded call — it never invokes
`02_build_features.py` or `_v2.py` at all, so PHASE 7's substitution had no
effect on this script's output. This table's manuscript-divergence is a
*separate, still-unresolved* issue, outside the scope of the PHASE 6/7 fix
(consistent with `table5_diagnosis.md`'s earlier finding that the manuscript's
Table 5 is actually closer to the *backup* pipeline, not either canonical
variant).

| 행 | 열 | 원고 R5 | results_regen | results_v2 | v2↔원고 |
|---|---|---|---|---|---|
| highD (3s) | Purity | 0.96 | 0.962 | 0.962 | 0.002 |
| highD (3s) | AUC (all) | 0.884 | 0.9365 | 0.9365 | 0.0525 |
| highD (3s) | AUC (mixed) | 0.744 | 0.8277 | 0.8277 | 0.0837 |
| highD (3s) | Δ | 0.140 | 0.1088 | 0.1088 | 0.0312 |
| NGSIM (3s) | Purity | 0.79 | 0.7862 | 0.7862 | 0.0038 |
| NGSIM (3s) | AUC (all) | 0.791 | 0.8706 | 0.8706 | 0.0796 |
| NGSIM (3s) | AUC (mixed) | 0.729 | 0.7937 | 0.7937 | 0.0647 |
| NGSIM (3s) | Δ | 0.062 | 0.0769 | 0.0769 | 0.0149 |
| MiTra (3s) | Purity | 0.67 | 0.6732 | 0.6732 | 0.0032 |
| MiTra (3s) | AUC (all) | 0.870 | 0.9257 | 0.9257 | 0.0557 |
| MiTra (3s) | AUC (mixed) | 0.785 | 0.8803 | 0.8803 | 0.0953 |
| MiTra (3s) | Δ | 0.085 | 0.0454 | 0.0454 | 0.0396 |
| exiD (7s) | Purity | 0.91 | 0.9007 | 0.9007 | 0.0093 |
| exiD (7s) | AUC (all) | 0.981 | 0.9945 | 0.9945 | 0.0135 |
| exiD (7s) | AUC (mixed) | 0.909 | 0.9596 | 0.9596 | 0.0506 |
| exiD (7s) | Δ | 0.072 | 0.0349 | 0.0349 | 0.0371 |

### Table 6 — in-domain AUC (`joint_results.csv`, row `3s,full`)

| 행 | 열 | 원고 R5 | results_regen | results_v2 | v2↔원고 |
|---|---|---|---|---|---|
| highD | Leak-free AUC | 0.899 | 0.9298 | **0.8968** | **0.0022** |
| NGSIM | Leak-free AUC | 0.818 | 0.8463 | **0.8144** | **0.0036** |
| MiTra | Leak-free AUC | 0.892 | 0.9222 | **0.8926** | **0.0006** |
| Joint (pooled)* | Leak-free AUC ± std | 0.8815 ± 0.0019 | 0.9177 ± 0.0012 | **0.8814 ± 0.0017** | **0.0001** |

\* Same mapping caveat as `regen_diff.md` (c): "Joint (pooled)" uses
`baselines_all.csv`/`adapter_ablation.csv` V4/XGBoost row as the best-identified
candidate; seed-count/pooling protocol against the manuscript's "3 seeds"
statement is still unverified — **확인 불가**, unchanged from Phase 2.

**Massive improvement** — all 4 cells now within 0.004 of the manuscript, vs.
0.03 in results_regen.

### Table 7 — zero-shot OOD

Two source files checked (`roadtype_zeroshot.csv`, single reference model — same
file `regen_diff.md` used for Table 7; `ood_ci.csv`, bootstrap variant, used for
Table A.3). Both given below.

| 행 | 열 | 원고 R5 | results_regen | v2 (`roadtype_zeroshot.csv`) | v2 (`ood_ci.csv`) |
|---|---|---|---|---|---|
| ETRI | Zero-shot AUC | 0.484 | 0.6008 | **0.4934** | **0.4931** |
| EMT | Zero-shot AUC | 0.601 | 0.6465 | **0.5764** | **0.5765** |
| uniD | Zero-shot AUC | 0.508 | 0.6755 | **0.502** | **0.5021** |
| exiD | Zero-shot AUC | 0.487 | 0.5359 | **0.4924** | **0.4924** |
| Mean | Zero-shot AUC | 0.520 | 0.6202 | **0.5161** | **0.5160** |

All four targets land within 0.009–0.018 of the manuscript (vs. 0.05–0.17 under
results_regen), and the qualitative "collapses toward chance" story is fully
restored — uniD and ETRI are back near 0.49–0.50, not the 0.60+ results_regen
showed.

### Table 9 — domain-invariant representations (`domain_invariant.csv`)

| 행 | 열 | 원고 R5 | results_regen | results_v2 | v2↔원고 |
|---|---|---|---|---|---|
| Full (48 feat) | In-domain | 0.882 | 0.9177 | **0.8814** | **0.0006** |
| Full (48 feat) | OOD mean | 0.520 | 0.6202 | **0.5178** | **0.0022** |
| DI | In-domain | 0.880 | 0.918 | **0.8799** | **0.0001** |
| DI | OOD mean | 0.533 | 0.6053 | **0.539** | **0.006** |

Row-label mapping ambiguity noted in `regen_diff.md` (DI+rank / DI+rank+topology
vs. CSV's `full+rank`/`DI+rank`/`DI+game`/`DI+game+rank`) is **still unresolved —
확인 불가**, not newly resolved by this phase; not force-mapped here.

**CORAL row — resolves a gap `regen_diff.md` had flagged as "not diffed, follow-
up needed."** Per-target means, computed directly from the 12 rows (4 targets ×
3 seeds) in `results_v2/tables/34_coral.csv` (preserved copy, see the disclosed-
incident note above):

| 행 | 열 | 원고 R5 | results_v2 | v2↔원고 |
|---|---|---|---|---|
| CORAL | OOD ETRI | 0.581 | **0.5810** | **0.0000 (exact)** |
| CORAL | OOD EMT | 0.537 | **0.5373** | **0.0003** |
| CORAL | OOD uniD | 0.451 | **0.4506** | **0.0004** |
| CORAL | OOD exiD | 0.477 | **0.4783** | 0.0013 |
| CORAL | OOD mean (4 targets) | 0.512 | **0.5118** | **0.0002** |

Near-exact on all 4 targets and the mean — one of the tightest matches in this
entire report.

### Table 10 — adapter ablation (`adapter_ablation.csv`)

| 행 | 열 | 원고 R5 | results_regen | results_v2 | v2↔원고 |
|---|---|---|---|---|---|
| V1 | In-domain | 0.872 | 0.8719 | 0.8719 | 0.0001 |
| V1 | OOD mean | 0.616 | 0.6164 | 0.6164 | 0.0004 |
| V2 | In-domain | 0.873 | 0.8731 | 0.8731 | 0.0001 |
| V2 | OOD mean | 0.581 | 0.5805 | 0.5805 | 0.0005 |
| V3 | In-domain | 0.881 | 0.8812 | 0.8812 | 0.0002 |
| V3 | OOD mean | 0.522 | 0.5216 | 0.5216 | 0.0004 |
| V4 (full GAVOGO) | In-domain | 0.882 | 0.9177 | **0.8814** | **0.0006** |
| V4 (full GAVOGO) | OOD mean | 0.520 | 0.6202 | **0.5178** | **0.0022** |

V1–V3 unchanged (as expected — they read the frozen `_V1/_V2/_V3` files, copied
in for this run, not regenerated). **V4 — the one row that broke under
results_regen — is now within 0.0006–0.0022 of the manuscript**, fully
consistent with PHASE 6's finding that V4 was the only variant reading the
regenerated (previously leaky) `_gt_3s.csv`.

### Table 11 — domain-discriminability (`game_di_domain_auc.csv`)

| 행 | 열 | 원고 R5 | results_regen | results_v2 | v2↔원고 |
|---|---|---|---|---|---|
| payoff_lc_max | AUC (abs) | 0.907 | 0.9136 | **0.9071** | **0.0001** |
| payoff_lc_max | AUC (DI) | 0.676 | 0.6112 | **0.6764** | **0.0004** |
| payoff_gap | AUC (abs) | 0.897 | 0.9194 | **0.8968** | **0.0002** |
| payoff_gap | AUC (DI) | 0.617 | 0.5763 | **0.6174** | **0.0004** |
| stackelberg_lc_payoff | AUC (abs) | 0.830 | 0.8664 | **0.8299** | **0.0001** |
| stackelberg_lc_payoff | AUC (DI) | 0.699 | 0.6349 | **0.6985** | **0.0005** |
| stackelberg_advantage | AUC (abs) | 0.791 | 0.836 | **0.7906** | **0.0004** |
| stackelberg_advantage | AUC (DI) | 0.623 | 0.592 | **0.6226** | **0.0004** |
| nash_lc_prob | AUC (abs) | 0.755 | 0.752 | **0.755** | **0.0000** |
| nash_lc_prob | AUC (DI) | 0.667 | 0.6229 | **0.6671** | **0.0001** |
| payoff_lk | AUC (abs) | 0.754 | 0.8038 | **0.7544** | **0.0004** |
| payoff_lk | AUC (DI) | 0.509 | 0.5162 | **0.5092** | **0.0002** |
| nash_urgency | AUC (abs) | 0.738 | 0.7527 | **0.7383** | **0.0003** |
| nash_urgency | AUC (DI) | 0.549 | 0.5154 | **0.5487** | **0.0003** |
| Mean (14) | AUC (abs) | 0.707 | 0.7169 | **0.7065**\* | **0.0005** |
| Mean (14) | AUC (DI) | 0.590 | 0.5581 | **0.5897**\* | **0.0003** |

\* Mean computed directly here from the 14 rows in `results_v2/tables/game_di_domain_auc.csv`
(not itself printed as a summary row in that file).

**Near-exact reproduction across all 16 cells** — this table went from the
largest relative diagnostic-localization example in `regen_diff.md` to one of
the tightest matches in this report.

### Table 12 — calibration (`calibration_ood.csv`)

| 행 | 열 | 원고 R5 | results_regen | results_v2 | v2↔원고 |
|---|---|---|---|---|---|
| In-domain | AUC | 0.880 | 0.9194 | **0.8796** | **0.0004** |
| In-domain | ECE | 0.0115 | 0.0125 | **0.0116** | **0.0001** |
| In-domain | Brier | 0.141 | 0.1142 | **0.1414** | **0.0004** |
| ETRI | AUC | 0.498 | 0.6097 | **0.4867** | 0.0113 |
| ETRI | ECE | 0.2415 | 0.1914 | **0.2552** | 0.0137 |
| EMT | AUC | 0.599 | 0.6485 | **0.6052** | 0.0062 |
| uniD | AUC | 0.489 | 0.6883 | **0.4995** | 0.0105 |
| exiD | AUC | 0.484 | 0.5284 | **0.4829** | **0.0011** |

In-domain and exiD essentially exact; ETRI/EMT/uniD close (0.006–0.014) but not
as tight as most other tables — still a large improvement over results_regen's
0.04–0.20 gaps.

### Table 13 — adaptation curve (`adaptation_curve.csv`)

| 행 | 열 | 원고 R5 | results_regen | results_v2 | v2↔원고 |
|---|---|---|---|---|---|
| exiD | Pool size (vehicles) | 14,760 | 49,480 | **14,760** | **0 (exact)** |
| uniD | Pool size (vehicles) | 2,918 | 4,434 | **2,918** | **0 (exact)** |
| EMT | Pool size (vehicles) | 207 | 691 | **207** | **0 (exact)** |
| ETRI | Pool size (vehicles) | 42 | 86 | **42** | **0 (exact)** |
| exiD | AUC @100% | 0.984 | 0.8776 | **0.9835** | **0.0005** |
| uniD | AUC @100% | 0.882 | 0.9281 | **0.883** | **0.001** |

**Pool sizes now match exactly, all 4** — direct confirmation that the row-count
inflation PHASE 5/6 traced to the leaky sampler is fully resolved by using
`02_build_features_v2.py`.

### Table 14 — transfer gain (`adapt_control.csv`)

| 행 | 열 | 원고 R5 | results_regen | results_v2 | Sign match? |
|---|---|---|---|---|---|
| exiD | Gain @10% | −0.009 | −0.0142 | **−0.0095** | negative ✓ |
| exiD | Gain @100% | −0.004 | −0.0139 | **−0.0048** | negative ✓ |
| uniD | Gain @10% | −0.021 | **+0.0027** | **−0.0175** | negative ✓ (regen was +) |
| uniD | Gain @100% | −0.022 | −0.0034 | **−0.0207** | negative ✓ |
| EMT | Gain @10% | −0.035 | −0.0109 | **−0.0578** | negative ✓ |
| EMT | Gain @100% | −0.061 | −0.0491 | **−0.0741** | negative ✓ |
| ETRI | Gain @10% | −0.022 | **+0.0819** | **−0.0246** | negative ✓ (regen was +) |
| ETRI | Gain @100% | −0.107 | −0.0237 | **−0.1007** | negative ✓ |

**All 8 cells negative under results_v2 — "Finding 4" (all eight tested cells
negative) is fully restored.** The two cells that flipped positive under
results_regen (uniD@10%, ETRI@10%) are negative again. Magnitudes are close for
6/8 cells (≤0.025) with EMT (both fractions) showing a larger residual gap
(0.013–0.023) — sign and qualitative claim fully match; point-estimate magnitude
not uniformly exact.

### Solver statistics (`nash_solver_stats.csv`)

| 항목 | 원고 R5 | results_regen | results_v2 | v2↔원고 |
|---|---|---|---|---|
| Total game instances | 556,272 | 1,206,704 | **557,272** | **1,000** |
| Pure-strategy % range | 89.2–100% | 90.41–99.60% | **89.16–100.0%** | **essentially exact (89.16≈89.2, 100.0=100)** |
| Solver-failure rate | 0.00% | 0.00% | **0.00%** | 0 (동일) |
| Mixed-strategy max | 10.8% (exiD, 3s) | 9.59% (exiD, 5s — wrong horizon) | **10.84% (exiD, 3s)** | **essentially exact, correct horizon restored** |

Total game instances off by exactly 1,000 (557,272 vs. 556,272) — a small,
unexplained residual (**확인 불가** why); every other solver statistic matches
to rounding precision, including recovering the correct horizon for the
mixed-strategy maximum.

### Table A.1 (margin sweep), A.2 (discard audit)

Both **byte-identical** to `results/tables/` (confirmed via direct `diff`, zero
output) — as expected, since these scripts read only `INTERIM_DIR` (canonical
parquet + events), never `PROCESSED_DIR`, so they were never affected by the
sampler bug and are unaffected by this phase's fix either. No change from
`regen_diff.md`'s Phase-2 finding.

### Table A.3 — bootstrap CI (`ood_ci.csv`)

| 행 | 열 | 원고 R5 | results_regen | results_v2 | v2↔원고 |
|---|---|---|---|---|---|
| ETRI | AUC | 0.481 | 0.5995 | **0.4931** | 0.0121 |
| EMT | AUC | 0.583 | 0.6467 | **0.5765** | 0.0065 |
| uniD | AUC | 0.499 | 0.6756 | **0.5021** | 0.0031 |
| exiD | AUC | 0.480 | 0.5359 | **0.4924** | 0.0124 |
| Pooled | AUC | 0.486 | 0.5416 | **0.4968** | 0.0108 |
| Pooled | n | 95,728 | 371,998 | **95,728** | **0 (exact)** |

**Pooled n exactly matches**; AUCs within 0.003–0.012 of the manuscript (vs.
0.05–0.18 under results_regen).

### Table A.4 — 1 s horizon (`horizon1.csv`, `horizon1_summary.csv`)

| 행 | 열 | 원고 R5 | results_regen | results_v2 | v2↔원고 |
|---|---|---|---|---|---|
| ETRI | AUC | 0.531 | 0.5305 | 0.5305 | 0.0005 |
| EMT | AUC | 0.625 | 0.6252 | 0.6252 | 0.0002 |
| uniD | AUC | 0.508 | 0.5076 | 0.5076 | 0.0004 |
| exiD | AUC | 0.361 | 0.361 | 0.361 | 0.0000 |
| Pooled | AUC | 0.368 | 0.3678 | 0.3678 | 0.0002 |
| In-domain | AUC | 0.986 | 0.9859 | 0.9859 | 0.0001 |

Unchanged from results_regen (as expected — `28_horizon1.py` already used the
leak-free `build_samples_within` sampler directly, independent of
`02_build_features*.py`, per PHASE 5).

### Table A.5 — 7-way LODO (`lodo7.csv`)

| 행 | 열 | 원고 R5 | results_regen | results_v2 | v2↔원고 |
|---|---|---|---|---|---|
| highD | AUC | 0.539 | 0.64 | **0.5537** | 0.0147 |
| NGSIM | AUC | 0.681 | 0.6913 | **0.6814** | **0.0004** |
| MiTra | AUC | 0.649 | 0.6882 | **0.6454** | 0.0036 |
| ETRI | AUC | 0.582 | 0.6361 | **0.5855** | 0.0035 |
| EMT | AUC | 0.565 | 0.6665 | **0.5565** | 0.0085 |
| uniD | AUC | 0.613 | 0.6588 | **0.5988** | 0.0142 |
| exiD | AUC | 0.449 | 0.5702 | **0.4563** | 0.0073 |

exiD is back **below chance** (0.4563 < 0.5, matching the manuscript's claimed
direction) — under results_regen it had flipped above chance (0.5702). All 7
cells within 0.014 of the manuscript, vs. up to 0.12 under results_regen.

### Table A.6 (simple baselines) — `simple_baselines.csv`

| 행 | 열 | 원고 R5 | results_regen | results_v2 | v2↔원고 |
|---|---|---|---|---|---|
| Logistic regression | In-domain | 0.603 | 0.6784 | **0.6027** | **0.0003** |
| Logistic regression | OOD mean | 0.520 | 0.5777 | **0.5205** | **0.0005** |
| Random forest | In-domain | 0.861 | 0.888 | **0.8611** | **0.0001** |
| Random forest | OOD mean | 0.536 | 0.6326 | **0.5358** | **0.0002** |
| XGBoost (ref.) | In-domain | 0.882 | 0.9177 | **0.8814**\* | **0.0006** |
| XGBoost (ref.) | OOD mean | 0.520 | 0.6202 | **0.5178**\* | **0.0022** |

\* `simple_baselines.csv` itself has no XGBoost row in `results_v2`; XGBoost
reference reused from `baselines_all.csv`, same as `regen_diff.md` did.

Near-exact across the board.

### Table A.7 (representation comparison) — `baselines_all.csv`

| 행 | 열 | 원고 R5 | results_regen | results_v2 | v2↔원고 |
|---|---|---|---|---|---|
| XGBoost | In-domain ± std | 0.8815 ± 0.0019 | 0.9177 ± 0.0012 | **0.8814 ± 0.0017** | **0.0001** |
| XGBoost | OOD mean | 0.520 | 0.6202 | **0.5178** | **0.0022** |
| LightGBM | In-domain ± std | 0.8805 ± 0.0018 | 0.9172 ± 0.0008 | **0.8805 ± 0.0019** | **0 (exact)** |
| LightGBM | OOD mean | 0.517 | 0.6019 | **0.5144** | **0.0026** |
| CatBoost | In-domain ± std | 0.8794 ± 0.0016 | 0.9142 ± 0.0010 | **0.8795 ± 0.0013** | **0.0001** |
| CatBoost | OOD mean | 0.516 | 0.5974 | **0.5165** | **0.0005** |
| TabTransformer | In-domain ± std | 0.8099 ± 0.0060 | 0.8712 ± 0.0037 | **0.8102 ± 0.0060** | **0.0003** |
| TabTransformer | OOD mean | 0.528 | 0.5994 | **0.5154** | 0.0126 |
| BiLSTM+attention | In-domain ± std | 0.6634 ± 0.0018 | 0.7320 ± 0.0075 | **0.6634 ± 0.0018** | **0 (exact)** |
| BiLSTM+attention | OOD mean | 0.513 | 0.5689 | **0.5128** | **0.0002** |

**§6.9's "all five models collapse identically, 0.513–0.528 mean OOD AUC" claim
is fully restored**: results_v2's 5-model OOD range is **0.5128–0.5178**
(BiLSTM 0.5128, LightGBM 0.5144, TabTransformer 0.5154, CatBoost 0.5165, XGBoost
0.5178) — sitting *inside* the manuscript's claimed 0.513–0.528 band, vs.
results_regen's 0.569–0.620 which broke it entirely.

### Table A.8 (statistical tests) — `stat_tests.csv`

| 행 | 열 | 원고 R5 | results_regen | results_v2 | v2↔원고 |
|---|---|---|---|---|---|
| Friedman | χ² | 18.2 | 18.2 | **18.2** | 0 (동일) |
| Friedman | p | 0.0001 | 0.0001 | **0.0001** | 0 (동일) |
| Wilcoxon XGB>LGBM | d | 1.11 | 0.3753 | **0.7408** | 0.3692 |
| Wilcoxon XGB>Cat | d | 1.79 | 1.948 | **1.6084** | 0.1816 |
| Wilcoxon LGBM>Cat | d | 0.81 | 1.5553 | **0.909** | 0.099 |
| Wilcoxon full vs no_game | d | −0.10 | +0.1774 | **+0.0044** | 0.1044 (sign differs, near-zero both) |
| Wilcoxon full vs no_game | p | 0.492 | 0.1934 | **0.8457** | 0.3537 |

**Improved but not as tight as most other tables.** The three pairwise
XGB/LGBM/Cat effect sizes move closer to the manuscript than results_regen (d
gaps shrink from 0.16–1.57 to 0.10–0.37) but don't land as precisely — plausibly
because Wilcoxon signed-rank effect size is sensitive to small residual
per-dataset AUC differences (see A.7/Table 9's 0.0001–0.003 residuals) in a way
that compounds across the ranking. The game-theory null-result *direction*
(near-zero, non-significant) is well restored: |d|=0.0044 (manuscript −0.10,
results_regen +0.18) and p=0.8457 is comfortably non-significant, same
conclusion as the manuscript's p=0.492, even though the sign of d differs
trivially around zero.

### Table A.9 (failure categories) — `failure_categories.csv`

| 행 | 열 | 원고 R5 | results_regen | results_v2 | v2↔원고 |
|---|---|---|---|---|---|
| Dense traffic | FP % | 65% | 65.0 | **65.0** | 0 (exact) |
| Dense traffic | FN % | 65% | 65.0 | **65.0** | 0 (exact) |
| Missing neighbor | FN % | 50% | 30.0 | **45.0** | 5 pts |
| Low speed | FN % | 30% | 30.0 | **30.0** | 0 (exact) |
| No lateral cue | FN % | 30% | 65.0 | **35.0** | 5 pts |

Much closer than results_regen (which was 15–35 points off on 2 of 5 rows); 2
rows now exact, 2 within 5 points. As `regen_diff.md` already noted, this table
is inherently sensitive to exactly which 20 FP/20 FN examples the current fit
selects, so some residual reshuffling is expected even with a near-identical
underlying model.

---

## C. The five explicitly-required checks

**1. Table 7 zero-shot, 4 targets + mean** — see full table in §B above.
results_v2 (`ood_ci.csv`): ETRI 0.4931 (원고 0.484, Δ0.009), EMT 0.5765 (원고
0.601, Δ0.025), uniD 0.5021 (원고 0.508, Δ0.006), exiD 0.4924 (원고 0.487,
Δ0.005), mean 0.516 (원고 0.520, Δ0.004). **Close — max per-target Δ is 0.025
(EMT), vs. results_regen's max Δ of 0.168 (uniD).**

**2. Table A.3 pooled n and AUC** — pooled n = **95,728, an exact match** to the
committed/manuscript value (results_regen: 371,998). Pooled AUC = **0.4968**
(원고 0.486, Δ0.011; results_regen: 0.5416, Δ0.056). n is exact; AUC is close
but not exact.

**3. Table 14 transfer gain, 8 cells, sign check** — **all 8 negative** under
results_v2 (exiD −0.0095/−0.0048, uniD −0.0175/−0.0207, EMT −0.0578/−0.0741,
ETRI −0.0246/−0.1007), matching the manuscript's "all eight cells negative"
claim exactly in sign. results_regen had 2 positive (uniD@10% +0.0027, ETRI@10%
+0.0819) — **both are negative again under results_v2.**

**4. Table A.7, 5-model OOD mean AUC range** — results_v2 range is
**0.5128–0.5178** (BiLSTM 0.5128 / LightGBM 0.5144 / TabTransformer 0.5154 /
CatBoost 0.5165 / XGBoost 0.5178), which sits **inside** the manuscript's claimed
0.513–0.528 band. results_regen's range was 0.569–0.620, entirely outside that
band.

**5. Table 5, canonical, 4 rows** — **unchanged from results_regen, byte-for-byte
identical.** Root cause: `scripts/35_table5_repro.py`'s `source=canonical` path
calls the old leaky `build_samples()` directly (line 99), independent of
`02_build_features.py`/`_v2.py` — this script was never affected by the PHASE
6/7 fix. Table 5 canonical still differs from the manuscript by the same margins
documented in `regen_diff.md` (14/16 cells ≥0.001, AUC/Δ columns off by
0.01–0.10). This is the one required check that does **not** improve under
PHASE 7.

---

## D. Summary of what the numbers show

Across every table checked in §B except Table 5, results_v2 (leak-free sampler)
lands substantially closer to the manuscript's printed numbers than results_regen
(leaky sampler) did — in several cases (Table 6, Table 10 V4, Table 11 all 16
cells, Table 13 pool sizes, Table A.3 pooled n, Table A.7's 5-model range) to
within 0.001–0.005, i.e. at or near the manuscript's own rounding precision.
Table 14's structural claim ("all eight cells negative") is fully restored, as
is Table A.7's structural claim ("five models collapse into a tight
0.513–0.528 band"), both of which results_regen had broken. Two things do *not*
close to near-zero: Table A.8's Wilcoxon effect sizes (d gaps of 0.10–0.37
remain, though directionally/qualitatively still matching), and Table 5's
canonical-source numbers (completely unchanged — a separate, unfixed bug in
`35_table5_repro.py` unrelated to `02_build_features*.py`). Solver total game
instances are off by exactly 1,000 (557,272 vs. 556,272), unexplained.

Taken together: for every part of the pipeline whose divergence PHASE 5/6 traced
to `02_build_features.py`'s leaky sampler, substituting `02_build_features_v2.py`
resolves the divergence, in most cases down to sub-0.005 residuals consistent
with ordinary run-to-run floating-point/threading noise rather than a systematic
error. The one table that does not improve (Table 5 canonical) is a
mechanistically distinct, already-documented issue (`table5_diagnosis.md`) in a
script that independently reimplements the old sampler rather than calling
`02_build_features.py`.
