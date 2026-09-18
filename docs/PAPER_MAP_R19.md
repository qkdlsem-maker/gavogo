# R19 paper-to-artifact map

All public paths below are relative to `releases/sr-r20/`. Original table indices are zero-based R18 indices supplied with R19. Aggregate inputs live in aggregates/R16 or aggregates/R17. Historical predictions are not silently substituted with public summaries.

| Paper table | R18 index | Public numerical table | Audit/source evidence | Public verification scope |
|---|---:|---|---|---|
| Main1 | 0 | tables/main_01.csv | R16 populations.json | Public aggregate counts; no raw replay |
| Main2 | 2 | tables/main_02.csv | R16 sequence_impact.csv | Archived mixed-sequence counts; synthetic corrected-code tests |
| Main3 | 6 | tables/main_03.csv | R17 A1_seed_metrics.csv/A1_summary.csv | Seed-aggregate mean and population SD replay |
| Main4 | 8 | tables/main_04.csv | R17 A2_seed_gains.csv/bootstrap_ci.csv | Public aggregate replay; original physical exiD, ETRI interval not estimable |
| Main5 | 9 | tables/main_05.csv | R17 B2_censoring.csv (private); public B2_counts.json | Counterexample arithmetic only; private IDs/raw regeneration unavailable publicly |
| S1 | 3 | tables/supp_01.csv | R16 prediction_metrics.csv/summary.json | Means/population SD from per-run aggregates |
| S2 | 7 | tables/supp_02.csv | R17 bootstrap_ci.csv | Archived seed42 CI check; keyed prediction/bootstrap replay unavailable |
| S3 | 12 | tables/supp_03.csv | R17 A2_seed_gains.csv/bootstrap_ci.csv | Literal exiD sensitivity aggregate replay |
| S4 | 4 | tables/supp_04.csv | R16 same_fit_metrics.csv/same_fit_ci.csv/calibration_bins.csv | Earlier fixed-fit aggregate replay, not A1 |
| S5 | 5 | tables/supp_05.csv | R16 adaptation_no_emt.csv/paired_ci.csv | Vehicle-split fit aggregates; no new bootstrap |
| S6 | 1 | tables/supp_06.csv | Archived table5_repro_canonical.csv; public archived_sampling.csv | Archived AUC contrast only; old group-count inconsistency noted |
| S7 | 10 | tables/supp_07.csv | Corrected kinematic.py/game_theory.py | Published source column counts |
| S8 | 11 | tables/supp_08.csv | R16 same_fit_ci.csv | Archived row vs vehicle CI; conditional fixed model |

| Paper figure | Original file stem | Public regeneration | Fit/input |
|---|---|---|---|
| Main1 | fig1_corrections | Yes | R16 three-seed before/after |
| Main2 | fig7_recording_uncertainty | Yes | R17 fixed A1 seed42 physical recording CI |
| Main3 | fig8_recording_gain | Yes | R17 primary A2 paired recording CI |
| S1 | fig2_samefit | Yes | R16 earlier fixed model calibration/CI |
| S2 | fig3_attribution | Yes | R16 mean-absolute SHAP aggregates; native SHAP not recomputed |
| S3 | fig4_shap_distribution | Fixed PNG/PDF only; optional --shap requires authorized arrays | R16 per-sample SHAP/features withheld |
| S4 | fig5_adaptation | Yes | R16 vehicle-split paired fits, population SD |
| S5 | fig6_paired_gain | Yes | R16 paired vehicle CI |

Fixed author figures: figures/<stem>.png and .pdf. reproduce/reproduce_public.py regenerates seven pairs into a new directory. The old eight-file numbering is preserved so provenance hashes remain traceable. Public data checks verify archived CI numbers, not their distribution from hidden sample arrays.
