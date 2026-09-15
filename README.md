# GAVOGO

A leak-free benchmark for cross-domain lane-change intention prediction.

Official code for **"A Leak-Free Benchmark for Cross-Domain Lane-Change Intention Prediction:
Near-Chance Zero-Shot Transfer Across Seven Naturalistic Trajectory Datasets."**

---

## Why this exists

Lane-change intention predictors routinely report ROC-AUC above 0.90, and several report that the
skill transfers to unseen datasets at modest cost. Both claims depend on an evaluation convention
that has not been audited for leakage. This repository removes three shortcuts from that
convention and re-measures what is left.

| Shortcut | How it arises | Diagnostic | Reading |
| --- | --- | --- | --- |
| Vehicle identity | Lane-keeping negatives drawn from vehicles that never change lanes, so the label is nearly a function of vehicle id | group–label purity | **0.96–0.99** conventional → **0.000** here |
| Split leakage | Row-level splitting puts near-identical consecutive frames of one vehicle on both sides | split-granularity audit | group-disjoint at `(dataset, recording, vehicle)` |
| Road geometry | "Lateral" defined on a fixed global axis, so tracking a curved lane looks like a maneuver | single-feature AUC of lateral speed | exiD **0.928** global → **0.774** road-relative |

After removing them, in-domain AUC is **0.88** and zero-shot transfer across four unseen target
domains averages **0.520** (pooled 0.486, 95% CI [0.482, 0.489]).

## What this repository provides

- **Dataset adapters** that map seven heterogeneous naturalistic driving datasets
  (highD, NGSIM, MiTra, ETRI, EMT, uniD, exiD) onto one canonical schema
- **Within-vehicle negative sampling** and **road-frame canonicalization**
- A **group–label purity diagnostic** that runs on any grouped dataset without training a model
- Every script that produces a table or figure in the paper

It does **not** redistribute raw trajectory data.

---

## Installation

```bash
git clone https://github.com/qkdlsem-maker/gavogo.git
cd gavogo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The core pipeline needs only `requirements.txt`. Four scripts need extras:

```bash
pip install lightgbm catboost shap torch
```

| Extra | Needed by |
| --- | --- |
| `lightgbm`, `catboost` | `19_baselines_all.py`, `11_significance.py`, `18_tiv_remaining.py` |
| `torch` | `19_baselines_all.py`, `04c_tabtransformer.py`, `13_dl_latency.py` |
| `shap` | `18_tiv_remaining.py`, `08_shap.py`, `fig_shap_ieee.py` |

## Getting the data

Obtain each dataset from its original provider and place it under `data/raw/<dataset>/`.
Paths, frame rates and recording ranges are declared in `src/config.py`.

| Dataset | Country | Road type | Source |
| --- | --- | --- | --- |
| highD | Germany | Highway | levelXdata |
| exiD | Germany | Highway + ramp | levelXdata |
| uniD | — | Urban | levelXdata |
| NGSIM | USA | Highway | US FHWA |
| MiTra | Italy | Highway | third-party licence |
| ETRI | Korea | Highway | ETRI |
| EMT | — | Urban | original provider |

## Quick start — run the diagnostic on your own data

```bash
python scripts/diag_purity_all.py
```

Prints group–label purity and in-domain AUC per dataset. Purity near 1.0 means the label is
close to a deterministic function of group identity, and any AUC measured on that sample set is
not a skill estimate.

## Pipeline

```bash
python scripts/01_build_events.py        # raw → canonical records → lane-change events
python scripts/02_build_features.py      # canonical + events → 34 kinematic/spatial/lateral features
python scripts/03_build_game_features.py # + 14 absolute-unit strategic descriptors
python scripts/03b_build_game_di.py      # + 14 dimensionless strategic descriptors, Nash solver stats
python scripts/04_train.py               # joint in-domain training and zero-shot evaluation
```

## Reproducing the paper

| Table / Figure | Script | Output |
| --- | --- | --- |
| Table 4 — datasets, sample counts | `29_dataset_stats.py` | `dataset_stats.csv` |
| Table 5 — vehicle-identity shortcut | `27_conventional_leakage.py`, `diag_purity_all.py` | `identity_leakage_contrast.csv`, `diag_purity_all.csv` |
| Table 6 — in-domain AUC | `04_train.py` | `joint_results.csv` |
| Table 7 — zero-shot OOD | `04_train.py` | `joint_results.csv` |
| Table 8 — OOD by road type | `24_roadtype.py` | `roadtype_zeroshot.csv`, `roadtype_highway_lodo.csv` |
| Table 9 — domain-invariant representations | `16_domain_invariant.py`, `34_coral.py` | `domain_invariant.csv`, `34_coral.csv` |
| Table 10 — adapter ablation | `14_adapter_ablation.py` | `adapter_ablation.csv` |
| Table 11 — strategic descriptors, absolute vs dimensionless | `17_game_di_eval.py` | `game_di_domain_auc.csv` |
| Table 12 — calibration under shift | `18_tiv_remaining.py` | `calibration_ood.csv` |
| Table 13 — adaptation curve | `05_domain_adapt_v2.py` | `adaptation_curve.csv` |
| Table 14 — adaptation control | `20_adapt_control.py` | `adapt_control.csv` |
| Table A.1 — exclusion-margin sweep | `21_margin_sweep.py` | `margin_sweep.csv` |
| Table A.2 — discarded-vehicle audit | `22_discard_audit.py` | `discard_audit.csv` |
| Table A.3 — zero-shot bootstrap CIs | `23_ood_ci.py` | `ood_ci.csv` |
| Table A.4 — 1 s horizon control | `28_horizon1.py` | `horizon1.csv` |
| Table A.5 — seven-way leave-one-dataset-out | `30_lodo7.py` | `lodo7.csv` |
| Table A.6 — simpler model classes | `31_simple_baselines.py` | `simple_baselines.csv` |
| Table A.7 — representation comparison | `19_baselines_all.py` | `baselines_all.csv` |
| Table A.8 — paired statistical tests | `18_tiv_remaining.py` | `stat_tests.csv` |
| Table A.9 — failure categories | `18_tiv_remaining.py` | `failure_categories.csv` |
| Fig. 2 — zero-shot collapse | `fig_ieee.py` | `fig_zeroshot_bar.png` |
| Fig. 3 — strategic-feature leakage | `fig_gamedi_scatter.py` | `fig_gamedi_scatter.png` |
| Fig. 4–6 — SHAP attribution | `fig_shap_ieee.py` | `fig_shap_by_dataset.png`, `fig_shap_global.png` |
| Fig. 7 — reliability diagram | `fig_reliability_standalone.py` | `fig_reliability.png` |
| Fig. 8 — adaptation curve | `05_domain_adapt_v2.py` | `adaptation_curve.png` |
| Fig. 9 — adaptation control | `20_adapt_control.py` | `adapt_control.png` |
| Fig. 10 — runtime scaling | `12_extras.py`, `13_dl_latency.py` | `latency.csv`, `latency_dl.csv` |
| Fig. A.1 — domain-classifier confusion | `33_domain_confusion.py` | `domain_confusion.csv` |

Also available: `07_lodo.py` (leave-one-domain-out), `15_loco.py` (leave-one-country-out and
leave-one-road-type-out), `32_purity_auc.py` (purity against AUC inflation),
`26_leakage_quant.py` (leakage decomposition).

Outputs are written to `results/tables/` and `results/figures/`.

### Seeds

| Experiment | Seeds |
| --- | --- |
| Paired statistical tests | 42, 0, 1, 7, 123, 2024, 5, 11, 77, 999 |
| Representation comparison, adapter ablation, strategic descriptors, adaptation | 42, 0, 1 |
| Correlation alignment (CORAL) | 0, 1, 2 |
| Leave-one-country-out, leave-one-road-type-out | 42, 0, 1, 7, 123 |
| Primary zero-shot reference fit | 42 |

## Repository layout

```
gavogo/
├── src/
│   ├── config.py               paths, frame rates, horizons, seeds, dataset roles
│   ├── features/
│   │   ├── sampling.py         within-vehicle negative sampling (require_both)
│   │   ├── roadframe.py        per-recording PCA road axis and lane centerlines
│   │   ├── kinematic.py        34 kinematic, spatial and lateral features
│   │   ├── game_core.py        absolute-unit Nash–Stackelberg payoffs
│   │   ├── game_core_di.py     dimensionless reformulation (Appendix A.5)
│   │   └── game_theory*.py     14 strategic descriptors
│   └── models/
│       ├── train.py            XGBoost fit, group-disjoint splitting
│       └── baselines.py        LightGBM, CatBoost, TabTransformer, BiLSTM
├── scripts/                    one script per table or figure — see the map above
├── requirements.txt
└── README.md
```

## Using the protocol on your own data

The protocol is not specific to lane changes. It applies wherever samples are grouped by a
subject whose identity correlates with the label — activity recognition, clinical event
prediction, speaker verification.

1. Write an adapter exposing two operations: load raw records, and emit canonical records.
   A canonical record carries identifiers `(dataset, recording, vehicle, frame)`, longitudinal
   and lateral position and velocity, acceleration, a lane identifier, and six neighbour slots.
2. Recover a road-local frame per recording if lateral motion matters in your setting
   (`src/features/roadframe.py`).
3. Sample negatives within each group, requiring every group that contributes a positive to
   contribute a negative (`src/features/sampling.py`, `require_both=True`).
4. Check purity before training. If it is not 0, the sample set still admits an identity
   shortcut.
5. Split at the group level, never at the row level.

## Citation

```bibtex
@article{choi2026gavogo,
  title   = {A Leak-Free Benchmark for Cross-Domain Lane-Change Intention Prediction:
             Near-Chance Zero-Shot Transfer Across Seven Naturalistic Trajectory Datasets},
  author  = {Choi, Hyerim and Kim, Tae-Kook and Kim, Tae-Wan},
  journal = {Engineering Applications of Artificial Intelligence},
  year    = {2026},
  note    = {Under review}
}
```

<!-- 채택 후 volume, pages, DOI 를 채우고 note 줄을 지우세요. Zenodo DOI 를 받으면 여기에 함께 적으세요. -->

## License

[MIT](LICENSE)
