# GAVOGO — Scientific Reports R19 reproducibility release

**Recording identities and feature timing affect the evaluation of lane change prediction across datasets**
Hyerim Choi, Tae-Kook Kim, Tae-Wan Kim. Corresponding author: Tae-Kook Kim.

Status: Scientific Reports submission preparation; not a claim of submission, peer review, or acceptance. Version **sr-r20-audit**, 2026-09-18.

[Versioned Release](https://github.com/qkdlsem-maker/gavogo/releases/tag/sr-r20-audit) · [Current public materials](releases/sr-r20/) · [Reproducibility](docs/REPRODUCIBILITY.md) · [R19 table/figure map](docs/PAPER_MAP_R19.md) · [Data access](releases/sr-r20/DATA_ACCESS.md) · [Reuse conditions](releases/sr-r20/DATA_LICENSES.md)

Main analysis uses highD/NGSIM/MiTra as source and ETRI/uniD/exiD as targets. EMT is excluded because the local conversion's physical interpretation remains unresolved. The study audits recording identities, feature timing, and matched continuation controls; it does not certify a universally leak-free or strict online benchmark.

Recording-aware recurrent source-test AUC changes **0.6631→0.7760**; target macro **0.5277→0.5312**. Recording-held-out trees give three-seed mean source AUC **0.8700**, target macro **0.4914**, and pooled target mean **0.4765**. The distinct fixed seed42 pooled estimate is **0.4975**, physical-recording interval **[0.4854,0.5131]**. All15 primary adaptation mean gains favor target-only, but ETRI's interval is not estimable. B3 was not executed. [Full limitations](docs/KNOWN_LIMITATIONS.md).

## Public quick-start (no training or raw data)

```bash
python releases/sr-r20/reproduce/verify_public.py --root releases/sr-r20
python releases/sr-r20/reproduce/reproduce_public.py --root releases/sr-r20 --out /tmp/gavogo-r20-replay
```

Use a new output directory. Python3.11 and exact tested packages are listed in [public requirements](releases/sr-r20/environment/public_requirements.txt); install in a separate environment if needed. The commands check numerical aggregates and regenerate seven figure pairs. Supplementary Fig.S3 is a fixed inspectable figure; regenerating its sample-level SHAP display requires authorized arrays not distributed here. Public aggregate replay is **not** prediction-level or full-training replay.

Code, study aggregates, all13 numerical tables, fixed plots and provenance are selected explicitly. Raw/converted trajectories, keyed predictions/splits, feature matrices, per-sample SHAP, medians and mixed-source checkpoints are withheld where redistribution scope is unresolved. See [file inventory](releases/sr-r20/PUBLICATION_INVENTORY.csv). The private R19 handoff is not a Release asset. MIT covers author code, not every dataset or derived artifact.

Historical root `src/`, `scripts/`, `results/`, and `run_all.sh` remain for provenance and may reflect superseded experiments. **They are not the current quick-start.** Updated audited source/drivers are under `releases/sr-r20/research/`. No new training was performed to prepare this release. GitHub publication does not establish a long-term archive DOI; that remains a separate step.
