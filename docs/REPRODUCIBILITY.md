# Reproducibility of sr-r20-audit

Use the versioned `releases/sr-r20/` subtree. The legacy root runner is not the current pipeline. Python3.11.7 was used for publication checks; exact tested packages are in environment/public_requirements.txt. Research fit versions and the distinct desktop replay versions are recorded separately. No package was upgraded and bit-identical fitting across environments is not promised.

## Public-only verification (executed in a clean directory)

From a fresh clone:
```bash
python releases/sr-r20/reproduce/verify_public.py --root releases/sr-r20
python releases/sr-r20/reproduce/reproduce_public.py --root releases/sr-r20 --out /tmp/gavogo-r20-replay
```
Choose a new output directory. To install missing packages, use your own isolated environment with `pip install -r releases/sr-r20/environment/public_requirements.txt`; this command is documentation, not an upgrade performed on the research environment.

The Release public ZIP contains the same subtree at its root. After safe extraction into an empty directory:
```bash
python reproduce/verify_public.py --root .
python reproduce/reproduce_public.py --root . --out /tmp/gavogo-r20-zip-replay
```
It checks file hashes, all13 tables' indicated arithmetic/source links, and writes seven figure pairs. Figure S3's fixed PNG/PDF is included but skipped explicitly during aggregate regeneration. Its optional authorized pathway is:
```bash
python reproduce/plot_vehicle_public.py --tables aggregates/R16 --shap /authorized/source_heldout.npz --out /tmp/gavogo-r20-authorized-plots
```
That optional pathway was not executed during this release. It requires values/features/feature_names from the same earlier fit. Do not claim it is reproducible solely from the public release.

## Corrected-code functional tests

```bash
PYTHONPATH=releases/sr-r20/research/r17/code/r16_snapshot python -m unittest discover -s releases/sr-r20/research/r17/code/r16_snapshot/tests -v
PYTHONPATH=releases/sr-r20/research/r17/code python releases/sr-r20/research/r17/code/test_R17_v2.py
```
These use synthetic fixtures and no private inputs. They check dataset/recording sequence boundaries, neighbors, metric endpoints, weighted AUC ties, physical recording grouping, and undefined replicate handling.

## Authorized research inputs (not public-only reproducibility)

Run `python releases/sr-r20/reproduce/check_inputs.py --data-root /path/to/authorized/data` for an explicit missing-input check. No models are trained by quick-start. DATA_ACCESS.md specifies data placement and provider access. The historical R17 driver chain is run_A_v1.py→run_B_v1.py --freeze→run_B_v1.py→run_C_v1.py --freeze→run_C_v1.py, in a new disposable run location with its inputs and execution freeze recorded first. The archived A freeze documents the historical code; it is not certification of a new run. Original raw→canonical/events→all features execution has not been repeated or certified here.

| Level | Executed now | Publicly reproducible |
|---|---|---|
| Input handoff hashes and1305-entry original manifest | Yes, privately | Only public derivatives' new manifest |
| Public13-table arithmetic/summary checks | Yes | Yes, with per-table limitations |
| Seven aggregate figure regenerations | Yes | Yes; rendering may vary by environment |
| Fixed S3 picture integrity | Yes | Yes; not pointwise SHAP regeneration |
| R16/R17 prediction metrics | Privately recomputed, no training | No: keyed records not distributed |
| Prior60000 bootstrap and native SHAP replay | Earlier audit evidence only | No; not repeated now |
| Raw B2 temporal censoring | Earlier server audit only | No; public counterexample/count arithmetic only |
| Full training/B3 | No | B3 remains unexecuted; licensed inputs required for training |

See PAPER_MAP_R19.md and KNOWN_LIMITATIONS.md. Public Release assets are deliberately different from private R19 Supplementary Data1/2. The submission's data package and availability wording must be revised accordingly.
