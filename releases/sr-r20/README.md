# sr-r20-audit public materials

Scientific Reports R19: Recording identities and feature timing affect the evaluation of lane change prediction across datasets. Hyerim Choi, Tae-Kook Kim, Tae-Wan Kim; corresponding author Tae-Kook Kim. Submission preparation only.

See docs/REPRODUCIBILITY.md, docs/PAPER_MAP_R19.md, DATA_ACCESS.md and DATA_LICENSES.md. Public figures retain old filenames but are mapped to Main1–3 and SupplementaryS1–S5. `python reproduce/verify_public.py --root .` checks the public manifest and aggregate arithmetic. `python reproduce/reproduce_public.py --root . --out /tmp/gavogo-r20-public-output` regenerates seven pairs; choose a new output directory. Fixed SupplementaryS3 requires authorized arrays for regeneration and is explicitly skipped.

Code is MIT; aggregate/result use is not a blanket MIT or new CC grant. Public ZIP excludes the private handoff, manuscript/contact text, raw/converted trajectories, row predictions/splits, per-sample SHAP/features, medians and checkpoints pending redistribution clarification. All decisions are file-specific in PUBLICATION_INVENTORY.csv. Publication does not certify full-training or strict-online reproducibility.
