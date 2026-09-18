# Provenance and execution identity

Public source version: sr-r20-audit. The public base was 971dc6c543f1ab4e7a0f1a78f7a782065af002d0. No private Git objects/history were merged. Current research checkout HEAD was 4426ce85a24e66e58cc32ae10bb4366ad7d9558c (the request contains a differently spelled hash; the actual Git value is authoritative).

The research commit is not the R17 driver execution identity. research/r17/configs/code_freeze_A_v1.json identifies the separately frozen A runner and source snapshot, dated 2026-09-18 before training. PUBLIC_SOURCE_TRANSFORMS.csv records incoming archive and public-file hashes. The R19 handoff is a desktop verification/repackaging of R16/R17 evidence; its verified tables and server output can have float round-trip differences. Publication helpers added here did not produce the historical fits.

R16/src/data contains 11 restored original modules (including seven adapters, base, schema, events, __init__) plus one new pixel annotation reader. EMT remains a failed local conversion case and is not restored to main analysis. Author source remains MIT with its original notice. Source snapshots and exact drivers are retained for authorized data users; public quick-start validates aggregate arithmetic and plotting, not raw-data training.

The historical repository root src/scripts/results/run_all.sh remains unchanged for provenance and is not the current reproducibility entrypoint. Some historical scripts retain outdated terminology. Follow releases/sr-r20 and the limitations here. No new training, B3, or 60000-replicate bootstrap run was performed for this release.

R19 replaced GAVOGO with TRAJECTORY_AUDIT in sampling.py and schema.py module docstrings (two modules in each source copy), breaking byte identity with the historical A freeze. R20 restores only those original project labels from the unchanged server frozen source. PUBLIC_SOURCE_TRANSFORMS records incoming and public hashes; all historical A code hashes then match. Algorithm logic was not changed.
