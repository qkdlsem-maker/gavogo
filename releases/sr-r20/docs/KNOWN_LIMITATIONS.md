# Known limitations

- Six datasets in main results; EMT is a local conversion failure case. No fps substitution repairs it.
- Retrospective, event-conditioned samples and offline full-record PCA/centerline/endpoint/event-sign preprocessing remain. Group-disjoint sampling is not a guarantee of strict online causality or absence of identity cues; purity=0 does not establish either.
- Recording-aware recurrent AUC .6631→.7760 source, .5277→.5312 target macro. Earlier tree/recurrent/SHAP/calibration fits are distinct from A1 recording fits.
- A1 mean source AUC .8700, target macro .4914, pooled mean .4765. Fixed seed42 pooled .4975, physical-recording CI [.4854,.5131]. Seed SD in paper uses ddof0; sample SD columns are labeled separately.
- All15 primary A2 mean gains are negative, but ETRI's two union test recordings give1003valid/997invalid and a degenerate conditional interval: not estimable for inference. uniD7 groups1866/134; physical exiD59 groups2000/0. Intervals condition on fixed models and valid draws, exclude training uncertainty and are unadjusted across budgets.
- Literal exiD r/r+1000 IDs split the same recording. Its full-budget gain+.0041, CI[-.0007,+.0083], is retained as sensitivity, not physical holdout. Different test populations prevent a pure leakage-effect interpretation.
- B2:234 pairs,28 both missing,206 finite,203 equal,3 changed at one ETRI time. B3 not executable under available provenance; no score is assigned. Provider upstream smoothing is not certified.
- Aggregate replay cannot recompute individual prediction AUC, bootstrap, SHAP, model fitting, or raw trajectories. These remain conditional on authorized unavailable inputs. Seven figures regenerate from aggregates; fixed S3 is inspectable, not regenerable without sample arrays.
- S6 matches archived aggregate values only. The old MiTra file reports a mixed-subset vehicle count larger than its total-test count (3945 vs3374); those counts are not in S6 and population reconstruction is not independently certified. Do not treat S6 as a causal identity-shortcut experiment.
- Public release differs from the private R19 Supplementary Data ZIPs. Revise the journal's supplementary-data package and availability wording before submission; do not upload the unreviewed originals.
- Scientific Reports submission preparation only; no claim of submission, review or acceptance. GitHub release is not a DOI archive. Zenodo account/link/deposition not authorized here; no DOI invented.
