# Data rights and scope — checked 2026-09-18

MIT in LICENSE_CODE applies to author software, not all data/results in this bundle. No new CC0, CC BY, or commercial-use grant is created for the combined release. Aggregate study results and author plots are provided for scientific inspection with attribution to the study and dataset providers; provider restrictions remain applicable. Contact the rights holders for other reuse. This is not permission to redistribute source datasets.

| Source | Evidence | Publication decision |
|---|---|---|
| highD | https://levelxdata.com/highd-dataset/ (the highD-specific terms block) | No raw or modified trajectory redistribution; abstract non-reconstructive derivatives may be allowed with attribution and non-commercial conditions. Release only study aggregates/plots; keyed rows/SHAP features withheld. |
| uniD | https://levelxdata.com/unid-dataset/ (uniD-specific block) | Same conditional abstract-derivative distinction; no claim that being processed alone makes rows redistributable. |
| exiD | https://levelxdata.com/exid-dataset/ (exiD-specific block) | Same conditional distinction; direction identifiers are still source recording identifiers. |
| NGSIM | https://catalog.data.gov/dataset/next-generation-simulation-ngsim-vehicle-trajectories-and-supporting-data | Current official catalog lists CC BY-SA 3.0, not an assumed public-domain label. No new row-level NGSIM subset is redistributed. Mixed-source predictions/checkpoints need compatibility clarification with highD non-commercial terms. |
| MiTra | https://api.datacite.org/dois/10.25532/OPARA-881 and https://doi.org/10.25532/OPARA-881 | Dataset metadata lists CC BY 4.0; the example-code MIT license is not the dataset license. No original trajectories redistributed. |
| ETRI | https://www.data.go.kr/data/15041797/fileData.do | Portal lists unrestricted use, but explicitly routes the full matching trajectory data through a signed use agreement/provider approval. The applicable local agreement was not supplied; full row/feature redistribution remains pending, not automatically permitted by catalog metadata. |
| EMT | https://github.com/AV-Lab/emt-dataset | Main study excludes EMT. The local conversion failed; data-level rights are not inferred from a code license. No EMT rows/models/SHAP arrays published. |

Two included-study mixed-source checkpoints are withheld pending a determination of derivative status and compatible source conditions; this is not a blanket statement that trained models are forbidden. Split IDs and prediction labels are linked event annotations; their non-reconstruction and applicable redistribution scope were not established. Removing only `features` from five SHAP NPZs would leave per-sample attributions/margins/bias, so no such incomplete sanitization is published. We publish mean-absolute SHAP aggregates instead and preserve the fixed normalized/rasterized author plot; feature-level recalculation needs authorized inputs.

PUBLICATION_INVENTORY.csv gives per-file decisions and PUBLIC_SOURCE_TRANSFORMS.csv identifies public derivatives. Source dataset names and citations are in DATA_ACCESS.md. Restricted row data and private correspondence are not hidden inside Release ZIPs. No provider inquiry was sent.
