# Data access clarification for R23

Updated 18 September 2026. This note clarifies data access without replacing the frozen `sr-r20-audit` release or certifying a new training run. The release still supports aggregate checks for 13 tables and regeneration of seven figures; fixed Supplementary Fig. S3 requires additional arrays for regeneration.

## ETRI files used in the study

The five study inputs are available through the download button on [Korean public-data-portal item 15041797](https://www.data.go.kr/data/15041797/fileData.do), dated 20191204. A fresh anonymous download produced a 1,374,350-byte ZIP with SHA-256 `4b43b4bef5ce0228fd60295ca86898935415c99440811d26248a3f8ef8e2fa83`. All five member sizes and SHA-256 hashes match the author-supplied folder, R16 protected-file inventory, R17 protected-file inventory and R17 B2 raw-file record. The B2 record was made after execution; it is not a pre-run freeze.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `180906_s2d_l3_00_crop_pp_lane_crt_lccrt.csv` | 858466 | `c04ee88e75aeb4c7cc45b7666f4dabb13d1a189480efb405e25447cc8b36255e` |
| `180906_s2d_l3_01_crop_pp_lane_crt_lccrt.csv` | 809704 | `41f888c25770891c08180b4be3135ed8ef6a7ba82040de4978250335fbe7910d` |
| `180906_s2d_l3_02_crop_pp_lane_crt_lccrt.csv` | 747129 | `c815c3426cc53115d92ec2146fe29bb04c1e58ede6afba5cafbc6623c955add3` |
| `180906_s2d_l3_03_crop_pp_lane_crt_lccrt.csv` | 1042120 | `1c5a90f7cc501d798053a541a64c8492736b45baaf68586ac53aaafe74f7b613` |
| `180906_s2d_l3_04_crop_pp_lane_crt_lccrt.csv` | 976133 | `1f97d33e6def718266463f33712dad80314e34af3e0a7c3e8bc76c4f3ad83239` |

The CSVs use cp949 encoding and eight columns for frame, vehicle, latitude, longitude, East/North coordinates, lane and lane-change annotation. Cite the 2019 portal item above, not the distinct 2023 item 15125115.

## Portal distribution and provider archive

The portal states that its entry has no restriction on use. Its description separately directs users to an agreement-based full-data distribution. The [current ETRI description](https://epretx.etri.re.kr/dataDetail?lang=ko&id=22) and [provider file listing](https://epretx.etri.re.kr/dataFileList?lang=ko&id=22) identify an 8.1 MB `Trajectory.zip` requiring an agreement. That separate archive was not downloaded or compared. Do not infer that a provider agreement is required simply to obtain the five directly downloadable portal files, and do not extend the portal conditions automatically to another dataset or a mixed-source model.

Readers can obtain the five exact ETRI inputs from the portal; this documentation revision does not mirror raw files. The independent portal-to-input hash comparison resolves their identity and access route. It does not demonstrate that all derived records or models may be distributed under MIT.

## Other inputs and reproducibility scope

For other datasets and historical input placement, consult [the archived access instructions](../releases/sr-r20/DATA_ACCESS.md) and [provider-specific reuse conditions](../releases/sr-r20/DATA_LICENSES.md), reading their ETRI agreement wording with the distinction above. Raw and transformed trajectories, keyed predictions/splits, sample-level SHAP/features, imputation files and checkpoints remain outside the versioned public archive. The newly established ETRI access route does not supply the other inputs needed to repeat all sample-level analyses or fitting. See [reproducibility instructions](REPRODUCIBILITY.md).

No numerical result, archived source, Release asset or tag was changed by this documentation clarification. No full-data agreement was accepted and no new experiment was run.
