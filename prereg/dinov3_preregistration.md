# Pre-registration: one-time test of the frozen DINOv3 system on S-test, PP2020 and PlantDoc

Status: written and committed **after the DINOv3 system was frozen and before any S-test, PP2020 or PlantDoc output exists**. No DINOv3 classifier, feature extractor or failure head has been run on any S-test, PP2020 or PlantDoc image, apart from a code smoke test on 4 images per set (seed 42), whose outputs were printed as shapes only and not saved.

Once this file is committed and pushed, its SHA-256 is written into `src/run_inference.py` and `src/evaluate.py`, which refuse to run if it changes.

## 1. Question

Do the findings of the MobileNetV3-Small study hold when the backbone is replaced by a self-supervised vision foundation model (DINOv3 ViT-S/16) and everything else stays the same?

## 2. Frozen system (nothing is retrained, recalibrated or re-thresholded)

- Plan: `notes/plan.md` (commit `cf9362f`, amendment 1 at `3f6f8e4`), written before any check or training.
- Backbone: DINOv3 ViT-S/16, `timm/vit_small_patch16_dinov3.lvd1689m`, revision `3bf4720a82ec2066db88137180ff1f83a675cef0`, weight SHA-256 `2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040`. Feature = CLS token after the final norm (384 numbers).
- Training used the original code imported read-only: classifier seeds 42, 123 and 2026; the 13 failure views per S-reliability image (view manifest byte-identical to the original); heads P1 (387 inputs), A1 (384) and B5 (3); temperature and tau80/tau90 cut-offs on S-calibration.
- Freeze record `results/frozen_system_manifest.json`, 39 files, made at commit `8e7bb90`, SHA-256 `f357f7f29e6489e66c27519e547316ca4d33314d1b9c1e5377f5e53fc5b54577`. `verify_frozen()` runs before inference and before evaluation.
- Preprocessing is the original one: RGB, resize to 224 x 224 without keeping the aspect ratio, ImageNet normalisation, no EXIF rotation, no augmentation.

## 3. Source-side facts known before this pre-registration (disclosure)

These come from source data only. They are stated because they shape what the tests can show.

1. **Classifiers:** validation macro-F1 1.00 for all three seeds (MobileNet: about 0.99).
2. **Few failure examples:** the DINOv3 classifiers fail on only **15, 11 and 11 of 3,978** reliability-training views (from 2, 3 and 3 original images), and on **0 of 1,001** reliability-validation views. MobileNet had 74 to 95 training failures. As fixed in the plan, the views were not changed. The heads were trained on these few positives (positive weight capped at 5), and their source validation quality cannot be measured.
3. **Degenerate temperature:** S-calibration holds 0, 1 and 0 wrong predictions. The original temperature search therefore stops at or near its limit: T = 0.050, 0.114 and 0.050. Many temperature-scaled MSP scores are exactly 0 (ties). The original tie-averaging AURC handles these ties.
4. **Batch-alarm thresholds** (`results/source_alarm_thresholds.csv`, source only, rule of the extended study): 99th percentile over 2,000 random batches of reliability-training original images.

| Seed | S2, batch 20 | S2, batch 50 | S1, batch 20 | S1, batch 50 |
|---|---|---|---|---|
| 42 | 0.295669 | 0.168031 | 0.20 | 0.16 |
| 123 | 0.255906 | 0.155906 | 0.25 | 0.18 |
| 2026 | 0.254331 | 0.168071 | 0.25 | 0.16 |

The hypotheses restate the MobileNet results (original study on PP2020; extended study on PlantDoc, `Disease_Detection_extended`, sealed pre-registration SHA `813e2283…`). H3's regime rule comes from the extended study's exploratory analysis, which was done after its PlantDoc results. It is tested here for the first time on a new system.

## 4. Data (label-free inputs, verified by hash)

| Set | Images | Inference list | Labels joined at evaluation from | Clusters |
|---|---|---|---|---|
| S-test | 382 | `splits/split_assignments.csv` (subset `s_test`) in the original project, SHA `d59a80cc…` | the same split file | `group_id` |
| PP2020 | 1,730 | original `manifests/target_inference_manifest.csv`, SHA `69dc708e…29cac` | original `manifests/manifest.csv` | `cluster_id` |
| PlantDoc | 265 | extended `manifests/plantdoc_inference_manifest.csv`, SHA `66c43f19…e149` | extended `manifests/plantdoc_eligible.csv` | `cluster_id` |

S-test is reported descriptively (no hypothesis), because its few errors make error ranking undefined or unstable.

## 5. Hypotheses (tested separately on PP2020 and on PlantDoc)

Definitions:
- An error is a predicted class that differs from the label.
- For every score, higher means less trustworthy.
- ΔAURC(X) = AURC(X) − AURC(MSP) on the same images, using the original `metrics.py` with tie averaging. It is computed per seed and as the seed mean.
- **Collapse index C** (label-free): the share of images in the most-predicted class, averaged over the three seeds. It is computed from the prediction files that are pushed before labels are joined.

| ID | Type | Hypothesis | Supported if |
|---|---|---|---|
| **H1** | **Primary** | The proposed head P1 does **not** rank errors better than MSP | The 95% interval of mean ΔAURC(P1) is not entirely below 0 |
| H2 | Secondary | The features-only head A1 does not rank errors better than MSP | Same rule for A1 |
| H3 | Secondary (regime) | B5's advantage over MSP depends on class collapse | If C ≥ 0.90: B5 beats MSP (interval entirely below 0). If C < 0.90: B5 does not beat MSP (interval not entirely below 0). |
| H4 | Secondary | The source-fitted temperature does not improve target calibration | Seed-mean NLL after the frozen temperature ≥ seed-mean NLL before |
| H5 | Secondary | The source-only batch alarm S2 fires on the target set | In every seed, at batch sizes 20 and 50, at least 95% of 2,000 random target batches exceed the frozen S2 threshold |
| H6 | Secondary (mechanism) | B5 follows logit magnitude rather than logit gap | The interval of the seed mean of [Spearman(B5, E) − Spearman(B5, −margin)] lies above 0. E = −log Σ exp(logits); margin = top logit − runner-up logit. |

**Interpretation:**
- H1 and H2 are "no improvement" hypotheses, so a wide interval supports them. The point estimates and intervals are therefore always reported, and an interval entirely above 0 is reported as "worse than MSP".
- H4 is expected to hold given fact 3 of section 3. It is kept because it is the same test as in the MobileNet studies.
- H3 on PP2020 and on PlantDoc together tests whether the regime rule predicts B5's direction on a new backbone. H3 is not supported if its prediction fails on either dataset.

## 6. Also reported (descriptive, no hypothesis)

- Classification (accuracy, macro-F1, per-class precision, recall and F1, confusion matrices) and calibration (NLL, ECE, Brier, before and after the frozen temperature) on all three sets.
- All seven methods (B1 to B5, A1, P1) plus E:
  - AURC, ΔAURC with intervals, AUROC-error and AUPRC-error;
  - coverage and selective risk at the frozen tau80 and tau90;
  - selective risk at 50%, 70%, 80% and 90% coverage.
- The S1 feature alarm (share beyond the S-calibration 95th-percentile centre distance).
- **Exploratory, labelled as such:**
  - between-class vs within-class AUROC decomposition;
  - logit magnitude of correct vs wrong predictions;
  - a comparison with the MobileNet numbers.

## 7. Statistics and decision rules

- **Bootstrap:** 2,000 resamples of whole clusters, random seed 2026, percentile 95% intervals. The same resamples are used for all methods and seeds, and the seed mean is resampled jointly.
- **Underpowered rule:** a seed with fewer than 50 errors on a target set is flagged, and that set's H1, H2, H3 and H6 verdicts are labelled underpowered. Undefined AUROC/AUPRC (no errors) is reported as undefined.
- **One run only:**
  - Inference runs once and refuses to run again if outputs exist.
  - The prediction files are hashed, committed and pushed **before** labels are joined.
  - The evaluation runs once. A dry run with random labels (real labels not read) may be used to test the evaluation code first.
- **No changes after results:** no retraining, recalibration, threshold change, dataset change or method change. Every outcome is reported.

## 8. Deviations from the original protocol, declared in advance

- **Backbone and its mapping** (CLS feature; the last 3 transformer blocks trainable in phase B; LayerNorms frozen) are as in the plan.
- **Failure supervision:** the minimum of 50 failing training views is not met (fact 2 of section 3). As the plan requires, this is reported, not fixed.
- **Monitoring metrics:** the heads' validation AUPRC/AUROC are recorded as NaN when the validation views contain no failure.
- **Proposed head P1:**
  - The head-training loop reproduces the original A1 and B5 checkpoints bit-for-bit (`notes/head_recipe_check.md`).
  - It reproduces the original P1 only up to a numerical difference caused by the older library used for that checkpoint.
  - Here, all heads use one loop in one environment.
