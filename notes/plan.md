# Backbone replication plan: DINOv3 (fixed before any check or training)

Written 2026-11-25, before model access was checked, before any fit test, and before any training. Every choice below copies the original MobileNetV3-Small recipe. Where the architectures differ, the mapping rule is stated here and is not changed later.

## 1. Question

Do the original findings hold for a self-supervised vision foundation model?
- Feature-based failure heads do not beat MSP.
- Source calibration and cut-offs do not transfer.
- The logits-only advantage appears only under class collapse.
- A source-only batch alarm fires on field data.

## 2. Backbone and fallback rule

- **Primary:** DINOv3 ViT-S/16, Hugging Face `facebook/dinov3-vits16-pretrain-lvd1689m` (gated, manual approval, custom licence).
- **Fallback:** DINOv2 ViT-S/14, `facebook/dinov2-small` (Apache 2.0).
- **The fallback is used if any of the following happens; the choice never depends on accuracy or reliability results:**
  1. access to DINOv3 is not granted by **2026-12-02**;
  2. the DINOv3 licence does not permit academic publication;
  3. the weights fail to load;
  4. the 4 GB fit test of section 4 fails even with the memory rules.
- The backbone actually used is recorded with its exact model revision (commit hash) and the SHA-256 of its weight file.

## 3. Recipe mapping (identical to the original unless stated)

| Item | Original (MobileNetV3-Small) | This study |
|---|---|---|
| Data, splits | PlantVillage apple 2,550 images; split file SHA `d59a80cc…`; reliability 80/20 partition | **Same files, read-only from the original project, hash-verified** |
| Input | 224 x 224 RGB, ImageNet mean/std, resize without keeping aspect ratio | Same (224 is a multiple of 16; for the DINOv2 fallback, 224 = 16 x 14) |
| Feature | 576-d global average pool | **384-d CLS token after the final layer norm** (`pooler_output`) |
| Disease head | Dropout 0.20 → Linear(576, 3) | Dropout 0.20 → Linear(384, 3) |
| Class weights | w_c = N/(C·n_c) from S-train | Same |
| Augmentation (S-train only) | flip 0.5, rotation ±15°, zoom ±10%, contrast ±10% | Same |
| Phase A | backbone frozen; Adam lr 1e-3; batch 32; max 12 epochs; patience 3; select by val loss | Same |
| Phase B | unfreeze last 3 inverted-residual blocks + final 1x1 conv; BatchNorm frozen; AdamW lr 3e-5, wd 1e-5; max 20; patience 5; select on strict val macro-F1 improvement | **Unfreeze the last 3 transformer blocks; all normalisation layers (LayerNorm, including the final norm) stay frozen**; same optimiser, learning rate, weight decay, epochs, patience and selection |
| Seeds | 42, 123, 2026 | Same |
| Failure views | v2: original + 3 each of brightness 0.7–1.3, contrast 0.7–1.3, blur σ 0.5–1.5, noise σ 0.01–0.05 (13 per image) | Same view definitions and per-view seeds (same `stable_seed` rule) |
| Minimum failures | ≥ 50 positive reliability-training views | Checked and reported. **The views and severity are not changed.** If fewer than 50, the heads are still trained and the shortfall is reported as a limitation. |
| Failure heads | P1 (579 in), A1 (576), B5 (3); Dense(64, ReLU, L2 1e-4), dropout 0.30; inputs standardised with reliability-training statistics; Adam lr 5e-4; batch 64; fixed 30 epochs, last kept; positive weight capped at 5 | Same, with P1 = 387 inputs and A1 = 384 |
| Calibration | one temperature per seed on S-calibration (NLL, search 0.05–20); tau80/tau90 = k-th smallest score, k = ceil(c·n) | Same |
| Freeze | hash all models, heads, settings, split, code | Same, in this folder |
| Evaluations | S-test once; PP2020 once | **S-test once; PP2020 once; PlantDoc once.** The existing label-free manifests are reused, verified by hash. |

## 4. Memory rules for the 4 GB GPU (fixed now)

1. Try full precision, batch 32.
2. If that runs out of memory, use automatic mixed precision (bfloat16 or float16 autocast) at batch 32.
3. If that still fails, use gradient accumulation of 2 x 16, giving the same effective batch of 32.
4. If all three fail, the backbone counts as failing the fit test (section 2).

Whichever rule is used is decided by the fit test before training and is recorded.

## 5. Environment

- Separate virtual environment in `venv/` (not tracked).
- Same versions as the frozen study for torch 2.11.0+cu128, torchvision 0.26.0, numpy, pandas, scikit-learn, scipy, Pillow and ImageHash; plus `transformers` and `huggingface_hub` to load the backbone.
- The original project and the extended project are only read.

## 6. Order of work

1. Access request, then licence check.
2. Load test and 4 GB fit test.
3. Commit this plan, the environment record and the check results.
4. Train classifiers (3 seeds).
5. Generate failure views and train the heads.
6. Fit temperature and cut-offs.
7. Freeze and hash.
8. **Pre-register** the hypotheses for S-test, PP2020 and PlantDoc.
9. Run the one-time evaluations, with predictions pushed before labels are joined.
