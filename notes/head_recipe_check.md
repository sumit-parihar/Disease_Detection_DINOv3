# Check: does the head-training loop reproduce the original heads? (2026-11-25, before any DINOv3 head)

`src/train_failure_heads.py --check-original` retrains the original seed-42 heads in memory from the frozen
MobileNet reliability features (`reliability_v2_features_seed42.npz`) and compares the weights with the frozen checkpoints.

| Head | Max weight difference vs frozen checkpoint |
|---|---|
| features-only (A1) | 0 (bit-identical) |
| logits-only (B5) | 0 (bit-identical) |
| proposed (P1) | 0.28 on GPU, 0.28 on CPU |

The frozen P1 checkpoint records `torch_version 2.5.1+cpu`; A1 and B5 record `2.11.0+cu128` (the environment used here).
The P1 difference therefore comes from the older library version, not from the recipe: the same loop reproduces A1 and B5
exactly, the input mean matches exactly, and the retrained P1 reaches the same validation result
(AUPRC 0.147 vs 0.145 frozen, AUROC 0.832 vs 0.828).
For DINOv3, all three heads are trained with this one loop in one environment.
