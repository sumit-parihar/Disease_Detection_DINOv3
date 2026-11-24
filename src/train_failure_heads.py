"""
train_failure_heads.py  -  Step 5b of the plan: the three failure heads per classifier seed.

  P1 proposed       features + logits (384 + 3 = 387 inputs)
  A1 features_only  features (384)
  B5 logits_only    logits (3)

Same recipe as the frozen retrain_failure_head.py / train_ablation_heads.py, using their own pieces
(SmallHead, set_random_seeds, FAILURE_L2, FAILURE_POS_WEIGHT_CAP, LEARNING_RATE, BATCH_SIZE, FIXED_EPOCHS,
imported read-only): inputs standardised with reliability-train mean/std; Dense(64, ReLU) -> dropout 0.30
-> 1; weighted BCE with positive weight min(5, neg/pos); L2 1e-4 on the hidden kernel; Adam lr 5e-4;
batch 64; fixed 30 epochs, last epoch kept; validation numbers recorded for monitoring only.

--check-original retrains one original head (MobileNet features from the frozen project, in memory only)
and compares it with the frozen checkpoint, to show that this loop reproduces the original recipe.

OUTPUT
  models/failure_head_<variant>_seed<S>.pt      (not tracked)
  results/failure_head_<variant>_history_seed<S>.csv
  results/failure_head_summary.txt
"""

import argparse
import sys

import numpy as np
import pandas as pd

from common import FROZEN_ROOT, MODELS_DIR, RESULTS_DIR, SEEDS, use_frozen_code

use_frozen_code()

import torch   # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score   # noqa: E402
from torch import nn   # noqa: E402
from torch.utils.data import DataLoader, TensorDataset   # noqa: E402

from config import FAILURE_L2, FAILURE_POS_WEIGHT_CAP   # noqa: E402
from failure_head_diagnostic import SmallHead, untrained_scores   # noqa: E402
from retrain_failure_head import FIXED_EPOCHS   # noqa: E402
from train_failure_head import BATCH_SIZE, LEARNING_RATE, set_random_seeds   # noqa: E402

VARIANTS = ("proposed", "features_only", "logits_only")
SUMMARY_FILE = RESULTS_DIR / "failure_head_summary.txt"


def head_matrix(variant, arrays):
    if variant == "proposed":
        return np.concatenate([arrays["features"], arrays["logits"]], axis=1).astype(np.float32)
    return (arrays["logits"] if variant == "logits_only" else arrays["features"]).astype(np.float32)


def train_head(variant, arrays, seed, device):
    train_mask = arrays["reliability_split"] == "reliability_train"
    val_mask = ~train_mask
    matrix = head_matrix(variant, arrays)
    y_all = arrays["failure_label"].astype(np.float32)
    x_train, x_val, y_train, y_val = matrix[train_mask], matrix[val_mask], y_all[train_mask], y_all[val_mask]

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    x_train_scaled = ((x_train - mean) / std).astype(np.float32)
    x_val_scaled = ((x_val - mean) / std).astype(np.float32)

    positives = int(y_train.sum())
    positive_weight = min(float(FAILURE_POS_WEIGHT_CAP), (len(y_train) - positives) / positives)

    set_random_seeds(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train_scaled), torch.from_numpy(y_train)),
                        batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=0)
    model = SmallHead(x_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    x_val_tensor = torch.from_numpy(x_val_scaled).to(device)
    y_val_tensor = torch.from_numpy(y_val).to(device)
    y_val_int = y_val.astype(np.int64)

    history = []
    for epoch in range(1, FIXED_EPOCHS + 1):
        model.train()
        total, seen = 0.0, 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y) + FAILURE_L2 * model.hidden.weight.square().sum()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().item()) * len(batch_x)
            seen += len(batch_x)
        model.eval()
        with torch.inference_mode():
            val_logits = model(x_val_tensor)
            val_loss = float(criterion(val_logits, y_val_tensor).item())
            val_scores = torch.sigmoid(val_logits).cpu().numpy()
        history.append({"epoch": epoch, "train_loss": total / seen, "validation_loss": val_loss,
                        "validation_auprc_error": float(average_precision_score(y_val_int, val_scores)),
                        "validation_auroc_error": float(roc_auc_score(y_val_int, val_scores))})
    checkpoint = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_mean": torch.from_numpy(mean), "input_std": torch.from_numpy(std),
        "inputs": variant, "classifier_seed": seed, "epochs": FIXED_EPOCHS, "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE, "positive_weight": positive_weight,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "recipe": "inputs standardised with reliability-train statistics; fixed 30 epochs; last epoch kept",
        "torch_version": str(torch.__version__),
    }
    return checkpoint, history, val_scores, y_val_int, positives


def load_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in ("features", "logits", "failure_label", "reliability_split")}


def check_original(device):
    """Retrain the frozen project's heads for seed 42 in memory and compare with its checkpoints."""
    arrays = load_arrays(FROZEN_ROOT / "results" / "reliability_v2_features_seed42.npz")
    frozen = {"proposed": "failure_head_scaled_seed42.pt", "features_only": "failure_head_scaled_features_only_seed42.pt",
              "logits_only": "failure_head_scaled_logits_only_seed42.pt"}
    worst = 0.0
    for variant, name in frozen.items():
        mine = train_head(variant, arrays, 42, device)[0]["state_dict"]
        theirs = torch.load(FROZEN_ROOT / "models" / name, map_location="cpu", weights_only=True)["state_dict"]
        diff = max(float((mine[k] - theirs[k]).abs().max()) for k in theirs)
        worst = max(worst, diff)
        print(f"  {variant:<14} max |difference| from frozen checkpoint: {diff:.3g}")
    return worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-original", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda")
    if args.check_original:
        print("Reproducing the frozen seed-42 heads with this loop:")
        check_original(device)
        return

    outputs = [SUMMARY_FILE] + [MODELS_DIR / f"failure_head_{v}_seed{s}.pt" for v in VARIANTS for s in SEEDS]
    if any(p.exists() for p in outputs):
        sys.exit("Outputs already exist; this step runs once.")
    lines = ["DINOv3 FAILURE HEADS (original recipe; validation numbers are for monitoring only)", "=" * 74, ""]
    for seed in SEEDS:
        arrays = load_arrays(RESULTS_DIR / f"reliability_features_seed{seed}.npz")
        val_mask = arrays["reliability_split"] == "reliability_val"
        msp = untrained_scores(arrays["logits"][val_mask])["MSP"]
        y_val = arrays["failure_label"][val_mask].astype(np.int64)
        lines.append(f"Classifier seed {seed}: validation failures {int(y_val.sum())}/{len(y_val)}")
        lines.append(f"  {'MSP (no training)':<22} AUPRC {average_precision_score(y_val, msp):.3f}  "
                     f"AUROC {roc_auc_score(y_val, msp):.3f}")
        for variant in VARIANTS:
            checkpoint, history, scores, _, positives = train_head(variant, arrays, seed, device)
            torch.save(checkpoint, MODELS_DIR / f"failure_head_{variant}_seed{seed}.pt")
            pd.DataFrame(history).to_csv(RESULTS_DIR / f"failure_head_{variant}_history_seed{seed}.csv",
                                         index=False, lineterminator="\n")
            lines.append(f"  {variant:<22} AUPRC {history[-1]['validation_auprc_error']:.3f}  "
                         f"AUROC {history[-1]['validation_auroc_error']:.3f}  "
                         f"(inputs {checkpoint['input_mean'].numel()}, params {checkpoint['parameter_count']:,}, "
                         f"train failures {positives}, pos weight {checkpoint['positive_weight']:.3g})")
        lines.append("")
    SUMMARY_FILE.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
