"""
calibrate.py  -  Step 6 of the plan: S-calibration outputs, temperature and cut-offs.

Part 1 (as the frozen extract_calibration_outputs.py): run S-calibration through each frozen DINOv3
classifier and its three failure heads with the ORIGINAL frozen_pipeline.run_images, and write the same
columns. Part 2 (as the frozen calibrate_and_thresholds.py): run its own process_seed and write_summary
unchanged (golden-section temperature search on 1/T in [0.05, 20], 500 resamples, seed 2026;
tau = k-th smallest score, k = ceil(c * n)), only pointed at this folder's results/.

Opens S-calibration images only.

OUTPUT (results/)
  s_calibration_outputs_seed<S>.csv, s_calibration_features_seed<S>.npz (not tracked)
  temperature_scaling.csv, calibration_thresholds.csv, frozen_calibration_settings.json, calibration_summary.txt
"""

import json
import sys

import numpy as np
import pandas as pd

from common import FROZEN_ROOT, MODELS_DIR, RESULTS_DIR, SEEDS, DinoClassifier, use_frozen_code, verify_split

use_frozen_code()

import torch   # noqa: E402

import calibrate_and_thresholds as cat   # noqa: E402  (frozen project, read-only)
from config import CLASS_NAMES, EVAL_BATCH_SIZE   # noqa: E402
from dataset import SourceDataset, make_loader   # noqa: E402
from failure_head_diagnostic import SmallHead   # noqa: E402
from frozen_pipeline import run_images   # noqa: E402

SUBSET = "s_calibration"
HEAD_VARIANTS = ("proposed", "logits_only", "features_only")


def load_classifier(seed, device):
    path = MODELS_DIR / f"dino_seed{seed}_phaseB.pt"
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("seed") != seed or ck.get("phase") != "B" or ck.get("class_names") != CLASS_NAMES:
        sys.exit(f"{path.name}: metadata mismatch")
    model = DinoClassifier()
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model.to(device)


def load_head(variant, seed, device):
    ck = torch.load(MODELS_DIR / f"failure_head_{variant}_seed{seed}.pt", map_location="cpu", weights_only=True)
    if ck.get("classifier_seed") != seed or ck.get("inputs") != variant:
        sys.exit(f"Head {variant} seed {seed}: metadata mismatch")
    mean, std = ck["input_mean"].float(), ck["input_std"].float()
    head = SmallHead(int(mean.numel()))
    head.load_state_dict(ck["state_dict"], strict=True)
    head.eval()
    head.requires_grad_(False)
    return head.to(device), mean.to(device), std.to(device)


def extract(seed, device):
    classifier = load_classifier(seed, device)
    heads = {v: load_head(v, seed, device) for v in HEAD_VARIANTS}
    loader = make_loader(SourceDataset(SUBSET), EVAL_BATCH_SIZE, shuffle=False)
    out = run_images(loader, classifier, heads, device)
    split = pd.read_csv(FROZEN_ROOT / "splits" / "split_assignments.csv")
    if list(out["paths"]) != split[split["subset"] == SUBSET].sort_values("path")["path"].tolist():
        sys.exit("Processed images do not match the S-calibration list.")
    probs = cat.softmax(out["logits"].astype(np.float64))
    ordered = np.sort(probs, axis=1)
    predicted = probs.argmax(axis=1)
    table = pd.DataFrame({
        "path": out["paths"], "label": [CLASS_NAMES[i] for i in out["label_index"]],
        "label_index": out["label_index"], "predicted_index": predicted,
        "correct": (predicted == out["label_index"]).astype(int),
        "logit_healthy": out["logits"][:, 0], "logit_scab": out["logits"][:, 1], "logit_rust": out["logits"][:, 2],
        "msp_score": 1.0 - ordered[:, -1],
        "entropy_score": -(probs * np.log(probs + 1e-12)).sum(axis=1),
        "margin_score": 1.0 - (ordered[:, -1] - ordered[:, -2]),
        "head_proposed": out["head_scores"]["proposed"],
        "head_logits_only": out["head_scores"]["logits_only"],
        "head_features_only": out["head_scores"]["features_only"],
    })
    table.to_csv(RESULTS_DIR / f"s_calibration_outputs_seed{seed}.csv", index=False, lineterminator="\n")
    np.savez(RESULTS_DIR / f"s_calibration_features_seed{seed}.npz", features=out["features"],
             logits=out["logits"], label_index=out["label_index"], paths=out["paths"])
    return table


def main():
    settings_file = RESULTS_DIR / "frozen_calibration_settings.json"
    if settings_file.exists():
        sys.exit("Calibration already done; this step runs once.")
    verify_split()
    device = torch.device("cuda")
    for seed in SEEDS:
        table = extract(seed, device)
        print(f"Seed {seed}: S-calibration {len(table)} images, wrong {int((table['correct'] == 0).sum())}")

    # the frozen calibration code, pointed at this folder
    cat.RESULTS_DIR = RESULTS_DIR
    cat.SUMMARY_FILE = RESULTS_DIR / "calibration_summary.txt"
    rng = np.random.default_rng(cat.BOOTSTRAP_SEED)
    temp_rows, threshold_rows, settings = [], [], {}
    for seed in SEEDS:
        temp_row, thresholds, seed_settings = cat.process_seed(seed, rng)
        temp_rows.append(temp_row)
        threshold_rows += thresholds
        settings[str(seed)] = seed_settings
    temp_table, threshold_table = pd.DataFrame(temp_rows), pd.DataFrame(threshold_rows)
    temp_table.to_csv(RESULTS_DIR / "temperature_scaling.csv", index=False, lineterminator="\n")
    threshold_table.to_csv(RESULTS_DIR / "calibration_thresholds.csv", index=False, lineterminator="\n")
    settings_file.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(cat.write_summary(temp_table, threshold_table))


if __name__ == "__main__":
    main()
