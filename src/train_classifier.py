"""
train_classifier.py  -  Step 4 of the plan: trains the DINOv3 disease classifier for one seed.

The recipe is the original one, run with the ORIGINAL code: set_seed, class weights, train_one_epoch,
evaluate, fit_phase (early stopping and best-version selection), save_checkpoint and print_confusion
are imported from the frozen project's src/train_classifier.py; the data come from its dataset.py
(same split file, same transforms, same seeded loader); the hyper-parameters come from its config.py.
Only the model differs (plan section 3): DINOv3 ViT-S/16, CLS token -> dropout 0.20 -> Linear(384, 3);
phase B trains the last 3 transformer blocks with every LayerNorm frozen. Full precision, batch 32
(fit test passed in fp32).

Opens s_train and s_val only.

OUTPUT
  models/dino_seed<S>_phaseA.pt, models/dino_seed<S>_phaseB.pt     (not tracked by Git)
  results/dino_seed<S>_training_log.csv, results/dino_seed<S>_run_record.json
RUN
  venv/Scripts/python.exe src/train_classifier.py --seed 42
"""

import argparse
import json
import platform
import sys

from common import (MODELS_DIR, RESULTS_DIR, DinoClassifier, count_parameters, sha256,
                    use_frozen_code, verify_split, HUB_REPO, HUB_REVISION, WEIGHTS_SHA256)

use_frozen_code()

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402
import sklearn              # noqa: E402
import timm                 # noqa: E402
import torch                # noqa: E402
import torchvision          # noqa: E402

import train_classifier as original   # noqa: E402  (frozen project, read-only)
from config import (BATCH_SIZE, CLASS_NAMES, EVAL_BATCH_SIZE, PHASE_A_LR, PHASE_A_MAX_EPOCHS,  # noqa: E402
                    PHASE_A_PATIENCE, PHASE_B_LR, PHASE_B_MAX_EPOCHS, PHASE_B_PATIENCE,
                    PHASE_B_WEIGHT_DECAY)
from dataset import SourceDataset, make_loader   # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    path_a = MODELS_DIR / f"dino_seed{args.seed}_phaseA.pt"
    path_b = MODELS_DIR / f"dino_seed{args.seed}_phaseB.pt"
    log_path = RESULTS_DIR / f"dino_seed{args.seed}_training_log.csv"
    record_path = RESULTS_DIR / f"dino_seed{args.seed}_run_record.json"
    for path in (path_a, path_b, record_path):
        if path.exists():
            sys.exit(f"{path} already exists; this run is done once per seed.")

    split_sha = verify_split()
    device = torch.device("cuda")
    original.set_seed(args.seed)

    train_set = SourceDataset("s_train", augment=True)
    val_set = SourceDataset("s_val")
    train_loader = make_loader(train_set, BATCH_SIZE, shuffle=True, num_workers=0, seed=args.seed)
    val_loader = make_loader(val_set, EVAL_BATCH_SIZE, shuffle=False, num_workers=0)
    weights, counts = original.class_weights_from_labels(train_set.labels, device)
    weights_list = [round(float(w), 6) for w in weights.cpu()]
    print(f"Seed {args.seed}; s_train {len(train_set)} {counts.astype(int).tolist()}; s_val {len(val_set)}; "
          f"class weights {weights_list}")

    model = DinoClassifier().to(device)
    total, _ = count_parameters(model)
    log_rows, results = [], {}

    print("ROUND 1 (phase A): output layer only")
    model.set_phase_a()
    _, trainable_a = count_parameters(model)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=PHASE_A_LR)
    results["phase_a"], _ = original.fit_phase("A", model, optimizer, train_loader, val_loader, weights,
                                               device, PHASE_A_MAX_EPOCHS, PHASE_A_PATIENCE, "loss", log_rows)
    results["phase_a"]["trainable_parameters"] = trainable_a
    original.save_checkpoint(path_a, model, args.seed, "A", results["phase_a"], weights_list)

    print("ROUND 2 (phase B): last 3 blocks + output layer, LayerNorms frozen")
    model.set_phase_b()
    _, trainable_b = count_parameters(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=PHASE_B_LR, weight_decay=PHASE_B_WEIGHT_DECAY)
    results["phase_b"], final_metrics = original.fit_phase(
        "B", model, optimizer, train_loader, val_loader, weights, device,
        PHASE_B_MAX_EPOCHS, PHASE_B_PATIENCE, "macro_f1", log_rows)
    results["phase_b"]["trainable_parameters"] = trainable_b
    original.save_checkpoint(path_b, model, args.seed, "B", results["phase_b"], weights_list)
    original.print_confusion(final_metrics)

    pd.DataFrame(log_rows).to_csv(log_path, index=False, lineterminator="\n")
    record = {
        "seed": args.seed, "backbone": {"repo": HUB_REPO, "revision": HUB_REVISION, "weights_sha256": WEIGHTS_SHA256},
        "gpu": torch.cuda.get_device_name(0),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
        "precision": "fp32", "batch_size": BATCH_SIZE, "eval_batch_size": EVAL_BATCH_SIZE,
        "class_weights": dict(zip(CLASS_NAMES, weights_list)),
        "train_images": len(train_set), "val_images": len(val_set),
        "parameters_total": total, "split_sha256": split_sha,
        "checkpoint_sha256": {"phaseA": sha256(path_a), "phaseB": sha256(path_b)},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "timm": timm.__version__,
                     "scikit_learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "results": results,
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Saved {path_b.name}, {log_path.name}, {record_path.name}. Only s_train and s_val were opened.")


if __name__ == "__main__":
    main()
