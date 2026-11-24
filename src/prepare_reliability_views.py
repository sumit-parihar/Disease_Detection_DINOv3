"""
prepare_reliability_views.py  -  Step 5a of the plan: the 13 failure views per S-reliability image,
run through the three DINOv3 phase-B classifiers.

The views are made by the ORIGINAL code (create_views, verify_original_split and
load_existing_secondary_split imported read-only from the frozen prepare_reliability_data_v2.py),
with the same root seed 42, so every view is the same image as in the original study. This is checked:
the view manifest built here must be byte-identical to the frozen manifests/reliability_views_manifest_v2.csv.
The view definitions and severity are never changed; fewer than 50 training failures is reported only.

Opens S-reliability images only.

OUTPUT
  results/reliability_features_seed<S>.npz        features (384), logits, labels, view metadata (not tracked)
  results/reliability_predictions_seed<S>.csv     one row per view
  results/reliability_generation_summary.txt      failure counts per seed and view type
"""

import io
import sys

import numpy as np
import pandas as pd

from common import FROZEN_ROOT, MODELS_DIR, RESULTS_DIR, SEEDS, DinoClassifier, sha256, use_frozen_code

use_frozen_code()

import torch   # noqa: E402

import prepare_reliability_data_v2 as v2   # noqa: E402  (frozen project, read-only)
from config import CLASS_NAMES, CLASS_TO_INDEX   # noqa: E402

FEATURE_DIM = 384
ROOT_SEED = 42
FROZEN_MANIFEST = FROZEN_ROOT / "manifests" / "reliability_views_manifest_v2.csv"
SUMMARY_FILE = RESULTS_DIR / "reliability_generation_summary.txt"


def feature_file(seed):
    return RESULTS_DIR / f"reliability_features_seed{seed}.npz"


def prediction_file(seed):
    return RESULTS_DIR / f"reliability_predictions_seed{seed}.csv"


def load_phase_b(seed, device):
    path = MODELS_DIR / f"dino_seed{seed}_phaseB.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("seed") != seed or checkpoint.get("phase") != "B" or checkpoint.get("class_names") != CLASS_NAMES:
        sys.exit(f"{path.name}: metadata mismatch")
    model = DinoClassifier()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model.to(device)


def main():
    outputs = [SUMMARY_FILE] + [feature_file(s) for s in SEEDS] + [prediction_file(s) for s in SEEDS]
    if any(p.exists() for p in outputs):
        sys.exit("Outputs already exist; this step runs once.")

    assignments, split_hash = v2.verify_original_split()
    secondary, secondary_hash = v2.load_existing_secondary_split(assignments)
    ordered = secondary.sort_values("path").reset_index(drop=True)
    device = torch.device("cuda")
    models = {seed: load_phase_b(seed, device) for seed in SEEDS}

    metadata, true_idx = [], []
    feats = {s: [] for s in SEEDS}
    logits = {s: [] for s in SEEDS}
    with torch.inference_mode():
        for i, row in enumerate(ordered.itertuples(index=False), start=1):
            views, meta = v2.create_views(row.path, row.label, row.group_id, row.reliability_split, ROOT_SEED)
            batch = torch.stack(views).to(device)
            metadata.extend(meta)
            true_idx.extend([CLASS_TO_INDEX[row.label]] * len(views))
            for seed, model in models.items():
                f = model.extract_features(batch)
                z = model.fc(model.dropout(f))
                if not (torch.isfinite(f).all() and torch.isfinite(z).all()):
                    sys.exit(f"Non-finite output for {row.path}, seed {seed}")
                feats[seed].append(f.cpu().numpy().astype(np.float32))
                logits[seed].append(z.cpu().numpy().astype(np.float32))
            if i % 50 == 0 or i == len(ordered):
                print(f"  {i}/{len(ordered)} originals")

    manifest = pd.DataFrame(metadata)
    buffer = io.StringIO()
    manifest.to_csv(buffer, index=False, lineterminator="\n")
    identical = buffer.getvalue().encode("utf-8") == FROZEN_MANIFEST.read_bytes()
    if not identical:
        sys.exit("View manifest differs from the frozen v2 manifest; stopping.")

    true = np.asarray(true_idx, dtype=np.int64)
    lines = ["DINOv3 RELIABILITY VIEWS (same 13 views per image as the original study)", "=" * 70, "",
             f"Original S-reliability images: {len(ordered)}; view records: {len(manifest)}",
             f"Split SHA-256: {split_hash}", f"Reliability train/val split SHA-256: {secondary_hash}",
             f"View manifest byte-identical to frozen {FROZEN_MANIFEST.name} "
             f"(SHA-256 {sha256(FROZEN_MANIFEST)}): YES", ""]
    for seed in SEEDS:
        F, Z = np.concatenate(feats[seed]), np.concatenate(logits[seed])
        pred = Z.argmax(axis=1).astype(np.int64)
        fail = (pred != true).astype(np.int8)
        assert F.shape == (len(manifest), FEATURE_DIM) and Z.shape == (len(manifest), 3)
        table = manifest.copy()
        table["true_label_index"] = true
        table["predicted_label_index"] = pred
        table["predicted_label"] = [CLASS_NAMES[i] for i in pred]
        table["failure_label"] = fail
        table["logit_healthy"], table["logit_scab"], table["logit_rust"] = Z[:, 0], Z[:, 1], Z[:, 2]
        table["classifier_seed"] = seed
        table.to_csv(prediction_file(seed), index=False, lineterminator="\n")
        np.savez_compressed(
            feature_file(seed), features=F, logits=Z, true_label_index=true, predicted_label_index=pred,
            failure_label=fail, view_ids=manifest["view_id"].to_numpy(dtype="U20"),
            paths=manifest["path"].to_numpy(dtype=str), group_ids=manifest["group_id"].to_numpy(dtype=str),
            reliability_split=manifest["reliability_split"].to_numpy(dtype=str),
            view_types=manifest["view_type"].to_numpy(dtype=str),
            replicate_index=manifest["replicate_index"].to_numpy(dtype=np.int64),
            classifier_seed=np.asarray(seed, dtype=np.int64))

        lines.append(f"Seed {seed}:")
        split = table["reliability_split"].to_numpy()
        for part in ("reliability_train", "reliability_val"):
            mask = split == part
            pos = int(fail[mask].sum())
            originals = int(table.loc[mask & (fail == 1), "path"].nunique())
            lines.append(f"  {part}: views={int(mask.sum())}, positive failures={pos}, "
                         f"non-failures={int(mask.sum()) - pos}, originals with >=1 failure={originals}")
            if part == "reliability_train" and pos < v2.MIN_POSITIVE_TRAIN_FAILURES:
                lines.append(f"  NOTE: fewer than {v2.MIN_POSITIVE_TRAIN_FAILURES} positive training views. "
                             "Views and severity are NOT changed (plan section 3); reported as a limitation.")
        lines.append("  Failure counts by view type (train / validation):")
        for view_type in v2.VIEW_TYPES:
            counts = []
            for part in ("reliability_train", "reliability_val"):
                mask = (table["view_type"].to_numpy() == view_type) & (split == part)
                counts.append(f"{int(fail[mask].sum())}/{int(mask.sum())}")
            lines.append(f"    {view_type}: {counts[0]} / {counts[1]}")
        lines.append("")
    SUMMARY_FILE.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
