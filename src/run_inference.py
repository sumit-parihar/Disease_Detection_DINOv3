"""
run_inference.py  -  Step 9a of the plan: the ONE-TIME inference of the frozen DINOv3 system on
S-test, PP2020 and PlantDoc. No label is read.

Checks before running (any failure stops):
  1. no output exists yet (run once; no overwrite option);
  2. the sealed pre-registration has its recorded SHA-256;
  3. this folder's freeze record verifies (freeze.py --verify);
  4. the label-free manifests match their recorded hashes, and every image matches its own hash
     (PP2020: original project's target_inference_manifest.csv, 1,730 images;
      PlantDoc: extended project's plantdoc_inference_manifest.csv, 265 images).
     S-test: paths from the hash-verified split file (subset s_test); the label column is not read.

Every image goes through each frozen classifier and its three heads with the ORIGINAL preparation
(dataset.build_eval_transform), the ORIGINAL frozen_pipeline.run_images and the ORIGINAL
run_target_inference.build_table (MSP, entropy, margin, temperature-scaled MSP with the frozen
temperature, three heads), plus energy_score = -logsumexp(logits) as in the extended study.

OUTPUT results/inference/<dataset>_predictions_seed<S>.csv (no labels),
       results/inference/<dataset>_features_seed<S>.npz (not tracked), results/inference/inference_hashes.txt
NEXT: commit and push the prediction files and hash file BEFORE any label is joined.
"""

import json
import sys

import numpy as np
import pandas as pd

from common import EXTENDED_ROOT, FROZEN_ROOT, RESULTS_DIR, ROOT, SEEDS, sha256, use_frozen_code, verify_split

use_frozen_code()

import torch   # noqa: E402
from PIL import Image   # noqa: E402
from torch.utils.data import DataLoader, Dataset   # noqa: E402

from calibrate import load_classifier, load_head, HEAD_VARIANTS   # noqa: E402
from dataset import build_eval_transform   # noqa: E402  (original, read-only)
from freeze import FREEZE_FILE, verify_frozen   # noqa: E402
from frozen_pipeline import run_images   # noqa: E402
from run_target_inference import build_table   # noqa: E402

PREREG = ROOT / "prereg" / "dinov3_preregistration.md"
PREREG_SHA = "TO BE SET AFTER SEALING"
BATCH = 32
OUT = RESULTS_DIR / "inference"
DATASETS = {
    "pp2020": {"root": FROZEN_ROOT, "manifest": FROZEN_ROOT / "manifests" / "target_inference_manifest.csv",
               "manifest_sha": "69dc708e9621ecd5adecafa1bc9836fe69a13d873015717c49ab55ae18929cac", "images": 1730},
    "plantdoc": {"root": EXTENDED_ROOT, "manifest": EXTENDED_ROOT / "manifests" / "plantdoc_inference_manifest.csv",
                 "manifest_sha": "66c43f19417c3fa39588ef88b13f1988d8a93e3e06d17931991615ea23ffe149", "images": 265},
}


class LabelFreeImages(Dataset):
    def __init__(self, root, paths):
        self.root, self.paths = root, list(paths)
        self.transform = build_eval_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.root / self.paths[index]) as image:
            return self.transform(image.convert("RGB")), self.paths[index]


def s_test_manifest():
    split = pd.read_csv(FROZEN_ROOT / "splits" / "split_assignments.csv", keep_default_na=False)
    part = split[split["subset"] == "s_test"].sort_values("path")
    return pd.DataFrame({"path": part["path"].to_numpy(), "cluster_id": part["group_id"].to_numpy()})


def manifests():
    out = {"s_test": (FROZEN_ROOT, s_test_manifest())}
    for name, spec in DATASETS.items():
        if sha256(spec["manifest"]) != spec["manifest_sha"]:
            sys.exit(f"{name}: label-free manifest has changed.")
        table = pd.read_csv(spec["manifest"], keep_default_na=False)
        if list(table.columns) != ["path", "sha256", "cluster_id", "cluster_size"] or len(table) != spec["images"]:
            sys.exit(f"{name}: unexpected manifest layout.")
        for path, recorded in zip(table["path"], table["sha256"]):
            if sha256(spec["root"] / path) != recorded:
                sys.exit(f"{name}: image differs from the manifest: {path}")
        out[name] = (spec["root"], table)
        print(f"{name}: {len(table)} image hashes verified.")
    return out


def energy(logits):
    logits = logits.astype(np.float64)
    top = logits.max(axis=1)
    return -(top + np.log(np.exp(logits - top[:, None]).sum(axis=1)))


def main():
    if OUT.exists() and any(OUT.iterdir()):
        sys.exit("Inference outputs already exist. Inference is run once.")
    if sha256(PREREG) != PREREG_SHA:
        sys.exit("The pre-registration differs from the sealed version.")
    verify_frozen()
    verify_split()
    print("Pre-registration, freeze record and split verified. No label is read.")
    sets = manifests()
    OUT.mkdir(parents=True, exist_ok=True)
    settings = json.loads((RESULTS_DIR / "frozen_calibration_settings.json").read_text(encoding="utf-8"))
    device = torch.device("cuda")

    written = []
    for seed in SEEDS:
        classifier = load_classifier(seed, device)
        heads = {v: load_head(v, seed, device) for v in HEAD_VARIANTS}
        for name, (root, table) in sets.items():
            loader = DataLoader(LabelFreeImages(root, table["path"]), batch_size=BATCH, shuffle=False, num_workers=0)
            out = run_images(loader, classifier, heads, device)
            for key, array in [("logits", out["logits"])] + list(out["head_scores"].items()):
                if not np.isfinite(array).all():
                    sys.exit(f"Non-finite {key} for {name}, seed {seed}.")
            predictions = build_table(out, table, float(settings[str(seed)]["temperature"]))
            predictions["energy_score"] = energy(out["logits"])
            csv = OUT / f"{name}_predictions_seed{seed}.csv"
            predictions.to_csv(csv, index=False, lineterminator="\n")
            np.savez(OUT / f"{name}_features_seed{seed}.npz", features=out["features"], logits=out["logits"],
                     paths=out["paths"])
            written.append(csv)
            print(f"  seed {seed} {name}: predicted counts {predictions['predicted_class'].value_counts().to_dict()}")

    lines = ["SHA-256 checksums for the one-time DINOv3 inference (no labels were read)", "", "INPUTS:",
             f"{sha256(PREREG)}  prereg/{PREREG.name}", f"{sha256(FREEZE_FILE)}  results/{FREEZE_FILE.name}"]
    lines += [f"{spec['manifest_sha']}  {name}: {spec['manifest'].name}" for name, spec in DATASETS.items()]
    lines += ["", "OUTPUT FILES:"] + [f"{sha256(p)}  {p.relative_to(ROOT).as_posix()}" for p in written]
    (OUT / "inference_hashes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("NEXT: commit and push results/inference/*.csv and inference_hashes.txt BEFORE any label is joined.")


if __name__ == "__main__":
    main()
