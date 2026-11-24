"""
freeze.py  -  Step 7 of the plan: hash every file the evaluations depend on, before any S-test,
PP2020 or PlantDoc result exists (as the frozen freeze_system.py).

Recorded: the original split file and target manifest (read-only), the DINOv3 weight file, the three
classifiers, the nine failure heads, the S-calibration outputs, the calibration settings, this folder's
code, and the frozen original code it imports.

RUN   venv/Scripts/python.exe src/freeze.py            create the record (once)
      venv/Scripts/python.exe src/freeze.py --verify   check nothing changed
OUTPUT results/frozen_system_manifest.json (+ .sha256)
"""

import argparse
import datetime
import json
import platform
import subprocess
import sys

from common import (FROZEN_ROOT, HUB_REPO, HUB_REVISION, MODELS_DIR, RESULTS_DIR, ROOT, SEEDS, sha256)

FREEZE_FILE = RESULTS_DIR / "frozen_system_manifest.json"
FREEZE_SIDECAR = RESULTS_DIR / "frozen_system_manifest.json.sha256"
HEAD_VARIANTS = ("proposed", "logits_only", "features_only")
METHODS = ["B1 MSP", "B2 Entropy", "B3 Margin", "B4 Temp-scaled MSP", "B5 Logits-only head",
           "A1 Features-only head", "P1 Proposed head"]
OWN_CODE = ["common.py", "train_classifier.py", "prepare_reliability_views.py", "train_failure_heads.py",
            "calibrate.py", "freeze.py"]
FROZEN_CODE = ["config.py", "dataset.py", "train_classifier.py", "prepare_reliability_data_v2.py",
               "failure_head_diagnostic.py", "train_failure_head.py", "retrain_failure_head.py",
               "calibrate_and_thresholds.py", "frozen_pipeline.py", "metrics.py"]


def weights_path():
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from huggingface_hub import hf_hub_download
    from pathlib import Path
    return Path(hf_hub_download(HUB_REPO, "model.safetensors", revision=HUB_REVISION))


def files():
    """{label: path}. Labels starting with 'frozen:' are files of the original project."""
    out = {"frozen:splits/split_assignments.csv": FROZEN_ROOT / "splits" / "split_assignments.csv",
           "frozen:manifests/manifest.csv": FROZEN_ROOT / "manifests" / "manifest.csv",
           "backbone:model.safetensors": weights_path()}
    for name in ("frozen_calibration_settings.json", "temperature_scaling.csv", "calibration_thresholds.csv"):
        out[f"results/{name}"] = RESULTS_DIR / name
    for seed in SEEDS:
        out[f"models/dino_seed{seed}_phaseB.pt"] = MODELS_DIR / f"dino_seed{seed}_phaseB.pt"
        for v in HEAD_VARIANTS:
            out[f"models/failure_head_{v}_seed{seed}.pt"] = MODELS_DIR / f"failure_head_{v}_seed{seed}.pt"
        out[f"results/s_calibration_outputs_seed{seed}.csv"] = RESULTS_DIR / f"s_calibration_outputs_seed{seed}.csv"
    for name in OWN_CODE:
        out[f"src/{name}"] = ROOT / "src" / name
    for name in FROZEN_CODE:
        out[f"frozen:src/{name}"] = FROZEN_ROOT / "src" / name
    return out


def verify_frozen():
    recorded = FREEZE_SIDECAR.read_text(encoding="utf-8").split()[0]
    if sha256(FREEZE_FILE) != recorded:
        sys.exit("The freeze record itself has changed.")
    record = json.loads(FREEZE_FILE.read_text(encoding="utf-8"))
    current = files()
    diffs = [label for label, h in record["files"].items()
             if label not in current or not current[label].is_file() or sha256(current[label]) != h]
    if diffs:
        sys.exit("The frozen system has changed:\n  " + "\n  ".join(diffs))
    return record


def git_commit():
    run = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    return {"commit": run.stdout.strip(), "uncommitted_file_count": len(status.stdout.splitlines())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        record = verify_frozen()
        print(f"All {len(record['files'])} frozen files unchanged.")
        return
    if FREEZE_FILE.exists():
        sys.exit("Freeze record already exists.")
    problems = [label for label, p in files().items() if not p.is_file()]
    settings = json.loads((RESULTS_DIR / "frozen_calibration_settings.json").read_text(encoding="utf-8"))
    for seed in SEEDS:
        for m in METHODS:
            if set(settings[str(seed)]["thresholds"].get(m, {})) != {"tau80", "tau90"}:
                problems.append(f"settings: seed {seed} {m}")
    for pattern in ("s_test_*", "target_*", "pp2020_*", "plantdoc_*"):
        problems += [f"result exists before freeze: {p.name}" for p in RESULTS_DIR.glob(pattern)]
    if problems:
        sys.exit("Cannot freeze:\n  " + "\n  ".join(problems))
    import numpy, pandas, sklearn, timm, torch, torchvision, PIL   # noqa: E401
    record = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "purpose": "Freeze of the DINOv3 system before any S-test, PP2020 or PlantDoc evaluation",
        "classifier_seeds": list(SEEDS), "backbone": {"repo": HUB_REPO, "revision": HUB_REVISION},
        "git": git_commit(),
        "libraries": {"python": platform.python_version(), "torch": torch.__version__,
                      "torchvision": torchvision.__version__, "timm": timm.__version__, "numpy": numpy.__version__,
                      "pandas": pandas.__version__, "sklearn": sklearn.__version__, "PIL": PIL.__version__},
        "files": {label: sha256(p) for label, p in files().items()},
    }
    FREEZE_FILE.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    FREEZE_SIDECAR.write_text(f"{sha256(FREEZE_FILE)}  {FREEZE_FILE.name}\n", encoding="utf-8", newline="\n")
    print(f"Frozen {len(record['files'])} files at commit {record['git']['commit']}.")
    print(f"Freeze record SHA-256: {sha256(FREEZE_FILE)}")


if __name__ == "__main__":
    main()
